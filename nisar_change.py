#!/usr/bin/env python3
"""
nisar_change.py - find, inspect and compare NISAR GCOV scenes, then export
web-ready layers for the viewer in ./web.

Commands
  search    list NISAR GCOV granules over a point, grouped by track and frame
  download  download granule URLs (needs an Earthdata token)
  inspect   print the layout of a NISAR HDF5 file
  change    compare two GCOV files and write PNG layers + meta.json
  demo      build two synthetic scenes and run `change` on them (no download)
  serve     start a local web server for the viewer

Typical flow with real data
  python nisar_change.py search --lon 90.41 --lat 23.81 --start 2025-11-01
  python nisar_change.py download URL_BEFORE URL_AFTER --out data/real
  python nisar_change.py inspect data/real/FIRST_FILE.h5
  python nisar_change.py change --before data/real/A.h5 --after data/real/B.h5 \
        --pol HH --bbox 90.2 23.6 90.6 24.0
  python nisar_change.py serve
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import http.server
import json
import os
import re
import socketserver
import sys
import time
import webbrowser
import zlib
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent
WEB_DATA = ROOT / "web" / "data"

# Pixels darker than a cutoff (in dB) count as open water. Calm water is very dark to radar, but wind and
# current make real rivers brighter, so a fixed number such as -20 dB can miss them. By default the cutoff
# is picked from the data (Otsu) and kept inside this range. Use --threshold to set it yourself.
AUTO_RANGE = (-26.0, -15.0)

TIMESTAMP = re.compile(r"(\d{8}T\d{6})")

# Class codes in the change map
NEW_WATER, RECEDED_WATER, PERMANENT_WATER, NODATA = 1, 2, 3, 255


# ---------------------------------------------------------------------------
# Reading NISAR GCOV files
# ---------------------------------------------------------------------------

def acquisition_time(path: Path) -> dt.datetime | None:
    """NISAR file names carry the acquisition start as YYYYMMDDTHHMMSS."""
    m = TIMESTAMP.search(path.name)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)


def locate_band(h5: h5py.File, pol: str) -> h5py.Dataset:
    """Find the covariance term for a polarization. HH -> 'HHHH', HV -> 'HVHV'."""
    term = pol.upper() * 2
    found: list[h5py.Dataset] = []
    seen_terms: set[str] = set()

    def visit(name: str, obj) -> None:
        base = name.rsplit("/", 1)[-1]
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
            if base in ("HHHH", "HVHV", "VHVH", "VVVV", "RHRH", "RVRV"):
                seen_terms.add(base)
            if base == term:
                found.append(obj)

    h5.visititems(visit)
    if not found:
        have = ", ".join(sorted(seen_terms)) or "none"
        raise SystemExit(
            f"No {term} layer in this file (layers found: {have}). "
            "Pass a different --pol, or run `inspect` to see the file layout."
        )
    found.sort(key=lambda d: ("frequencyA" not in d.name, d.name))
    return found[0]


def read_epsg(group: h5py.Group) -> int:
    if "projection" in group:
        ds = group["projection"]
        for key in ("epsg_code", "epsg"):
            if key in ds.attrs:
                return int(np.asarray(ds.attrs[key]).ravel()[0])
        try:
            value = int(np.asarray(ds[()]).ravel()[0])
            if 1000 < value < 40000:
                return value
        except Exception:
            pass
    raise SystemExit("Could not read the map projection (EPSG code) from this file.")


def window_from_bbox(xs, ys, epsg, bbox, looks):
    """Convert a lon/lat box to row and column ranges, snapped to the multilook grid."""
    from rasterio.warp import transform

    west, south, east, north = bbox
    px, py = transform("EPSG:4326", f"EPSG:{epsg}", [west, east, west, east], [south, south, north, north])
    cols = np.where((xs >= min(px)) & (xs <= max(px)))[0]
    rows = np.where((ys >= min(py)) & (ys <= max(py)))[0]
    if cols.size == 0 or rows.size == 0:
        raise SystemExit("The bounding box does not overlap this scene.")
    c0 = int(cols.min()) - int(cols.min()) % looks
    r0 = int(rows.min()) - int(rows.min()) % looks
    return r0, int(rows.max()) + 1, c0, int(cols.max()) + 1


def multilook_power(ds: h5py.Dataset, window, looks: int, strip_rows: int = 64) -> np.ndarray:
    """Average looks x looks blocks of linear power, reading in strips to save memory."""
    r0, r1, c0, c1 = window
    oy, ox = (r1 - r0) // looks, (c1 - c0) // looks
    if oy < 2 or ox < 2:
        raise SystemExit("The area is too small for this --looks value. Use a lower --looks or a larger --bbox.")
    out = np.full((oy, ox), np.nan, dtype=np.float32)
    min_valid = max(1, looks * looks // 2)
    for s in range(0, oy, strip_rows):
        e = min(s + strip_rows, oy)
        block = ds[r0 + s * looks : r0 + e * looks, c0 : c0 + ox * looks].astype(np.float32)
        valid = np.isfinite(block) & (block > 0)
        block = np.where(valid, block, 0.0)
        total = block.reshape(e - s, looks, ox, looks).sum(axis=(1, 3))
        count = valid.reshape(e - s, looks, ox, looks).sum(axis=(1, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[s:e] = np.where(count >= min_valid, total / count, np.nan)
    return out


class Scene:
    def __init__(self, path, when, pol, epsg, x, y, db):
        self.path, self.when, self.pol, self.epsg = path, when, pol, epsg
        self.x, self.y, self.db = x, y, db


def load_scene(path: Path, pol: str, looks: int, bbox) -> Scene:
    with h5py.File(path, "r") as h5:
        ds = locate_band(h5, pol)
        grp = ds.parent
        xs = np.asarray(grp["xCoordinates"][:], dtype=np.float64)
        ys = np.asarray(grp["yCoordinates"][:], dtype=np.float64)
        epsg = read_epsg(grp)
        if bbox:
            window = window_from_bbox(xs, ys, epsg, bbox, looks)
        else:
            window = (0, ds.shape[0], 0, ds.shape[1])
        power = multilook_power(ds, window, looks)

    r0, _, c0, _ = window
    oy, ox = power.shape
    x = xs[c0 : c0 + ox * looks].reshape(ox, looks).mean(axis=1)
    y = ys[r0 : r0 + oy * looks].reshape(oy, looks).mean(axis=1)
    # Normalise orientation: x increasing, y decreasing (north at the top)
    if x[1] < x[0]:
        x, power = x[::-1], power[:, ::-1]
    if y[1] > y[0]:
        y, power = y[::-1], power[::-1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.where(power > 0, 10.0 * np.log10(power), np.nan).astype(np.float32)
    return Scene(path, acquisition_time(path), pol, epsg, x, y, db)


def align(a: Scene, b: Scene):
    """Crop two scenes to their shared area. Same track and frame means the same grid."""
    if a.epsg != b.epsg:
        raise SystemExit(f"The scenes use different projections (EPSG:{a.epsg} vs EPSG:{b.epsg}). "
                         "Pick two scenes from the same track and frame.")
    dx, dy = a.x[1] - a.x[0], abs(a.y[1] - a.y[0])
    if abs((b.x[1] - b.x[0]) - dx) > 0.01 * dx or abs(abs(b.y[1] - b.y[0]) - dy) > 0.01 * dy:
        raise SystemExit("The scenes have different pixel sizes. Use the same product type and --looks for both.")
    left, right = max(a.x[0], b.x[0]), min(a.x[-1], b.x[-1])
    top, bottom = min(a.y[0], b.y[0]), max(a.y[-1], b.y[-1])
    if right <= left or top <= bottom:
        raise SystemExit("The scenes do not overlap. Pick two scenes from the same track and frame.")

    def crop(s: Scene):
        ix, iy = (left - s.x[0]) / dx, (s.y[0] - top) / dy
        if abs(ix - round(ix)) > 0.25 or abs(iy - round(iy)) > 0.25:
            print("  warning: the scenes are offset by a fraction of a pixel; "
                  "results near edges may be slightly off.")
        ix, iy = int(round(ix)), int(round(iy))
        return s.db[iy:, ix:], s.x[ix:], s.y[iy:]

    (da, xa, ya), (db_, _, _) = crop(a), crop(b)
    h, w = min(da.shape[0], db_.shape[0]), min(da.shape[1], db_.shape[1])
    right_idx = int(np.searchsorted(xa, right)) + 1
    w = min(w, right_idx)
    return da[:h, :w], db_[:h, :w], xa[:w], ya[:h]


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def smooth_db(db: np.ndarray, size: int = 3) -> np.ndarray:
    """Light speckle smoothing that ignores empty pixels."""
    valid = np.isfinite(db)
    num = ndi.uniform_filter(np.where(valid, db, 0.0).astype(np.float32), size, mode="nearest")
    den = ndi.uniform_filter(valid.astype(np.float32), size, mode="nearest")
    return np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)


def otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    cum = np.cumsum(hist * centers)
    m0 = cum / np.maximum(w0, 1)
    m1 = (cum[-1] - cum) / np.maximum(w1, 1)
    return float(centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def auto_threshold(b_s: np.ndarray, a_s: np.ndarray):
    """Pick the water cutoff from both dates together. Returns (cutoff, raw value before limiting)."""
    both = np.concatenate([b_s[np.isfinite(b_s)], a_s[np.isfinite(a_s)]])
    raw = otsu(np.clip(both, -40.0, 5.0))
    lo, hi = AUTO_RANGE
    return float(min(max(raw, lo), hi)), float(raw)


def report_backscatter(b_s, a_s, thr, raw, mode, pol):
    """Print what the radar values look like, so a bad water cutoff is easy to spot."""
    print(f"\nBackscatter check ({pol}, in dB):")
    shares = {}
    for label, arr in (("before", b_s), ("after ", a_s)):
        v = arr[np.isfinite(arr)]
        p = np.percentile(v, [1, 5, 25, 50, 75, 95])
        shares[label.strip()] = float(np.mean(v < thr))
        print(f"  {label} 1%={p[0]:6.1f}  5%={p[1]:6.1f}  25%={p[2]:6.1f}  median={p[3]:6.1f}  "
              f"75%={p[4]:6.1f}  95%={p[5]:6.1f}")
    both = np.concatenate([b_s[np.isfinite(b_s)], a_s[np.isfinite(a_s)]])
    edges = np.arange(-35.0, 0.01, 2.5)
    hist, _ = np.histogram(np.clip(both, -34.99, -0.01), bins=edges)
    pct = hist / max(both.size, 1) * 100
    top = max(float(pct.max()), 1e-9)
    print("  how the pixels are spread (both dates together):")
    for i, share in enumerate(pct):
        lo, hi = edges[i], edges[i + 1]
        mark = "   <-- water cutoff" if lo <= thr < hi else ""
        print(f"   {lo:6.1f} to {hi:6.1f}  {'#' * int(round(share / top * 30)):<30} {share:5.1f}%{mark}")
    how = "picked automatically" if mode == "auto" else "set by you"
    note = ""
    if mode == "auto" and abs(raw - thr) > 0.05:
        note = f" (the histogram suggested {raw:.1f} dB; it was held inside the safe range)"
    print(f"  water cutoff: {thr:.1f} dB, {how}{note}")
    print(f"  pixels darker than the cutoff: before {shares['before'] * 100:.1f}%, after {shares['after'] * 100:.1f}%")
    if min(shares.values()) < 0.005:
        print("  WARNING: almost nothing is darker than the cutoff on at least one date. The water may be\n"
              "  brighter than the cutoff (rough river water often is). Try --threshold -17 or -15,\n"
              "  or compare with --pol HV.")
    elif max(shares.values()) > 0.5:
        print("  WARNING: more than half of the pixels count as water. The cutoff is probably too high.\n"
              "  Try a lower one such as --threshold -20.")


def drop_small(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return mask
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    keep = np.bincount(labels.ravel()) >= min_pixels
    keep[0] = False
    return keep[labels]


def classify(before_db, after_db, threshold, min_drop, min_blob):
    valid = np.isfinite(before_db) & np.isfinite(after_db)
    diff = after_db - before_db
    water_before = before_db < threshold
    water_after = after_db < threshold
    cls = np.zeros(before_db.shape, dtype=np.uint8)
    cls[valid & water_before & water_after] = PERMANENT_WATER
    new = drop_small(valid & ~water_before & water_after & (diff <= -min_drop), min_blob)
    gone = drop_small(valid & water_before & ~water_after & (diff >= min_drop), min_blob)
    cls[new] = NEW_WATER
    cls[gone] = RECEDED_WATER
    cls[~valid] = NODATA
    return cls, diff


# ---------------------------------------------------------------------------
# Export for the web viewer
# ---------------------------------------------------------------------------

def to_wgs84(arr, x, y, epsg, resampling, nodata, dtype):
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform, reproject

    dx, dy = x[1] - x[0], y[1] - y[0]
    src_t = Affine(dx, 0, x[0] - dx / 2, 0, dy, y[0] - dy / 2)
    h, w = arr.shape
    src_crs = CRS.from_epsg(epsg)
    left, top = src_t * (0, 0)
    right, bottom = src_t * (w, h)
    dst_t, dw, dh = calculate_default_transform(src_crs, "EPSG:4326", w, h,
                                                left=left, bottom=bottom, right=right, top=top)
    dst = np.full((dh, dw), nodata, dtype=dtype)
    reproject(source=arr.astype(dtype), destination=dst, src_transform=src_t, src_crs=src_crs,
              dst_transform=dst_t, dst_crs="EPSG:4326", resampling=resampling,
              src_nodata=nodata, dst_nodata=nodata)
    west, north = dst_t * (0, 0)
    east, south = dst_t * (dw, dh)
    return dst, (south, west, north, east)


def save_png(rgba: np.ndarray, path: Path) -> None:
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def gray_rgba(db, lo, hi):
    t = np.nan_to_num(np.clip((db - lo) / (hi - lo), 0, 1), nan=0.0)
    g = (t * 255).astype(np.uint8)
    a = np.where(np.isfinite(db), 255, 0).astype(np.uint8)
    return np.dstack([g, g, g, a])


def class_rgba(cls):
    palette = {NEW_WATER: (0, 200, 255, 235), RECEDED_WATER: (255, 196, 0, 235), PERMANENT_WATER: (23, 48, 107, 170)}
    rgba = np.zeros(cls.shape + (4,), dtype=np.uint8)
    for code, colour in palette.items():
        rgba[cls == code] = colour
    return rgba


def diff_rgba(diff, floor=1.5, full=6.0):
    rgba = np.zeros(diff.shape + (4,), dtype=np.uint8)
    mag = np.abs(np.nan_to_num(diff, nan=0.0))
    strength = np.clip((mag - floor) / (full - floor), 0, 1)
    alpha = np.where(mag >= floor, 70 + 160 * strength, 0).astype(np.uint8)
    darker = np.nan_to_num(diff, nan=0.0) < 0
    rgba[darker] = (123, 31, 162, 0)
    rgba[~darker] = (230, 81, 0, 0)
    rgba[..., 3] = alpha
    return rgba


def iso(when: dt.datetime | None) -> str | None:
    return when.strftime("%Y-%m-%dT%H:%M:%SZ") if when else None


def parse_date(text: str | None) -> dt.datetime | None:
    if not text:
        return None
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def run_change(before, after, out_dir, pol, looks, threshold, min_drop, min_blob,
               bbox, title, synthetic=False, before_date=None, after_date=None):
    from rasterio.crs import CRS
    from rasterio.enums import Resampling

    before, after = Path(before), Path(after)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pol = pol.upper()

    print(f"Reading {before.name}")
    sb = load_scene(before, pol, looks, bbox)
    print(f"Reading {after.name}")
    sa = load_scene(after, pol, looks, bbox)
    if before_date:
        sb.when = parse_date(before_date)
    if after_date:
        sa.when = parse_date(after_date)
    if sb.when and sa.when and sb.when > sa.when:
        print("  note: the 'before' scene is later than the 'after' scene, so I swapped them.")
        sb, sa = sa, sb

    b_db, a_db, x, y = align(sb, sa)
    print(f"Common grid: {b_db.shape[1]} x {b_db.shape[0]} pixels")

    b_s, a_s = smooth_db(b_db), smooth_db(a_db)
    if threshold in (None, "auto"):
        mode = "auto"
        thr, raw = auto_threshold(b_s, a_s)
    else:
        mode, thr, raw = "fixed", float(threshold), None
    report_backscatter(b_s, a_s, thr, raw, mode, pol)
    cls, diff = classify(b_s, a_s, thr, min_drop, min_blob)

    # Areas are counted on the projected grid, where pixels have a known size in metres
    crs = CRS.from_epsg(sb.epsg)
    dx, dy = x[1] - x[0], abs(y[1] - y[0])
    pixel_km2 = (dx * dy) / 1e6 if crs.is_projected else None
    stats = {"pixel_km2": pixel_km2}
    if pixel_km2:
        stats.update(
            valid_km2=float(np.count_nonzero(cls != NODATA) * pixel_km2),
            new_water_km2=float(np.count_nonzero(cls == NEW_WATER) * pixel_km2),
            receded_water_km2=float(np.count_nonzero(cls == RECEDED_WATER) * pixel_km2),
            permanent_water_km2=float(np.count_nonzero(cls == PERMANENT_WATER) * pixel_km2),
        )
    stats["median_db_before"] = float(np.nanmedian(b_db))
    stats["median_db_after"] = float(np.nanmedian(a_db))

    lo, hi = np.nanpercentile(np.concatenate([b_db[np.isfinite(b_db)], a_db[np.isfinite(a_db)]]), [2, 98])

    bilinear = Resampling.bilinear
    wgs_before, bounds = to_wgs84(b_db, x, y, sb.epsg, bilinear, np.nan, np.float32)
    wgs_after, _ = to_wgs84(a_db, x, y, sb.epsg, bilinear, np.nan, np.float32)
    wgs_diff, _ = to_wgs84(diff, x, y, sb.epsg, bilinear, np.nan, np.float32)
    wgs_cls, _ = to_wgs84(cls, x, y, sb.epsg, Resampling.nearest, NODATA, np.uint8)

    save_png(gray_rgba(wgs_before, lo, hi), out / "before.png")
    save_png(gray_rgba(wgs_after, lo, hi), out / "after.png")
    save_png(class_rgba(wgs_cls), out / "change.png")
    save_png(diff_rgba(wgs_diff), out / "diff.png")

    south, west, north, east = bounds
    meta = {
        "title": title,
        "synthetic": synthetic,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S"),
        "bounds": [[south, west], [north, east]],
        "before": {"date": iso(sb.when), "file": sb.path.name},
        "after": {"date": iso(sa.when), "file": sa.path.name},
        "polarization": pol,
        "threshold_db": round(thr, 1),
        "threshold_mode": mode,
        "min_drop_db": min_drop,
        "looks": looks,
        "pixel_size_m": round(float(dx), 1),
        "gray_range_db": [round(float(lo), 1), round(float(hi), 1)],
        "stats": stats,
        "layers": {"before": "before.png", "after": "after.png", "change": "change.png", "diff": "diff.png"},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\nDone. Files written to", out)
    if pixel_km2:
        print(f"  imaged area      {stats['valid_km2']:.1f} km2")
        print(f"  new water        {stats['new_water_km2']:.1f} km2")
        print(f"  receded water    {stats['receded_water_km2']:.1f} km2")
        print(f"  permanent water  {stats['permanent_water_km2']:.1f} km2")
    return meta


# ---------------------------------------------------------------------------
# Synthetic demo scenes (same file layout as a NISAR GCOV product)
# ---------------------------------------------------------------------------

def make_demo_scenes(folder: Path, size=2400, spacing=20.0, seed=7):
    from rasterio.warp import transform

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    epsg = 32646  # UTM 46N, covers the Bengal delta
    cx, cy = transform("EPSG:4326", f"EPSG:{epsg}", [90.41], [23.81])
    cx, cy = cx[0], cy[0]
    xs = cx + (np.arange(size) - size / 2 + 0.5) * spacing
    ys = cy + (size / 2 - np.arange(size) - 0.5) * spacing

    def field(sigma):
        f = ndi.gaussian_filter(rng.standard_normal((size, size)).astype(np.float32), sigma)
        return (f - f.mean()) / f.std()

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    centre = size * 0.52 + 260 * np.sin(xx / size * 2 * np.pi * 1.6 + 0.6) + 90 * np.sin(xx / size * 2 * np.pi * 5.1)
    river = np.abs(yy - centre) < 16
    lake = ((xx - size * 0.2) / 150) ** 2 + ((yy - size * 0.2) / 90) ** 2 < 1
    pond = ((xx - size * 0.78) / 70) ** 2 + ((yy - size * 0.82) / 70) ** 2 < 1
    dist = ndi.distance_transform_edt(~river) * spacing
    flood = (dist < 4200) & (field(90) + dist / 3000 < 0.5) & ~river
    growth = field(60) > 0.8

    texture, water_texture = field(25), field(6)

    def write(name, water, regrown):
        hh = -10.5 + 1.6 * texture
        hv = -17.0 + 1.6 * texture
        hh = np.where(regrown & ~water, hh + 3.0, hh)
        hv = np.where(regrown & ~water, hv + 3.0, hv)
        hh = np.where(water, -25.0 + 0.8 * water_texture, hh)
        hv = np.where(water, -31.0 + 0.8 * water_texture, hv)
        local = np.random.default_rng(zlib.crc32(name.encode()))
        out = {}
        for key, db in (("HHHH", hh), ("HVHV", hv)):
            power = (10.0 ** (db / 10.0)) * local.exponential(1.0, size=db.shape)
            out[key] = power.astype(np.float32)
        path = folder / name
        with h5py.File(path, "w") as h5:
            grp = h5.create_group("science/LSAR/GCOV/grids/frequencyA")
            for key, arr in out.items():
                grp.create_dataset(key, data=arr, chunks=(256, 256), compression="gzip", compression_opts=1)
            grp.create_dataset("xCoordinates", data=xs)
            grp.create_dataset("yCoordinates", data=ys)
            proj = grp.create_dataset("projection", data=np.int32(epsg))
            proj.attrs["epsg_code"] = epsg
        return path

    before = write("DEMO_GCOV_20251110T033000.h5", river | lake | pond, np.zeros_like(river))
    after = write("DEMO_GCOV_20251122T033000.h5", river | lake | flood, growth)
    return before, after


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _size_mb(value) -> float:
    """ASF gives 'bytes' as a number for some products and as a dict of files for others."""
    if isinstance(value, (int, float)):
        return float(value) / 1e6
    if isinstance(value, dict):
        total = 0.0
        for v in value.values():
            if isinstance(v, dict):
                inner = v.get("bytes")
                total += float(inner) if isinstance(inner, (int, float)) else 0.0
            elif isinstance(v, (int, float)):
                total += float(v)
        return total / 1e6
    return 0.0


def _text(value, default="?") -> str:
    """Turn whatever ASF returned into a printable string."""
    if value is None or value == "":
        return default
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(v) for v in value)
    return str(value)


def cmd_search(a):
    import asf_search as asf

    if a.bbox:
        w, s, e, n = a.bbox
        area = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    elif a.lon is not None and a.lat is not None:
        area = f"POINT({a.lon} {a.lat})"
    else:
        raise SystemExit("Give --lon and --lat, or --bbox WEST SOUTH EAST NORTH.")
    kwargs = dict(dataset="NISAR", processingLevel=[a.level], intersectsWith=area, maxResults=a.max)
    if a.start:
        kwargs["start"] = a.start
    if a.end:
        kwargs["end"] = a.end
    # NASA's catalog can be slow, especially on busy days. Wait longer than the default 30 s and retry.
    from asf_search.constants import INTERNAL
    INTERNAL.CMR_TIMEOUT = a.timeout
    results = None
    for attempt in range(1, a.retries + 1):
        try:
            print(f"Searching NASA's catalog (attempt {attempt} of {a.retries}, waiting up to {a.timeout} s)...")
            results = asf.geo_search(**kwargs)
            break
        except Exception as exc:
            print(f"  attempt {attempt} failed: {type(exc).__name__}")
            if attempt < a.retries:
                time.sleep(5)
    if results is None:
        raise SystemExit(
            "\nCould not get an answer from NASA's catalog. Things to try:\n"
            "  1. Wait a few minutes and run the same command again. The servers are busy today.\n"
            "  2. Switch networks, for example to your phone's hotspot.\n"
            "  3. Search in your browser at https://search.asf.alaska.edu instead (set Dataset to NISAR),\n"
            "     download the .h5 files there, and skip `search` and `download`."
        )
    print(f"{len(results)} granules found")
    if not results:
        return

    rows = []
    for r in results:
        p = r.properties
        rows.append({
            "time": _text(p.get("startTime"), ""),
            "track": _text(p.get("pathNumber")),
            "frame": _text(p.get("frameNumber")),
            "direction": _text(p.get("flightDirection")),
            "mb": _size_mb(p.get("bytes")),
            "name": _text(p.get("fileID") or p.get("sceneName"), ""),
            "url": _text(p.get("url"), ""),
        })
        if a.debug:
            print({k: type(v).__name__ for k, v in sorted(p.items())})
    rows.sort(key=lambda r: r["time"])

    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["track"], r["frame"], r["direction"]), []).append(r)
    for (track, frame, direction), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        tag = "usable for change detection" if len(items) >= 2 else "single pass"
        print(f"\ntrack {track}  frame {frame}  {direction}  ({len(items)} passes, {tag})")
        for r in items:
            print(f"  {r['time'][:19]}  {r['mb']:8.0f} MB  {r['name']}")
            if a.urls:
                print(f"      {r['url']}")
    print("\nPick two passes from the same track and frame group. Add --urls to print download links.")


def _strip_auth_if_aws(r, *args, **kwargs):
    """Same redirect handling as asf_search: Amazon rejects requests that carry NASA's auth header."""
    import urllib.parse

    if 300 <= r.status_code <= 399 and "amazonaws.com" in urllib.parse.urlparse(r.headers.get("location", "")).netloc:
        location = r.headers["location"]
        r.headers.clear()
        r.headers["location"] = location


def _fetch(session, url, start=0):
    headers = {"Range": f"bytes={start}-"} if start else {}
    return session.get(url, stream=True, headers=headers, timeout=(30, 90),
                       hooks={"response": _strip_auth_if_aws})


def _show_progress(done, total, rate):
    text = f"  {done / 1e6:8.0f} MB"
    if total:
        text += f" of {total / 1e6:.0f} MB  ({done / total * 100:3.0f}%)"
        if rate > 0:
            text += f"  about {max(1, round((total - done) / rate / 60))} min left"
    if rate > 0:
        text += f"  {rate / 1e6:.1f} MB/s"
    print("\r" + text + "   ", end="", flush=True)


def download_resumable(session, url, folder, retries=30):
    """Download one file with a progress display. Safe to stop with Ctrl+C and run again."""
    import requests

    folder = Path(folder)
    name = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    final, part = folder / name, folder / (name + ".part")

    # A file with the final name may be a truncated leftover from an interrupted run.
    if final.exists():
        probe = _fetch(session, url)
        probe.raise_for_status()
        expected = int(probe.headers.get("Content-Length", 0))
        probe.close()
        have = final.stat().st_size
        if expected and have == expected:
            print(f"Already complete: {name}")
            return final
        print(f"{name} is incomplete ({have / 1e6:.0f} of {expected / 1e6:.0f} MB). Continuing it.")
        if have < expected and not part.exists():
            final.replace(part)
        else:
            final.unlink()

    print(f"Downloading {name}")
    done = part.stat().st_size if part.exists() else 0
    total, failures = None, 0
    while True:
        try:
            r = _fetch(session, url, start=done)
            if r.status_code == 416:  # nothing left to fetch
                break
            r.raise_for_status()
            if done and r.status_code != 206:  # the server ignored our resume request: start over
                done = 0
            length = int(r.headers.get("Content-Length", 0))
            total = done + length if length else total
            last_time, last_done = time.time(), done
            with open(part, "ab" if done else "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last_time >= 1.0:
                        _show_progress(done, total, (done - last_done) / (now - last_time))
                        last_time, last_done = now, done
            r.close()
            if total and done < total:
                raise requests.exceptions.ChunkedEncodingError("connection ended early")
            break
        except KeyboardInterrupt:
            print(f"\nStopped. Progress is saved ({done / 1e6:.0f} MB). Run the same command again to continue.")
            raise SystemExit(130)
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if 400 <= code < 500:
                raise SystemExit(
                    f"\nNASA refused the download (HTTP {code}). Check that you approved the ASF app "
                    "at urs.earthdata.nasa.gov (Applications, then Authorized Apps) and that your login is valid."
                )
            failures += 1
            problem = f"HTTP {code}"
        except (requests.exceptions.RequestException, OSError) as exc:
            failures += 1
            problem = type(exc).__name__
        if problem:
            if failures > retries:
                raise SystemExit(f"\nGiving up after {retries} retries ({problem}). Run the same command again later.")
            done = part.stat().st_size if part.exists() else 0
            print(f"\n  connection problem ({problem}). Retrying in 5 s and resuming from {done / 1e6:.0f} MB...")
            time.sleep(5)

    part.replace(final)
    size = final.stat().st_size
    print(f"\n  saved {name}  ({size / 1e6:.0f} MB)")
    if total and size != total:
        print(f"  warning: expected {total / 1e6:.0f} MB. Run the same command again to repair it.")
    return final


def cmd_download(a):
    import asf_search as asf

    token = os.environ.get("EARTHDATA_TOKEN")
    user, pwd = os.environ.get("EARTHDATA_USER"), os.environ.get("EARTHDATA_PASS")
    if token:
        session = asf.ASFSession().auth_with_token(token)
    elif user and pwd:
        session = asf.ASFSession().auth_with_creds(user, pwd)
    else:
        raise SystemExit("Set EARTHDATA_TOKEN (from urs.earthdata.nasa.gov, 'Generate Token'), "
                         "or EARTHDATA_USER and EARTHDATA_PASS.")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    print("Downloads show progress. Press Ctrl+C any time; run the same command again to continue.\n")
    for url in a.urls:
        download_resumable(session, url, a.out)
    print("\nAll done. Files are in", a.out)


def cmd_inspect(a):
    path = Path(a.file)
    with h5py.File(path, "r") as h5:
        def visit(name, obj):
            if name.count("/") > a.depth:
                return
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}  {obj.shape} {obj.dtype}")
            else:
                print(f"{name}/")
        h5.visititems(visit)
        print()
        for pol in ("HH", "HV", "VV", "VH"):
            try:
                ds = locate_band(h5, pol)
                print(f"{pol}: {ds.name}  shape {ds.shape}  EPSG:{read_epsg(ds.parent)}")
            except SystemExit:
                pass
    when = acquisition_time(path)
    print("acquisition time (from file name):", iso(when) or "not found")


def cmd_change(a):
    bbox = tuple(a.bbox) if a.bbox else None
    thr = a.threshold if a.threshold in (None, "auto") else float(a.threshold)
    run_change(a.before, a.after, a.out, a.pol, a.looks, thr, a.min_drop, a.min_blob,
               bbox, a.title, False, a.before_date, a.after_date)


def cmd_demo(a):
    folder = ROOT / "data" / "demo"
    print("Building two synthetic scenes (about 30 seconds)...")
    before, after = make_demo_scenes(folder)
    run_change(before, after, a.out, "HH", 4, None, 3.0, 4, None,
               "Tidemark", synthetic=True)


def cmd_serve(a):
    web = ROOT / "web"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(web))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", a.port), handler) as httpd:
        url = f"http://localhost:{a.port}/"
        print(f"Viewer running at {url}  (Ctrl+C to stop)")
        if not a.no_open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="list NISAR granules over a place")
    s.add_argument("--lon", type=float)
    s.add_argument("--lat", type=float)
    s.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    s.add_argument("--start", help="e.g. 2025-11-01")
    s.add_argument("--end")
    s.add_argument("--level", default="GCOV")
    s.add_argument("--max", type=int, default=100)
    s.add_argument("--timeout", type=int, default=120, help="seconds to wait for NASA's catalog (default 120)")
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--urls", action="store_true", help="also print download URLs")
    s.add_argument("--debug", action="store_true", help="print the raw property names")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("download", help="download granule URLs")
    s.add_argument("urls", nargs="+")
    s.add_argument("--out", default="data/real")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("inspect", help="show what is inside a NISAR .h5 file")
    s.add_argument("file")
    s.add_argument("--depth", type=int, default=6)
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("change", help="compare two GCOV files")
    s.add_argument("--before", required=True)
    s.add_argument("--after", required=True)
    s.add_argument("--pol", default="HH", help="HH, HV, VV or VH")
    s.add_argument("--looks", type=int, default=8, help="pixels averaged per side (8 means 160 m pixels)")
    s.add_argument("--threshold", default=None,
                   help="water cutoff in dB, for example -17. Default: picked from the data ('auto')")
    s.add_argument("--min-drop", type=float, default=3.0, help="minimum change in dB to count as new or receded water")
    s.add_argument("--min-blob", type=int, default=4, help="ignore patches smaller than this many pixels")
    s.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                   help="crop to this lon/lat box (strongly recommended)")
    s.add_argument("--before-date", help="override the date, e.g. 2025-11-10T03:30:00Z")
    s.add_argument("--after-date")
    s.add_argument("--title", default="Tidemark")
    s.add_argument("--out", default=str(WEB_DATA))
    s.set_defaults(func=cmd_change)

    s = sub.add_parser("demo", help="run on synthetic scenes")
    s.add_argument("--out", default=str(WEB_DATA))
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="serve the viewer locally")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
