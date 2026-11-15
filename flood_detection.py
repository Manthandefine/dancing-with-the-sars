"""
flood_detection.py  -  NISAR EcoFlood : SAR change-detection flood mapping
===========================================================================

Pipeline
--------
 1. Read pre-flood SAR raster
 2. Read post-flood SAR raster
 3. Identify water pixels (low radar backscatter, automatic Otsu threshold)
 4. Compare water coverage between the two dates
 5. Remove permanent water bodies (water in BOTH dates)
 6. Identify newly inundated areas (water AFTER, not BEFORE)
 7. Calculate flooded area in km^2
 8. Generate flood-extent map + figures + JSON stats

Input
-----
 GeoTIFFs from NISAR (L-band GCOV/GSLC -> HH or HV backscatter) or, until NISAR
 products are available to you, Sentinel-1 GRD as a stand-in. Both images must
 cover the same area; if grids differ, the post image is resampled to the pre
 image grid automatically.
 Values may be linear power/sigma0 or already in dB (auto-detected).

Usage
-----
    python flood_detection.py --pre pre.tif --post post.tif --out results/
    python flood_detection.py --demo                      # synthetic test data

    # from the web backend (FastAPI/Flask):
    from flood_detection import detect_flood
    result = detect_flood("pre.tif", "post.tif", out_dir="results")
    result["flooded_area_km2"]

Dependencies (numpy, scipy, matplotlib) are installed automatically on first run.
    Windows:  py flood_detection.py            (no arguments -> demo)
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Automatic dependency installation (so "No module named ..." cannot happen)
# --------------------------------------------------------------------------- #
import importlib.util
import math
import os
import subprocess
import sys


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _pip(*args: str) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    try:
        return subprocess.call(cmd) == 0
    except Exception:
        return False


def _bootstrap() -> None:
    required = [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib")]
    missing = [pip for mod, pip in required if not _have(mod)]
    if missing:
        print(f"[setup] Installing missing libraries: {', '.join(missing)} ...", flush=True)
        if not _pip(*missing):
            print("\n[setup] Automatic install failed. Run this command manually, "
                  "then run the script again:\n"
                  f"    py -m pip install {' '.join(missing)}\n")
            sys.exit(1)
        importlib.invalidate_caches()

    # GeoTIFF readers are optional: try rasterio (binary wheel only), else tifffile.
    if not _have("rasterio") and not _have("tifffile"):
        print("[setup] Installing a GeoTIFF reader ...", flush=True)
        if not _pip("--only-binary=:all:", "rasterio"):
            _pip("tifffile")
        importlib.invalidate_caches()


_bootstrap()

MODULE_API = 2   # used by ecoflood_ai.py to detect outdated copies

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: safe on servers
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage as ndi

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except Exception:
    HAS_TIFFFILE = False


# --------------------------------------------------------------------------- #
# Version-proof morphology helpers (scipy only)
# --------------------------------------------------------------------------- #
def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than min_size pixels."""
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def remove_small_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    """Fill background holes smaller than area_threshold pixels."""
    inv = ~mask
    big_bg = remove_small_objects(inv, area_threshold)   # background kept (large)
    return ~big_bg


def binary_opening(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Erosion then dilation with a disk of given radius."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return ndi.binary_opening(mask, structure=(x * x + y * y) <= radius * radius)


def threshold_otsu(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu's automatic threshold (pure numpy - no scikit-image needed)."""
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w1 = np.cumsum(hist).astype(float)
    w2 = np.cumsum(hist[::-1])[::-1].astype(float)
    m1 = np.cumsum(hist * centers) / np.maximum(w1, 1e-12)
    m2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(w2[::-1], 1e-12))[::-1]
    between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return float(centers[:-1][np.argmax(between)])


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class GeoInfo:
    """Minimal georeferencing info needed for area computation & export."""
    pixel_area_km2: float
    pixel_size_m: tuple  # (x, y) metres
    crs: Optional[str] = None
    transform: Optional[object] = None


# --------------------------------------------------------------------------- #
# 1 & 2. Reading data
# --------------------------------------------------------------------------- #
def _pixel_size_metres(transform, crs, height, width):
    """Return (dx_m, dy_m) for a raster, handling geographic (degree) CRS."""
    dx, dy = abs(transform.a), abs(transform.e)
    if crs is not None and crs.is_geographic:
        # centre latitude of the raster
        lat = transform.f + transform.e * (height / 2.0)
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        return dx * m_per_deg_lon, dy * m_per_deg_lat
    return dx, dy  # projected CRS: assumed metres


def _finish(arr, nodata, dx, dy, crs, transform):
    arr = arr.astype("float32")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    geo = GeoInfo(pixel_area_km2=(dx * dy) / 1e6, pixel_size_m=(float(dx), float(dy)),
                  crs=crs, transform=transform)
    return arr, geo


def read_sar(path: str, band: int = 1, reference: Optional[str] = None,
             pixel_size_m: Optional[float] = None):
    """
    Read one SAR band -> (float32 array with NaN for nodata, GeoInfo).
    Supports GeoTIFF (via rasterio or tifffile) and .npy arrays.
    If `reference` is given (rasterio only) the raster is resampled onto its grid.
    `pixel_size_m` overrides/provides the pixel size when the file has no geo-info.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()

    # ---- plain numpy array ----
    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr[band - 1]
        if pixel_size_m is None:
            raise ValueError(".npy input has no geo-info: add --pixel-size <metres>")
        return _finish(arr, None, pixel_size_m, pixel_size_m, None, None)

    # ---- rasterio ----
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            nodata = src.nodata
            if reference is None:
                arr = src.read(band)
                transform, crs = src.transform, src.crs
            else:
                with rasterio.open(reference) as ref:
                    transform, crs = ref.transform, ref.crs
                    shape = (ref.height, ref.width)
                arr = np.full(shape, np.nan, dtype="float32")
                reproject(source=rasterio.band(src, band), destination=arr,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=crs,
                          resampling=Resampling.bilinear,
                          src_nodata=nodata, dst_nodata=np.nan)
        h, w = arr.shape
        dx, dy = abs(transform.a), abs(transform.e)
        if crs is not None and crs.is_geographic:
            lat = transform.f + transform.e * (h / 2.0)
            dx, dy = dx * 111320.0 * math.cos(math.radians(lat)), dy * 111320.0
        if pixel_size_m is not None:
            dx = dy = pixel_size_m
        return _finish(arr, nodata, dx, dy, str(crs) if crs else None, transform)

    # ---- tifffile fallback ----
    if HAS_TIFFFILE:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            data = page.asarray()
            if data.ndim == 3:
                if data.shape[0] <= 16 and data.shape[0] < data.shape[-1]:
                    data = data[band - 1]
                else:
                    data = data[..., band - 1]
            tags = page.tags
            nodata = None
            t = tags.get(42113)  # GDAL_NODATA
            if t is not None:
                try:
                    nodata = float(str(t.value).strip("\x00 "))
                except ValueError:
                    pass
            dx = dy = pixel_size_m
            if pixel_size_m is None:
                sc, tp = tags.get(33550), tags.get(33922)  # PixelScale, TiePoint
                if sc is None:
                    raise ValueError("No geo-info in file: add --pixel-size <metres>")
                sx, sy = float(sc.value[0]), float(sc.value[1])
                if sx < 0.1:  # degrees
                    lat = (float(tp.value[4]) - sy * data.shape[0] / 2) if tp is not None else 0.0
                    dx, dy = sx * 111320.0 * math.cos(math.radians(lat)), sy * 111320.0
                else:
                    dx, dy = sx, sy
        return _finish(data, nodata, dx, dy, None, None)

    raise ImportError("No GeoTIFF reader available. Run: py -m pip install tifffile")


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
def to_db(arr: np.ndarray) -> np.ndarray:
    """Convert to dB. Auto-detects data that is already in dB (has negatives)."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        raise ValueError("Raster contains no valid pixels.")
    if np.nanmin(valid) < 0:           # linear power is never negative -> already dB
        return arr.astype("float32")
    out = 10.0 * np.log10(np.clip(arr, 1e-6, None))
    out[~np.isfinite(arr)] = np.nan
    return out.astype("float32")


def speckle_filter(db: np.ndarray, size: int = 5) -> np.ndarray:
    """Median filter to suppress SAR speckle (NaN-safe)."""
    nan_mask = ~np.isfinite(db)
    filled = np.where(nan_mask, np.nanmedian(db), db)
    out = ndi.median_filter(filled, size=size)
    out[nan_mask] = np.nan
    return out.astype("float32")


# --------------------------------------------------------------------------- #
# 3. Water detection
# --------------------------------------------------------------------------- #
def detect_water(db: np.ndarray, threshold_db: Optional[float] = None,
                 min_area_px: int = 50):
    """
    Water is specular -> very low backscatter. Threshold in dB.
    If threshold_db is None, Otsu's method is used, clamped to a physically
    sensible range (-26 .. -12 dB) so a scene with no water can't make Otsu
    label land as water.
    Returns (water_mask, threshold_used).
    """
    valid = np.isfinite(db)
    if threshold_db is None:
        t = float(threshold_otsu(db[valid]))
        threshold_db = float(np.clip(t, -26.0, -12.0))

    water = np.zeros(db.shape, dtype=bool)
    water[valid] = db[valid] < threshold_db

    # clean-up: remove speckle blobs and fill tiny holes
    water = binary_opening(water, 1)
    water = remove_small_objects(water, min_size=min_area_px)
    water = remove_small_holes(water, area_threshold=min_area_px)
    water &= valid
    return water, threshold_db


# --------------------------------------------------------------------------- #
# 4-6. Change detection
# --------------------------------------------------------------------------- #
def compare_water(pre_water: np.ndarray, post_water: np.ndarray,
                  pre_db: np.ndarray, post_db: np.ndarray,
                  min_drop_db: float = 3.0, min_flood_px: int = 50):
    """
    permanent water = water before AND after      (rivers, lakes, reservoirs)
    new flood      = water after AND NOT before   (inundated land)
    Extra guard: flooded pixels must also show a real backscatter drop
    (post - pre <= -min_drop_db), which reduces false alarms.
    """
    permanent = pre_water & post_water
    candidate = post_water & ~pre_water

    diff = post_db - pre_db
    drop_ok = np.where(np.isfinite(diff), diff <= -min_drop_db, False)
    flood = candidate & drop_ok

    flood = remove_small_objects(flood, min_size=min_flood_px)
    flood = remove_small_holes(flood, area_threshold=min_flood_px)
    flood &= ~permanent
    return permanent, flood


# --------------------------------------------------------------------------- #
# 7. Area
# --------------------------------------------------------------------------- #
def area_km2(mask: np.ndarray, geo: GeoInfo) -> float:
    return float(mask.sum() * geo.pixel_area_km2)


# --------------------------------------------------------------------------- #
# 8. Visualisation
# --------------------------------------------------------------------------- #
def _gray(db):
    lo, hi = np.nanpercentile(db, [2, 98])
    return np.clip((db - lo) / (hi - lo + 1e-9), 0, 1)


def _save_single(img, path, title, cmap="gray"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_outputs(pre_db, post_db, pre_water, post_water, permanent, flood,
                 stats: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    # (a) original satellite image (post-flood, plus pre-flood for reference)
    paths["original_pre"] = os.path.join(out_dir, "1_original_pre_flood.png")
    paths["original_post"] = os.path.join(out_dir, "1_original_post_flood.png")
    _save_single(_gray(pre_db), paths["original_pre"], "Original SAR - pre-flood")
    _save_single(_gray(post_db), paths["original_post"], "Original SAR - post-flood")

    # (b) detected water (post-flood): permanent = dark blue, new = cyan
    water_rgb = np.dstack([_gray(post_db)] * 3)
    water_rgb[post_water] = [0.10, 0.30, 0.90]
    paths["water"] = os.path.join(out_dir, "2_detected_water.png")
    _save_single(water_rgb, paths["water"], "Detected water (post-flood)")

    # (c) flood extent map: newly flooded highlighted in red
    flood_rgb = np.dstack([_gray(post_db)] * 3) * 0.85
    flood_rgb[permanent] = [0.10, 0.30, 0.90]
    flood_rgb[flood] = [1.00, 0.10, 0.10]
    paths["flood_map"] = os.path.join(out_dir, "3_flood_extent_map.png")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(flood_rgb)
    ax.set_title(f"Flood extent - newly flooded: {stats['flooded_area_km2']:.2f} km²")
    ax.legend(handles=[Patch(color=(1, .1, .1), label="Newly flooded"),
                       Patch(color=(.1, .3, .9), label="Permanent water"),
                       Patch(color=(.5, .5, .5), label="Land")],
              loc="lower right")
    ax.axis("off")
    fig.savefig(paths["flood_map"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (d) one-glance summary panel
    paths["summary"] = os.path.join(out_dir, "4_summary_panel.png")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    axes[0].imshow(_gray(pre_db), cmap="gray");  axes[0].set_title("Pre-flood SAR")
    axes[1].imshow(_gray(post_db), cmap="gray"); axes[1].set_title("Post-flood SAR")
    axes[2].imshow(water_rgb);                   axes[2].set_title("Detected water")
    axes[3].imshow(flood_rgb)
    axes[3].set_title(f"Newly flooded: {stats['flooded_area_km2']:.2f} km²")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(paths["summary"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (e) machine-readable mask for the web app / GIS
    paths["flood_mask_npy"] = os.path.join(out_dir, "flood_mask.npy")
    np.save(paths["flood_mask_npy"], flood.astype("uint8"))

    paths["permanent_water_npy"] = os.path.join(out_dir, "permanent_water_mask.npy")
    np.save(paths["permanent_water_npy"], permanent.astype("uint8"))

    paths["stats_json"] = os.path.join(out_dir, "flood_stats.json")
    with open(paths["stats_json"], "w") as f:
        json.dump(stats, f, indent=2)
    return paths


def save_geotiff(mask: np.ndarray, geo: GeoInfo, path: str):
    """Write the flood mask as a georeferenced GeoTIFF (for maps/GIS/overlays)."""
    if not HAS_RASTERIO or geo.transform is None:
        return None
    with rasterio.open(path, "w", driver="GTiff", height=mask.shape[0],
                       width=mask.shape[1], count=1, dtype="uint8",
                       crs=geo.crs, transform=geo.transform, nodata=255,
                       compress="lzw") as dst:
        dst.write(mask.astype("uint8"), 1)
    return path


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(pre_arr, post_arr, geo: GeoInfo, out_dir="flood_output",
                 threshold_db=None, speckle_size=5, min_area_px=50,
                 min_drop_db=3.0) -> dict:
    if pre_arr.shape != post_arr.shape:
        raise ValueError(f"Shape mismatch {pre_arr.shape} vs {post_arr.shape}")

    pre_db = speckle_filter(to_db(pre_arr), speckle_size)
    post_db = speckle_filter(to_db(post_arr), speckle_size)

    pre_water, t_pre = detect_water(pre_db, threshold_db, min_area_px)
    post_water, t_post = detect_water(post_db, threshold_db, min_area_px)

    permanent, flood = compare_water(pre_water, post_water, pre_db, post_db,
                                     min_drop_db, min_area_px)

    valid_px = int((np.isfinite(pre_db) & np.isfinite(post_db)).sum())
    stats = {
        "flooded_area_km2": round(area_km2(flood, geo), 4),
        "permanent_water_km2": round(area_km2(permanent, geo), 4),
        "pre_flood_water_km2": round(area_km2(pre_water, geo), 4),
        "post_flood_water_km2": round(area_km2(post_water, geo), 4),
        "analysed_area_km2": round(valid_px * geo.pixel_area_km2, 4),
        "flooded_percent_of_scene": round(100 * flood.sum() / max(valid_px, 1), 3),
        "pixel_size_m": [round(geo.pixel_size_m[0], 3), round(geo.pixel_size_m[1], 3)],
        "threshold_pre_db": round(t_pre, 2),
        "threshold_post_db": round(t_post, 2),
        "crs": geo.crs,
    }

    paths = make_outputs(pre_db, post_db, pre_water, post_water, permanent,
                         flood, stats, out_dir)
    tif = save_geotiff(flood, geo, os.path.join(out_dir, "flood_mask.tif"))
    if tif:
        paths["flood_mask_tif"] = tif

    return {**stats, "outputs": paths, "flood_mask": flood,
            "permanent_water_mask": permanent}


def detect_flood(pre_path: str, post_path: str, out_dir="flood_output",
                 band: int = 1, pixel_size_m=None, **kwargs) -> dict:
    """Convenience entry point for the web backend."""
    pre_arr, geo = read_sar(pre_path, band, pixel_size_m=pixel_size_m)
    post_arr, _ = read_sar(post_path, band, pixel_size_m=pixel_size_m,
                           reference=pre_path if HAS_RASTERIO else None)  # co-register
    return run_pipeline(pre_arr, post_arr, geo, out_dir, **kwargs)


# --------------------------------------------------------------------------- #
# Synthetic demo data (so the algorithm can be verified without real imagery)
# --------------------------------------------------------------------------- #
def make_demo_data(size=600, seed=42):
    """River + lake (permanent) and a flood plain that is inundated post-event."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]

    def scene(flooded: bool):
        land_db = rng.normal(-8.0, 1.5, (size, size))            # land ~ -8 dB
        water = np.zeros((size, size), bool)
        river = np.abs(yy - (size * 0.5 + 40 * np.sin(xx / 70.0))) < 12
        lake = (xx - 150) ** 2 + (yy - 150) ** 2 < 45 ** 2
        water |= river | lake
        if flooded:
            plain = (np.abs(yy - (size * 0.5 + 40 * np.sin(xx / 70.0))) < 90) & (xx > 200)
            water |= plain
        db = np.where(water, rng.normal(-21.0, 1.5, (size, size)), land_db)
        return (10 ** (db / 10.0)).astype("float32")             # linear power

    pre, post = scene(False), scene(True)
    geo = GeoInfo(pixel_area_km2=(30 * 30) / 1e6, pixel_size_m=(30.0, 30.0))
    return pre, post, geo


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="NISAR EcoFlood - flood detection")
    ap.add_argument("--pre", help="pre-flood SAR GeoTIFF")
    ap.add_argument("--post", help="post-flood SAR GeoTIFF")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "flood_output"),
                    help="output folder (default: next to this script)")
    ap.add_argument("--pixel-size", type=float, default=None,
                    help="pixel size in metres (only if the file has no geo-info)")
    ap.add_argument("--band", type=int, default=1, help="raster band (e.g. HH/VV/HV)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="water threshold in dB (default: automatic Otsu)")
    ap.add_argument("--min-area", type=int, default=50, help="min blob size (pixels)")
    ap.add_argument("--min-drop", type=float, default=3.0,
                    help="min backscatter drop (dB) to count as flooding")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    args = ap.parse_args()

    if args.demo or not (args.pre and args.post):
        if not args.demo:
            print("No input images given -> running the built-in DEMO.\n"
                  "For real data use:  py flood_detection.py --pre a.tif --post b.tif\n")
        pre, post, geo = make_demo_data()
        res = run_pipeline(pre, post, geo, args.out, args.threshold,
                           min_area_px=args.min_area, min_drop_db=args.min_drop)
    else:
        res = detect_flood(args.pre, args.post, args.out, args.band, args.pixel_size,
                           threshold_db=args.threshold,
                           min_area_px=args.min_area, min_drop_db=args.min_drop)

    print("\n=== NISAR EcoFlood : flood detection result ===")
    print(f"Estimated flooded area : {res['flooded_area_km2']:.2f} km2")
    print(f"Permanent water        : {res['permanent_water_km2']:.2f} km2 (excluded)")
    print(f"Flooded % of scene     : {res['flooded_percent_of_scene']:.2f} %")
    print(f"Outputs saved in       : {args.out}/")


if __name__ == "__main__":
    main()
