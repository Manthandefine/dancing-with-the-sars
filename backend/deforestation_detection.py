"""
deforestation_detection.py  -  NISAR EcoFlood : SAR forest-change detection
===========================================================================

Pipeline
--------
 1. Read earlier (before) forest SAR observation
 2. Read recent  (after)  forest SAR observation
 3. Compare radar backscatter between the two dates
 4. Identify potential forest disturbances
 5. Highlight areas showing possible forest loss
 6. Calculate the estimated disturbed area
 7. Generate a forest-change map

Why it works
------------
 L-band radar (NISAR) penetrates the canopy and is dominated by volume
 scattering from trunks and branches, especially in cross-pol (HV). When forest
 is cleared or heavily degraded, HV backscatter drops sharply (typically 3-6+ dB).
 Water (e.g. flooding) also causes a drop, so pixels that became open water
 are separated out and NOT counted as deforestation.

How to run (Windows)
--------------------
    py deforestation_detection.py                    # no arguments -> demo
    py deforestation_detection.py --before a.tif --after b.tif
    py deforestation_detection.py --before a.tif --after b.tif --pol hv --out results

 Missing libraries (numpy, scipy, matplotlib) are installed automatically
 the first time you run the script.

From the web backend:
    from deforestation_detection import detect_deforestation
    r = detect_deforestation("before.tif", "after.tif", out_dir="results")
    r["disturbed_area_km2"], r["forest_loss_mask"]
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
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no GUI needed; works everywhere
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
# Settings
# --------------------------------------------------------------------------- #
# Approximate backscatter levels (dB). Tune these on a scene you can verify.
#   forest_min : earlier image must be at least this bright to count as forest
#   water      : darker than this = open water (specular reflection)
POL_PRESETS = {
    "hv": {"forest_min": -16.0, "water": -24.0},   # L-band cross-pol (best for biomass)
    "hh": {"forest_min": -11.0, "water": -20.0},   # L-band co-pol
}

# Class codes in the forest-change raster
NODATA, NONFOREST, STABLE, DISTURBED, LOSS, WATER = 0, 1, 2, 3, 4, 5
CLASS_COLORS = ["#ffffff", "#d9d9d9", "#2e7d32", "#ff9800", "#d32f2f", "#1976d2"]


@dataclass
class GeoInfo:
    pixel_area_km2: float
    pixel_size_m: tuple
    crs: Optional[str] = None
    transform: Optional[object] = None


# --------------------------------------------------------------------------- #
# 1 & 2. Reading data
# --------------------------------------------------------------------------- #
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


def read_mask(path: str, reference: Optional[str] = None):
    """Read an optional mask (.npy or GeoTIFF) as a boolean array."""
    if path.lower().endswith(".npy"):
        return np.load(path).astype(bool)
    arr, _ = read_sar(path, reference=reference, pixel_size_m=1.0)
    return np.nan_to_num(arr, nan=0.0) > 0.5


# --------------------------------------------------------------------------- #
# Pre-processing and small image helpers (scipy only, version-proof)
# --------------------------------------------------------------------------- #
def to_db(arr: np.ndarray) -> np.ndarray:
    """Linear power -> dB. Data that already contains negatives is assumed dB."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        raise ValueError("Raster contains no valid pixels.")
    if valid.min() < 0:
        return arr.astype("float32")
    out = 10.0 * np.log10(np.clip(arr, 1e-6, None))
    out[~np.isfinite(arr)] = np.nan
    return out.astype("float32")


def speckle_filter(db: np.ndarray, size: int = 5) -> np.ndarray:
    nan_mask = ~np.isfinite(db)
    filled = np.where(nan_mask, np.nanmedian(db), db)
    out = ndi.median_filter(filled, size=size)
    out[nan_mask] = np.nan
    return out.astype("float32")


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    keep = np.bincount(labels.ravel()) >= min_size
    keep[0] = False
    return keep[labels]


def fill_small_holes(mask: np.ndarray, max_size: int) -> np.ndarray:
    return ~remove_small_objects(~mask, max_size)


def opening(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return ndi.binary_opening(mask, structure=(x * x + y * y) <= radius * radius)


def area_km2(mask: np.ndarray, geo: GeoInfo) -> float:
    return float(mask.sum() * geo.pixel_area_km2)


# --------------------------------------------------------------------------- #
# 3-5. Core change detection
# --------------------------------------------------------------------------- #
def detect_changes(pre_db, post_db, geo: GeoInfo, pol="hv",
                   forest_min_db=None, water_db=None,
                   drop_db=3.0, severe_drop_db=6.0, z=2.5,
                   min_area_ha=0.5, forest_mask=None, exclude_mask=None):
    """
    Returns dict with class raster and diagnostics.

    Disturbance rule (per pixel):
        was forest  AND  not water  AND  not excluded
        AND  normalised backscatter change <= -max(drop_db, z * noise_sigma)

    * normalisation removes the scene-wide offset between dates (soil moisture,
      season, calibration) using the median change of the stable forest.
    * `z * noise_sigma` makes the threshold adapt to how noisy the scene is.
    * severe drops (>= severe_drop_db) are labelled 'probable forest loss';
      smaller ones 'possible disturbance / degradation'.
    """
    preset = POL_PRESETS[pol]
    forest_min = preset["forest_min"] if forest_min_db is None else forest_min_db
    water_thr = preset["water"] if water_db is None else water_db

    valid = np.isfinite(pre_db) & np.isfinite(post_db)
    water_before = valid & (pre_db < water_thr)
    water_after = valid & (post_db < water_thr)

    forest = valid & (pre_db >= forest_min) & ~water_before
    if forest_mask is not None:
        forest &= forest_mask
    excl = exclude_mask.astype(bool) if exclude_mask is not None else np.zeros_like(valid)

    if forest.sum() < 100:
        raise ValueError(
            "Almost no forest found in the earlier image. Check --pol (hv/hh) or lower "
            "--forest-min (current: %.1f dB)." % forest_min)

    diff = (post_db - pre_db).astype("float32")
    ref = diff[forest & ~water_after & ~excl]
    offset = float(np.clip(np.median(ref), -2.0, 2.0))
    sigma = max(1.4826 * float(np.median(np.abs(ref - np.median(ref)))), 0.3)
    diff_n = diff - offset

    thr = max(drop_db, z * sigma)
    with np.errstate(invalid="ignore"):
        cand = forest & ~water_after & ~excl & np.where(valid, diff_n <= -thr, False)

    min_px = max(4, int(math.ceil(min_area_ha * 1e4 /
                                  (geo.pixel_size_m[0] * geo.pixel_size_m[1]))))
    cand = opening(cand, 1)
    cand = remove_small_objects(cand, min_px)
    cand = fill_small_holes(cand, min_px)
    cand &= forest

    with np.errstate(invalid="ignore"):
        severe = cand & np.where(valid, diff_n <= -severe_drop_db, False)
    moderate = cand & ~severe

    classes = np.full(pre_db.shape, NODATA, dtype="uint8")
    classes[valid] = NONFOREST
    classes[forest] = STABLE
    classes[water_after & ~forest] = WATER
    classes[water_after & forest & ~cand] = WATER
    classes[moderate] = DISTURBED
    classes[severe] = LOSS

    return {
        "classes": classes, "diff_n": diff_n, "forest": forest,
        "disturbed": cand, "severe": severe, "moderate": moderate,
        "water_after": water_after,
        "offset_db": offset, "sigma_db": sigma, "threshold_db": thr,
        "forest_min_db": forest_min, "water_db": water_thr, "min_px": min_px,
    }


# --------------------------------------------------------------------------- #
# 7. Visualisation
# --------------------------------------------------------------------------- #
def _gray(db):
    lo, hi = np.nanpercentile(db, [2, 98])
    return np.nan_to_num(np.clip((db - lo) / (hi - lo + 1e-9), 0, 1))


def _legend():
    return [Patch(color=CLASS_COLORS[LOSS], label="Probable forest loss"),
            Patch(color=CLASS_COLORS[DISTURBED], label="Possible disturbance"),
            Patch(color=CLASS_COLORS[STABLE], label="Stable forest"),
            Patch(color=CLASS_COLORS[WATER], label="Water"),
            Patch(color=CLASS_COLORS[NONFOREST], label="Non-forest")]


def make_outputs(pre_db, post_db, res, stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmap = ListedColormap(CLASS_COLORS)
    paths = {}

    def single(img, name, title, **kw):
        p = os.path.join(out_dir, name)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, **kw)
        ax.set_title(title)
        ax.axis("off")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return p

    paths["before"] = single(_gray(pre_db), "1_forest_before.png",
                             "Earlier SAR observation", cmap="gray")
    paths["after"] = single(_gray(post_db), "1_forest_after.png",
                            "Recent SAR observation", cmap="gray")

    # backscatter change map (dB), diverging colours
    paths["change_db"] = os.path.join(out_dir, "2_backscatter_change_db.png")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.clip(res["diff_n"], -10, 10), cmap="RdBu", vmin=-10, vmax=10)
    ax.set_title("Backscatter change (after - before, dB)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="dB  (red = decrease)")
    fig.savefig(paths["change_db"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # forest-change map
    paths["forest_change_map"] = os.path.join(out_dir, "3_forest_change_map.png")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(res["classes"], cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
    ax.set_title(f"Forest change - disturbed area: {stats['disturbed_area_km2']:.2f} km² "
                 f"({stats['disturbed_area_ha']:.0f} ha)")
    ax.legend(handles=_legend(), loc="lower right", fontsize=8)
    ax.axis("off")
    fig.savefig(paths["forest_change_map"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # summary panel
    paths["summary"] = os.path.join(out_dir, "4_summary_panel.png")
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.5))
    axes[0].imshow(_gray(pre_db), cmap="gray");  axes[0].set_title("Before")
    axes[1].imshow(_gray(post_db), cmap="gray"); axes[1].set_title("After")
    axes[2].imshow(np.clip(res["diff_n"], -10, 10), cmap="RdBu", vmin=-10, vmax=10)
    axes[2].set_title("Change (dB)")
    axes[3].imshow(res["classes"], cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
    axes[3].set_title(f"Disturbed: {stats['disturbed_area_km2']:.2f} km²")
    axes[3].legend(handles=_legend(), loc="lower right", fontsize=7)
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(paths["summary"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    paths["mask_npy"] = os.path.join(out_dir, "forest_loss_mask.npy")
    np.save(paths["mask_npy"], res["disturbed"].astype("uint8"))
    paths["classes_npy"] = os.path.join(out_dir, "forest_change_classes.npy")
    np.save(paths["classes_npy"], res["classes"])
    paths["stats_json"] = os.path.join(out_dir, "deforestation_stats.json")
    with open(paths["stats_json"], "w") as f:
        json.dump(stats, f, indent=2)
    return paths


def save_geotiff(classes, geo: GeoInfo, path):
    if not HAS_RASTERIO or geo.transform is None:
        return None
    with rasterio.open(path, "w", driver="GTiff", height=classes.shape[0],
                       width=classes.shape[1], count=1, dtype="uint8",
                       crs=geo.crs, transform=geo.transform, nodata=0,
                       compress="lzw") as dst:
        dst.write(classes.astype("uint8"), 1)
    return path


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(before, after, geo: GeoInfo, out_dir, pol="hv", speckle_size=5,
                 **kwargs) -> dict:
    if before.shape != after.shape:
        raise ValueError(f"Image sizes differ {before.shape} vs {after.shape}. "
                         "Both images must cover the same area on the same grid.")
    pre_db = speckle_filter(to_db(before), speckle_size)
    post_db = speckle_filter(to_db(after), speckle_size)

    res = detect_changes(pre_db, post_db, geo, pol=pol, **kwargs)

    forest_km2 = area_km2(res["forest"], geo)
    dist_km2 = area_km2(res["disturbed"], geo)
    stats = {
        "disturbed_area_km2": round(dist_km2, 4),
        "disturbed_area_ha": round(dist_km2 * 100, 2),
        "probable_forest_loss_km2": round(area_km2(res["severe"], geo), 4),
        "possible_disturbance_km2": round(area_km2(res["moderate"], geo), 4),
        "forest_area_before_km2": round(forest_km2, 4),
        "percent_of_forest_disturbed": round(100 * dist_km2 / forest_km2, 3) if forest_km2 else 0.0,
        "water_after_km2": round(area_km2(res["water_after"], geo), 4),
        "scene_offset_db": round(res["offset_db"], 3),
        "noise_sigma_db": round(res["sigma_db"], 3),
        "detection_threshold_db": round(res["threshold_db"], 3),
        "forest_min_db": res["forest_min_db"],
        "water_db": res["water_db"],
        "polarisation_preset": pol,
        "pixel_size_m": [round(geo.pixel_size_m[0], 3), round(geo.pixel_size_m[1], 3)],
        "note": "Backscatter decrease indicates POSSIBLE forest loss; verify with "
                "optical imagery or field data before reporting.",
    }
    paths = make_outputs(pre_db, post_db, res, stats, out_dir)
    tif = save_geotiff(res["classes"], geo, os.path.join(out_dir, "forest_change_classes.tif"))
    if tif:
        paths["classes_tif"] = tif
    return {**stats, "outputs": paths, "forest_loss_mask": res["disturbed"],
            "forest_change_classes": res["classes"]}


def detect_deforestation(before_path, after_path, out_dir="deforestation_output",
                         band=1, pixel_size_m=None, forest_mask_path=None,
                         exclude_mask_path=None, **kwargs) -> dict:
    """Convenience entry point for the web backend."""
    before, geo = read_sar(before_path, band, pixel_size_m=pixel_size_m)
    after, _ = read_sar(after_path, band, reference=before_path if HAS_RASTERIO else None,
                        pixel_size_m=pixel_size_m)
    fm = read_mask(forest_mask_path, before_path) if forest_mask_path else None
    em = read_mask(exclude_mask_path, before_path) if exclude_mask_path else None
    return run_pipeline(before, after, geo, out_dir, forest_mask=fm, exclude_mask=em, **kwargs)


# --------------------------------------------------------------------------- #
# Synthetic demo (verifies the algorithm without real data)
# --------------------------------------------------------------------------- #
def make_demo_data(size=600, seed=7):
    """
    Forest with: a farm area, a river, two clear-cuts, one selective-logging patch,
    and a flooded forest patch (must NOT be counted as deforestation).
    Returns (before, after, geo, truth_loss_mask, flooded_forest_mask)  [linear power]
    """
    rng = np.random.default_rng(seed)
    before = rng.normal(-13.0, 1.2, (size, size))          # forest HV ~ -13 dB
    yy, xx = np.mgrid[0:size, 0:size]
    farm = xx < 100
    river = np.abs(yy - (300 + 30 * np.sin(xx / 60.0))) < 8
    before[farm] = rng.normal(-19.5, 1.2, farm.sum())
    before[river] = rng.normal(-28.0, 1.2, river.sum())

    after = before - 0.6 + rng.normal(0, 1.2, (size, size))   # season offset + new noise
    after[farm] = rng.normal(-19.5, 1.2, farm.sum()) - 0.6
    after[river] = rng.normal(-28.0, 1.2, river.sum())

    cut1 = np.zeros((size, size), bool); cut1[80:180, 250:380] = True
    cut2 = np.zeros((size, size), bool); cut2[400:480, 150:300] = True
    degr = np.zeros((size, size), bool); degr[100:160, 450:560] = True
    flood = np.zeros((size, size), bool); flood[450:560, 380:520] = True

    after[cut1 | cut2] = rng.normal(-20.0, 1.2, (cut1 | cut2).sum())
    after[degr] = before[degr] - 4.5 + rng.normal(0, 0.6, degr.sum())
    after[flood] = rng.normal(-27.0, 1.2, flood.sum())

    lin = lambda a: (10 ** (a / 10.0)).astype("float32")
    geo = GeoInfo(pixel_area_km2=(30 * 30) / 1e6, pixel_size_m=(30.0, 30.0))
    return lin(before), lin(after), geo, (cut1 | cut2 | degr), flood


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_out(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def main():
    ap = argparse.ArgumentParser(description="NISAR EcoFlood - deforestation detection")
    ap.add_argument("--before", help="earlier SAR image (GeoTIFF or .npy)")
    ap.add_argument("--after", help="recent SAR image (GeoTIFF or .npy)")
    ap.add_argument("--out", default=_default_out("deforestation_output"),
                    help="output folder (default: next to this script)")
    ap.add_argument("--pol", choices=["hv", "hh"], default="hv",
                    help="polarisation preset (HV is best for forest)")
    ap.add_argument("--band", type=int, default=1, help="raster band number")
    ap.add_argument("--pixel-size", type=float, default=None,
                    help="pixel size in metres (needed only if the file has no geo-info)")
    ap.add_argument("--forest-min", type=float, default=None,
                    help="min dB in earlier image to count as forest")
    ap.add_argument("--water-db", type=float, default=None, help="dB below which = water")
    ap.add_argument("--drop", type=float, default=3.0, help="min backscatter drop (dB)")
    ap.add_argument("--severe-drop", type=float, default=6.0,
                    help="drop (dB) labelled 'probable forest loss'")
    ap.add_argument("--min-area-ha", type=float, default=0.5, help="min patch size (hectares)")
    ap.add_argument("--forest-mask", help="optional forest mask (.npy/.tif) to restrict analysis")
    ap.add_argument("--exclude-mask", help="optional mask to exclude, e.g. flood_mask.npy")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    args = ap.parse_args()

    kw = dict(pol=args.pol, forest_min_db=args.forest_min, water_db=args.water_db,
              drop_db=args.drop, severe_drop_db=args.severe_drop,
              min_area_ha=args.min_area_ha)

    if args.demo or not (args.before and args.after):
        if not args.demo:
            print("No input images given -> running the built-in DEMO.\n"
                  "For real data use:  py deforestation_detection.py "
                  "--before a.tif --after b.tif\n")
        before, after, geo, truth, flooded = make_demo_data()
        res = run_pipeline(before, after, geo, args.out, **kw)
        f = res["forest_loss_mask"]
        iou = (f & truth).sum() / max((f | truth).sum(), 1)
        print(f"[demo check] overlap with true disturbed area (IoU): {iou:.3f}")
        print(f"[demo check] flooded forest wrongly flagged: "
              f"{100 * (f & flooded).sum() / flooded.sum():.1f} %")
    else:
        res = detect_deforestation(args.before, args.after, args.out, args.band,
                                   args.pixel_size, args.forest_mask,
                                   args.exclude_mask, **kw)

    print("\n=== NISAR EcoFlood : deforestation detection result ===")
    print(f"Estimated disturbed area   : {res['disturbed_area_km2']:.2f} km2 "
          f"({res['disturbed_area_ha']:.0f} ha)")
    print(f"  probable forest loss     : {res['probable_forest_loss_km2']:.2f} km2")
    print(f"  possible disturbance     : {res['possible_disturbance_km2']:.2f} km2")
    print(f"Forest area (earlier)      : {res['forest_area_before_km2']:.2f} km2")
    print(f"Share of forest disturbed  : {res['percent_of_forest_disturbed']:.2f} %")
    print(f"Outputs saved in           : {args.out}")


if __name__ == "__main__":
    main()
