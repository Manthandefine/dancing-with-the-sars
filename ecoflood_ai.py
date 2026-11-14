"""
ecoflood_ai.py  -  NISAR EcoFlood AI : one program, one click
=============================================================

The AI runs the WHOLE pipeline for you and then explains it:

    1. Flood detection         (flood_detection.py)
    2. Deforestation detection (deforestation_detection.py)
    3. Flood vulnerability     (vulnerability.py)
    4. AI analyst: writes a plain-language briefing and answers your questions

It opens a web dashboard in your browser with maps, numbers, a region table,
the AI briefing and a chat box ("Why is Region F high?", "What should we do?").

Keep these 4 files in the SAME folder:
    ecoflood_ai.py   flood_detection.py   deforestation_detection.py   vulnerability.py

How to run (Windows)
--------------------
    py ecoflood_ai.py                 -> opens the dashboard, click "Run demo"
    py ecoflood_ai.py --cli           -> no browser: prints the AI briefing in the terminal
    py ecoflood_ai.py --cli --chat    -> ... and lets you ask questions in the terminal

    Your own data (GeoTIFF or .npy):
    py ecoflood_ai.py --pre pre.tif --post post.tif --dem dem.tif --rainfall rain.tif

Two kinds of AI explanation
---------------------------
  * Built-in explainer : works offline, no account, no key (default).
  * Claude (online)    : much richer answers to any question. Needs an Anthropic API key:
        PowerShell, this window only :   $env:ANTHROPIC_API_KEY = "sk-ant-..."
        PowerShell, permanent        :   setx ANTHROPIC_API_KEY "sk-ant-..."   (then reopen VS Code)
        or run with  --ask-key  to type the key in (it is not saved anywhere).
    If the online call ever fails, the app falls back to the built-in explainer.

Everything stays on your computer except the short results summary that is sent to
Claude when (and only when) you use an API key. Satellite images are never uploaded.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Setup: paths, automatic library installation, import the 3 analysis modules
# --------------------------------------------------------------------------- #
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
OUT_ROOT = os.path.join(HERE, "ecoflood_output")


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _bootstrap() -> None:
    required = [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib")]
    missing = [pip for mod, pip in required if not _have(mod)]
    if missing:
        print(f"[setup] Installing missing libraries: {', '.join(missing)} ...", flush=True)
        try:
            ok = subprocess.call([sys.executable, "-m", "pip", "install", *missing]) == 0
        except Exception:
            ok = False
        if not ok:
            print("\n[setup] Automatic install failed. Run this, then start again:\n"
                  f"    py -m pip install {' '.join(missing)}\n")
            sys.exit(1)
        importlib.invalidate_caches()


_bootstrap()

import argparse
import base64
import getpass
import html
import json
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlparse

import numpy as np

NEEDED_FILES = ("flood_detection.py", "deforestation_detection.py", "vulnerability.py")


def _import_modules():
    missing = [f for f in NEEDED_FILES if not os.path.exists(os.path.join(HERE, f))]
    if missing:
        print("\n[error] These files must be in the same folder as ecoflood_ai.py:\n  "
              + "\n  ".join(missing) + f"\nFolder: {HERE}\n")
        sys.exit(1)
    import flood_detection as fd
    import deforestation_detection as dd
    import vulnerability as vd
    old = [n for n, m in (("flood_detection.py", fd), ("deforestation_detection.py", dd),
                          ("vulnerability.py", vd)) if getattr(m, "MODULE_API", 0) < 2]
    if old:
        print("\n[error] Outdated copy of: " + ", ".join(old) +
              "\nDownload the latest version of every file again and replace the old ones "
              f"in {HERE}\n")
        sys.exit(1)
    return fd, dd, vd


fd, dd, vd = _import_modules()


# --------------------------------------------------------------------------- #
# Small text helpers
# --------------------------------------------------------------------------- #
DRIVER_TEXT = {
    "forest_loss": "forest loss nearby/upstream", "rainfall": "heavy rainfall",
    "elevation": "low elevation", "slope": "flat terrain (water pools)",
    "soil_moisture": "wet soil (little infiltration)", "river_distance": "closeness to a river",
    "drainage": "poor natural drainage",
}


def md_to_html(text: str) -> str:
    """Tiny, safe markdown subset: ## headings, - bullets, **bold**, `code`."""
    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)

    out, in_ul = [], False
    for line in text.splitlines():
        st = line.strip()
        if st.startswith(("- ", "* ", "• ")):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{inline(st[2:].strip())}</li>")
            continue
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if not st:
            continue
        m = re.match(r"#{1,4}\s+(.*)", st)
        out.append(f"<h3>{inline(m.group(1))}</h3>" if m else f"<p>{inline(st)}</p>")
    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


def _km2(x):
    return f"{x:,.1f} km²" if x >= 0.1 else f"{x:,.2f} km²"


def _level(pct, edges, words):
    for e, w in zip(edges, words):
        if pct < e:
            return w
    return words[-1]


# --------------------------------------------------------------------------- #
# Synthetic demo scene (one consistent area for all three analyses)
# --------------------------------------------------------------------------- #
def make_combined_demo(seed=11):
    """Forest + river valley + hills. Radar before/after with clear-cuts, a logging patch
    and a flood along the river. Also DEM, rainfall and soil moisture."""
    layers, geo = vd.make_demo_data(n=400, seed=5)
    n = 400
    rng = np.random.default_rng(seed)
    river = layers["rivers"] > 0.5
    flood = layers["flood_mask"] > 0.5
    loss = layers["forest_loss"] > 0.5

    base = rng.normal(-13.0, 1.2, (n, n))                       # forest, HV dB
    before = base.copy()
    before[river] = rng.normal(-28.0, 1.2, river.sum())
    after = base - 0.6 + rng.normal(0, 1.2, (n, n))
    after[river] = rng.normal(-28.0, 1.2, river.sum())
    degr = np.zeros((n, n), bool)
    degr[20:70, 300:380] = True
    after[degr] = base[degr] - 4.5 + rng.normal(0, 0.6, degr.sum())
    cut = loss & ~river
    after[cut] = rng.normal(-20.0, 1.2, cut.sum())
    after[flood] = rng.normal(-27.0, 1.2, flood.sum())          # flooded forest looks like water

    lin = lambda a: (10 ** (a / 10.0)).astype("float32")
    return {"before": lin(before), "after": lin(after), "geo": geo,
            "dem": layers["dem"], "rainfall": layers["rainfall"],
            "soil_moisture": layers["soil_moisture"]}


# --------------------------------------------------------------------------- #
# The whole pipeline
# --------------------------------------------------------------------------- #
def _clean_path(p):
    p = (p or "").strip().strip('"').strip("'")
    return p or None


def run_full_analysis(cfg: dict, log=print) -> dict:
    run_dir = os.path.join(OUT_ROOT, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    demo = cfg.get("mode", "demo") == "demo"
    pol = cfg.get("pol", "hv") if cfg.get("pol") in ("hv", "hh") else "hv"
    band = int(cfg.get("band") or 1)
    px = float(cfg["pixel_size"]) if cfg.get("pixel_size") else None
    water_thr = dd.POL_PRESETS[pol]["water"]
    res = {"demo": demo, "run_dir": run_dir, "pol": pol, "analyses": []}
    t0 = time.time()

    # ---------- inputs ----------
    pre = post = before = after = dem_p = rain_p = soil_p = None
    if demo:
        log("Building a synthetic demo scene (forest, river, flood, clear-cuts) ...")
        scene = make_combined_demo()
        geo = scene["geo"]
    else:
        pre, post = _clean_path(cfg.get("pre")), _clean_path(cfg.get("post"))
        before = _clean_path(cfg.get("before")) or pre
        after = _clean_path(cfg.get("after")) or post
        dem_p, rain_p = _clean_path(cfg.get("dem")), _clean_path(cfg.get("rainfall"))
        soil_p = _clean_path(cfg.get("soil_moisture"))
        if not ((pre and post) or (before and after) or dem_p or rain_p or soil_p):
            raise ValueError("No input files given. Provide at least a before/after satellite "
                             "pair, or a DEM plus rainfall.")
        first = next(p for p in (pre, before, dem_p, rain_p, soil_p) if p)
        _, geo = vd.read_layer(first, band=band, pixel_size_m=px)
        if geo is None:
            raise ValueError("Pixel size unknown for these files - fill in 'Pixel size (m)'.")

    fr = dr = vr = None

    # ---------- 1. flood ----------
    if demo or (pre and post):
        log("Step 1/3  Detecting flood water (before vs after radar) ...")
        fdir = os.path.join(run_dir, "flood")
        if demo:
            fr = fd.run_pipeline(scene["before"], scene["after"], fd.GeoInfo(
                pixel_area_km2=geo.pixel_area_km2, pixel_size_m=geo.pixel_size_m),
                fdir, threshold_db=water_thr)
        else:
            fr = fd.detect_flood(pre, post, fdir, band=band, pixel_size_m=px,
                                 threshold_db=water_thr)
        res["analyses"].append("flood")
        log(f"          flooded area = {fr['flooded_area_km2']:.2f} km2")
    else:
        log("Step 1/3  Skipped flood detection (needs a before/after radar pair).")

    # ---------- 2. forest ----------
    if demo or (before and after):
        log("Step 2/3  Detecting forest change (radar backscatter drop) ...")
        ddir = os.path.join(run_dir, "forest")
        if demo:
            excl = fr["flood_mask"] if fr else None
            dr = dd.run_pipeline(scene["before"], scene["after"], dd.GeoInfo(
                pixel_area_km2=geo.pixel_area_km2, pixel_size_m=geo.pixel_size_m),
                ddir, pol=pol, exclude_mask=excl)
        else:
            same_pair = fr is not None and (before, after) == (pre, post)
            excl_path = fr["outputs"].get("flood_mask_npy") if same_pair else None
            dr = dd.detect_deforestation(before, after, ddir, band=band, pixel_size_m=px,
                                         exclude_mask_path=excl_path, pol=pol)
        res["analyses"].append("forest")
        log(f"          disturbed forest = {dr['disturbed_area_ha']:.0f} ha")
    else:
        log("Step 2/3  Skipped forest analysis (needs a before/after radar pair).")

    # ---------- 3. vulnerability ----------
    vlayers = {}
    if dr is not None:
        vlayers["forest_loss"] = dr["forest_loss_mask"].astype("float32")
    if fr is not None:
        vlayers["rivers"] = fr["permanent_water_mask"].astype("float32")
        vlayers["flood_mask"] = fr["flood_mask"].astype("float32")
    vgeo = geo
    if demo:
        vlayers.update(dem=scene["dem"], rainfall=scene["rainfall"],
                       soil_moisture=scene["soil_moisture"])
    else:
        for key, path in (("dem", dem_p), ("rainfall", rain_p), ("soil_moisture", soil_p)):
            if path:
                arr, g = vd.read_layer(path, band=band, pixel_size_m=px if key == "dem" else None)
                vlayers[key] = arr
                if key == "dem" and g is not None:
                    vgeo = g
    n_usable = len([k for k in vlayers if k not in ("flood_mask",)])
    if n_usable >= 2:
        log("Step 3/3  Scoring flood vulnerability (7 factors) ...")
        vr = vd.run_pipeline(vlayers, vgeo, os.path.join(run_dir, "vulnerability"))
        res["analyses"].append("vulnerability")
        hi = vr["area_by_class_km2"]["High"]
        log(f"          high-vulnerability area = {hi:.2f} km2")
    else:
        log("Step 3/3  Skipped vulnerability (needs at least 2 layers, e.g. DEM + rainfall).")

    if not res["analyses"]:
        raise ValueError("Nothing could be analysed with the inputs given.")

    res.update(flood=fr, forest=dr, vuln=vr, geo=geo)
    res["links"] = compute_links(fr, dr, vr)
    res["ctx"] = build_context(res)
    res["seconds"] = round(time.time() - t0, 1)
    log(f"Analysis finished in {res['seconds']} s. Writing AI explanation ...")
    return res


def compute_links(fr, dr, vr):
    """How do forest loss, flooding and vulnerability overlap? (on the vulnerability grid)"""
    if vr is None or (dr is None and fr is None):
        return None
    score, cls = vr["vulnerability_score"], vr["vulnerability_classes"]
    out = {}
    px_km2 = vr["pixel_size_m"][0] * vr["pixel_size_m"][1] / 1e6
    ok = np.isfinite(score)
    if dr is not None:
        loss = vd.match_shape(dr["forest_loss_mask"].astype("float32"), score.shape, order=0) > 0.5
        loss &= ok
        if loss.sum() > 10:
            out["forest_loss_area_km2"] = round(float(loss.sum() * px_km2), 2)
            out["percent_of_forest_loss_in_high_class"] = round(100 * float((cls[loss] == vd.HIGH).mean()), 1)
            out["mean_score_in_forest_loss_areas"] = round(float(score[loss].mean()), 1)
            out["mean_score_elsewhere"] = round(float(score[ok & ~loss].mean()), 1)
            if fr is not None:
                fl = vd.match_shape(fr["flood_mask"].astype("float32"), score.shape, order=0) > 0.5
                out["flooded_area_overlapping_forest_loss_km2"] = round(float((fl & loss).sum() * px_km2), 2)
    return out or None


def build_context(res: dict) -> dict:
    fr, dr, vr = res["flood"], res["forest"], res["vuln"]
    ctx = {
        "data_source": ("SYNTHETIC DEMO DATA (illustrative only, not a real place)"
                        if res["demo"] else "user-provided satellite/terrain files"),
        "polarisation_preset": res["pol"],
        "analyses_run": res["analyses"],
    }
    if fr:
        ctx["flood"] = {k: fr[k] for k in ("flooded_area_km2", "permanent_water_km2",
                        "flooded_percent_of_scene", "analysed_area_km2", "pre_flood_water_km2",
                        "post_flood_water_km2", "threshold_post_db")}
    if dr:
        ctx["forest"] = {k: dr[k] for k in ("disturbed_area_km2", "disturbed_area_ha",
                         "probable_forest_loss_km2", "possible_disturbance_km2",
                         "forest_area_before_km2", "percent_of_forest_disturbed",
                         "scene_offset_db", "noise_sigma_db", "detection_threshold_db")}
    if vr:
        regions = [{
            "region": r["region"], "vulnerability": r["vulnerability"],
            "mean_score": r["mean_score"], "high_area_km2": r["high_area_km2"],
            "high_percent": r["high_percent"],
            "main_drivers": [DRIVER_TEXT[k] for k in r.get("top_drivers", [])],
        } for r in sorted(vr["regions"], key=lambda r: -r["mean_score"])]
        ctx["vulnerability"] = {
            "mean_score_0_100": vr["mean_score"],
            "area_by_class_km2": vr["area_by_class_km2"],
            "percent_by_class": vr["percent_by_class"],
            "class_rule": "Low < 40, Medium 40-60, High >= 60 (score 0-100)",
            "factors_used": vr["layers_used"], "factors_missing": vr["layers_missing"],
            "weights_used": vr["weights_used"], "rivers_source": vr["rivers_source"],
            "regions_sorted_high_to_low": regions,
            "validation_against_observed_flood": vr.get("validation_against_observed_flood"),
        }
    if res.get("links"):
        ctx["forest_loss_and_vulnerability_overlap"] = res["links"]
    return ctx


# --------------------------------------------------------------------------- #
# AI explainer 1: built-in (offline, rule-based, always available)
# --------------------------------------------------------------------------- #
def _drivers_sentence(r):
    d = r.get("main_drivers") or []
    return ", ".join(d[:-1]) + (" and " if len(d) > 1 else "") + d[-1] if d else "a mix of factors"


def t_flood(c):
    f = c.get("flood")
    if not f:
        return "Flood detection was not part of this run (it needs a before/after radar pair)."
    lvl = _level(f["flooded_percent_of_scene"], (1, 5, 15), ("limited", "moderate", "substantial", "extensive"))
    return (
        "## Flooding\n"
        f"- Newly flooded area: **{_km2(f['flooded_area_km2'])}**, which is **{f['flooded_percent_of_scene']:.1f}%** "
        f"of the analysed area, a **{lvl}** flood.\n"
        f"- Permanent water that was already there before the event (rivers, lakes) covers **{_km2(f['permanent_water_km2'])}** "
        "and was removed so normal water is not counted as flood.\n"
        "- How it works: smooth water reflects radar away from the satellite, so it looks very dark. "
        "The app finds dark pixels before and after, and keeps only pixels that turned dark "
        "(a backscatter drop of at least 3 dB).\n"
        "- On the map: **red = newly flooded**, blue = permanent water.")


def t_forest(c):
    f = c.get("forest")
    if not f:
        return "Forest analysis was not part of this run (it needs a before/after radar pair)."
    lvl = _level(f["percent_of_forest_disturbed"], (1, 5, 15), ("small", "noticeable", "significant", "severe"))
    return (
        "## Forest change\n"
        f"- Radar shows possible forest disturbance over **{_km2(f['disturbed_area_km2'])}** "
        f"(**{f['disturbed_area_ha']:,.0f} hectares**), **{f['percent_of_forest_disturbed']:.1f}%** of the forest: a **{lvl}** change.\n"
        f"- Probable forest loss (large backscatter drop, 6 dB or more): **{_km2(f['probable_forest_loss_km2'])}** (red). "
        f"Possible disturbance or degradation (3 to 6 dB): **{_km2(f['possible_disturbance_km2'])}** (orange).\n"
        "- How it works: dense forest scatters L-band radar strongly. When trees are removed the signal drops. "
        "Flooded pixels are excluded so water is not mistaken for deforestation.\n"
        "- Important: a drop means **possible** loss. Confirm it with optical images or field checks before reporting it.")


def t_vuln(c):
    v = c.get("vulnerability")
    if not v:
        return "Vulnerability scoring was not part of this run (it needs at least 2 data layers, e.g. DEM + rainfall)."
    a, p = v["area_by_class_km2"], v["percent_by_class"]
    regs = v["regions_sorted_high_to_low"]
    top, low = regs[0], regs[-1]
    lines = [
        "## Flood vulnerability",
        f"- Average score **{v['mean_score_0_100']}/100**. High: **{_km2(a['High'])}** ({p['High']}%), "
        f"Medium: **{_km2(a['Medium'])}** ({p['Medium']}%), Low: **{_km2(a['Low'])}** ({p['Low']}%).",
        f"- Most vulnerable region: **{top['region']}** ({top['vulnerability']}, score {top['mean_score']}), mainly because of "
        f"{_drivers_sentence(top)}.",
    ]
    if len(regs) > 2 and regs[1]["vulnerability"] != "Low":
        s = regs[1]
        lines.append(f"- Next: **{s['region']}** ({s['vulnerability']}, score {s['mean_score']}): {_drivers_sentence(s)}.")
    lines.append(f"- Least vulnerable: **{low['region']}** ({low['vulnerability']}, score {low['mean_score']}).")
    lines.append("- The score blends forest loss, rainfall, elevation, slope, soil moisture, distance to rivers "
                 "and drainage. It is a **relative risk index**, not a flood probability.")
    return "\n".join(lines)


def t_link(c):
    k = c.get("forest_loss_and_vulnerability_overlap")
    if not k:
        return ("The link between forest loss and flood vulnerability needs both the forest analysis "
                "and the vulnerability analysis in the same run.")
    lines = ["## How forest loss and flooding connect",
             "Trees slow rainfall, soak up water and hold soil. Where forest is removed, more water runs off quickly "
             "into rivers, which raises flood vulnerability downstream."]
    if "percent_of_forest_loss_in_high_class" in k:
        lines.append(f"- **{k['percent_of_forest_loss_in_high_class']}%** of the cleared/disturbed forest "
                     f"({_km2(k['forest_loss_area_km2'])}) lies in the High-vulnerability class.")
        lines.append(f"- Average vulnerability score is **{k['mean_score_in_forest_loss_areas']}** inside forest-loss areas vs "
                     f"**{k['mean_score_elsewhere']}** elsewhere.")
    if "flooded_area_overlapping_forest_loss_km2" in k:
        lines.append(f"- Observed flooding overlaps forest-loss areas over **{_km2(k['flooded_area_overlapping_forest_loss_km2'])}**.")
    lines.append("- Caution: forest loss is one of the seven inputs, so some of this overlap is built in. "
                 "It shows where the two problems coincide, not proof that loss caused the flood.")
    return "\n".join(lines)


def t_actions(c):
    acts = []
    v = c.get("vulnerability")
    if v:
        high = [r for r in v["regions_sorted_high_to_low"] if r["vulnerability"] == "High"][:2]
        if high:
            acts.append("Prioritise field checks, early-warning and drainage upkeep in **" +
                        " and ".join(r["region"] for r in high) + "** (the High regions).")
        else:
            acts.append("No region reached High; keep monitoring the Medium regions, especially after heavy rain.")
    if c.get("forest", {}).get("probable_forest_loss_km2", 0) > 0:
        acts.append("Verify the red forest-loss patches with optical imagery (e.g. Sentinel-2) or a field visit before "
                    "reporting them as deforestation.")
    k = c.get("forest_loss_and_vulnerability_overlap") or {}
    if k.get("percent_of_forest_loss_in_high_class", 0) > 25:
        acts.append("Restore vegetation on cleared land near rivers and on slopes above High-risk areas to slow runoff.")
    if "vulnerability" in c:
        acts.append("Next step: overlay population, roads and buildings on the High class to see which communities are exposed.")
    if "SYNTHETIC" in c["data_source"]:
        acts.append("Replace the demo with real NISAR/Sentinel-1 scenes, a DEM and rainfall data for your area.")
    return "## Recommended next steps\n" + "\n".join(f"- {a}" for a in acts)


def t_trust(c):
    pts = ["## How much to trust this"]
    if "SYNTHETIC" in c["data_source"]:
        pts.append("- This run uses **synthetic demo data**, so the numbers are illustrative only. Real scenes will be noisier.")
    pts += [
        "- Radar thresholds (water, forest, 3 dB drop) are sensible starting values. Tune them on an area you can verify.",
        "- The vulnerability score is a weighted index with adjustable weights, **not a validated flood probability**.",
        "- Burned areas, regrowth, very wet soil and steep terrain (radar shadow) can mimic or hide changes.",
    ]
    val = (c.get("vulnerability") or {}).get("validation_against_observed_flood")
    if val:
        pts.append(f"- Check against the mapped flood: {val['percent_of_flood_in_medium_or_high']}% of flooded pixels fall in "
                   f"Medium/High, AUC {val['auc_roc']} (0.5 = no skill, 1 = perfect). Meaningful only if the flood map is "
                   "independent of the inputs.")
    return "\n".join(pts)


def t_factors(c):
    v = c.get("vulnerability")
    w = v["weights_used"] if v else vd.DEFAULT_WEIGHTS
    names = {"forest_loss": "Forest loss", "rainfall": "Rainfall", "elevation": "Elevation",
             "slope": "Slope", "soil_moisture": "Soil moisture", "river_distance": "Distance to rivers",
             "drainage": "Drainage"}
    why = {"forest_loss": "less forest means faster runoff", "rainfall": "more rain means more water",
           "elevation": "low land floods first", "slope": "flat land lets water pool",
           "soil_moisture": "wet soil absorbs less", "river_distance": "closer to a river is riskier",
           "drainage": "poorly drained land stays wet (Topographic Wetness Index)"}
    lines = ["## What goes into the score"]
    for k, wt in sorted(w.items(), key=lambda kv: -kv[1]):
        lines.append(f"- **{names[k]}** (weight {wt:.0%}): {why[k]}.")
    if v and v["factors_missing"]:
        lines.append("- Not available this run: " + ", ".join(names[k] for k in v["factors_missing"]) +
                     " (weights were re-balanced).")
    return "\n".join(lines)


def t_method(c):
    return ("## How the app works\n"
            "- **Satellite radar (SAR)**, such as NASA-ISRO NISAR, sees through cloud and works day and night, "
            "which makes it ideal for floods.\n"
            "- **Flood:** compare dark (water) pixels before and after, remove permanent water, keep the new ones.\n"
            "- **Forest:** compare radar backscatter before and after. A strong drop over former forest means possible clearing.\n"
            "- **Vulnerability:** combine seven factors into a 0-100 score and group land into Low, Medium and High.\n"
            "- **AI analyst:** turns all the numbers into this plain-language explanation.")


def t_region(c, name):
    v = c.get("vulnerability")
    if not v:
        return "No vulnerability results are available yet."
    for r in v["regions_sorted_high_to_low"]:
        if r["region"].lower().endswith(name.lower()):
            return (f"## {r['region']}\n"
                    f"- Vulnerability: **{r['vulnerability']}** (average score {r['mean_score']}/100).\n"
                    f"- **{r['high_percent']}%** of the region ({_km2(r['high_area_km2'])}) is in the High class.\n"
                    f"- Main drivers: {_drivers_sentence(r)}.")
    names = ", ".join(r["region"] for r in v["regions_sorted_high_to_low"])
    return f"I could not find that region. Available regions: {names}."


def t_why(c):
    v = c.get("vulnerability")
    if not v:
        return "No vulnerability results are available yet."
    top = v["regions_sorted_high_to_low"][0]
    return (f"## Why is {top['region']} the most vulnerable?\n"
            f"Its average score is **{top['mean_score']}/100** ({top['vulnerability']}). The factors pushing it up most are "
            f"{_drivers_sentence(top)}.\n"
            "A factor counts more when it scores high **and** has a large weight. "
            "Ask about a specific region (for example \"Why is Region C low?\") to see its drivers.")


def offline_briefing(c) -> str:
    bits = []
    if "flood" in c:
        bits.append(f"about **{_km2(c['flood']['flooded_area_km2'])}** looks newly flooded")
    if "forest" in c:
        bits.append(f"**{c['forest']['disturbed_area_ha']:,.0f} ha** of forest shows radar signs of disturbance")
    if "vulnerability" in c:
        hi = c["vulnerability"]["area_by_class_km2"]["High"]
        bits.append(f"**{_km2(hi)}** is in the High flood-vulnerability class")
    head = ("## Overall picture\n"
            f"Data used: {c['data_source']}. Analyses run: {', '.join(c['analyses_run'])}. "
            + ("In short: " + "; ".join(bits) + "." if bits else ""))
    parts = [head]
    for fn in (t_flood, t_forest, t_vuln, t_link, t_actions, t_trust):
        if fn is t_flood and "flood" not in c:
            continue
        if fn is t_forest and "forest" not in c:
            continue
        if fn is t_vuln and "vulnerability" not in c:
            continue
        if fn is t_link and not c.get("forest_loss_and_vulnerability_overlap"):
            continue
        parts.append(fn(c))
    return "\n\n".join(parts)


def offline_answer(q: str, c: dict) -> str:
    t = q.lower()
    m = re.search(r"\bregion\s*([a-z0-9]+)\b", t)
    if m and "vulnerability" in c:
        return t_region(c, m.group(1))
    if re.search(r"reliab|trust|accura|valid|limit|error|confiden|sure|correct|wrong|certain", t):
        return t_trust(c)
    if re.search(r"recommend|should|action|first|priorit|mitigat|plan|suggest|next step|what to do|help", t):
        return t_actions(c)
    if re.search(r"\bwhy\b", t) and re.search(r"high|risk|vulnerab|region|worst|top", t):
        return t_why(c)
    if re.search(r"link|connect|relation|forest.*flood|deforest.*flood|flood.*forest|loss.*flood|cause|impact", t):
        return t_link(c)
    if re.search(r"forest|deforest|tree|disturb|canopy|logging|clear", t):
        return t_forest(c)
    if re.search(r"flood|inundat|submerg|water", t):
        return t_flood(c)
    if re.search(r"weight|factor|rain|slope|elevation|soil|drainage|river|twi|input|made of|calculat", t):
        return t_factors(c)
    if re.search(r"vulnerab|risk|score|class|high|medium|low|region", t):
        return t_vuln(c)
    if re.search(r"how|method|work|algorithm|nisar|sar\b|backscatter|radar|satellite", t):
        return t_method(c)
    return offline_briefing(c) + ("\n\nTip: you can ask things like \"Why is Region F high?\", "
                                  "\"How reliable is this?\" or \"What should be done first?\".")


# --------------------------------------------------------------------------- #
# AI explainer 2: Claude (online, optional)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are EcoFlood AI, the analyst inside the NISAR EcoFlood app (flood mapping, deforestation "
    "detection and flood-vulnerability scoring from radar satellite data). You receive the JSON results of "
    "the analysis the user just ran. Explain them in clear, plain language for non-experts: what the numbers mean, "
    "why they look that way, and what to do next.\n"
    "Rules: use only numbers that appear in the JSON and never invent figures; always give units (km2, hectares, dB); "
    "if data_source says SYNTHETIC DEMO DATA, say the results are illustrative; the vulnerability score is a relative "
    "weighted index and NOT a validated flood probability; a radar backscatter drop indicates POSSIBLE forest loss that "
    "needs verification; if the question is not covered by the results, say so and suggest what data would answer it. "
    "Be concise (under 250 words unless asked for detail). Use short paragraphs or '-' bullets and '## ' headings. "
    "Do not use tables."
)
BRIEFING_PROMPT = (
    "Write the analysis briefing for this run. Use these '## ' sections (skip any that do not apply): "
    "Overall picture, Flooding, Forest change, Flood vulnerability, How forest loss and flooding connect, "
    "Recommended next steps, How much to trust this. Keep it under 350 words."
)


class Explainer:
    def __init__(self):
        self.key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self.model = os.environ.get("ECOFLOOD_MODEL", "claude-sonnet-5-5")
        self.url = os.environ.get("ECOFLOOD_API_URL", "https://api.anthropic.com/v1/messages")
        self.last_error = None

    @property
    def label(self):
        return f"Claude ({self.model})" if self.key else "Built-in explainer (offline)"

    def _call(self, messages, max_tokens=900):
        body = json.dumps({"model": self.model, "max_tokens": max_tokens,
                           "system": SYSTEM_PROMPT, "messages": messages}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "content-type": "application/json", "x-api-key": self.key,
            "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code} {detail}".strip())
        except Exception as e:
            raise RuntimeError(str(e))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        if not text.strip():
            raise RuntimeError("empty answer")
        return text.strip()

    def _ctx_msg(self, ctx):
        return "ANALYSIS RESULTS (JSON):\n" + json.dumps(ctx, indent=1, default=float)

    def briefing(self, ctx):
        if self.key:
            try:
                txt = self._call([{"role": "user", "content": self._ctx_msg(ctx) + "\n\n" + BRIEFING_PROMPT}], 1200)
                self.last_error = None
                return txt, self.label
            except Exception as e:
                self.last_error = str(e)
        note = f" (online AI unavailable: {self.last_error})" if self.last_error else ""
        return offline_briefing(ctx), "Built-in explainer (offline)" + note

    def answer(self, question, ctx, history):
        if self.key:
            try:
                msgs = [{"role": "user", "content": self._ctx_msg(ctx) + "\n\nAnswer the user's questions using these results."},
                        {"role": "assistant", "content": "Understood. I will explain using only these results."}]
                for q, a in history[-6:]:
                    msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                msgs.append({"role": "user", "content": question})
                txt = self._call(msgs)
                self.last_error = None
                return txt, self.label
            except Exception as e:
                self.last_error = str(e)
        note = f" (online AI unavailable: {self.last_error})" if self.last_error else ""
        return offline_answer(question, ctx), "Built-in explainer (offline)" + note


EXPLAINER = Explainer()


# --------------------------------------------------------------------------- #
# Presenting results (cards, figures, report file)
# --------------------------------------------------------------------------- #
def _url(path):
    rel = os.path.relpath(path, OUT_ROOT).replace("\\", "/")
    return "/out/" + quote(rel)


def build_payload(res, briefing, ai_label):
    kpis, figs = [], []
    fr, dr, vr = res["flood"], res["forest"], res["vuln"]
    if fr:
        kpis.append({"label": "Newly flooded", "value": _km2(fr["flooded_area_km2"]),
                     "sub": f"{fr['flooded_percent_of_scene']:.1f}% of the area", "tone": "blue"})
        kpis.append({"label": "Permanent water (excluded)", "value": _km2(fr["permanent_water_km2"]),
                     "sub": "rivers and lakes already present", "tone": "grey"})
    if dr:
        kpis.append({"label": "Forest disturbed", "value": f"{dr['disturbed_area_ha']:,.0f} ha",
                     "sub": f"{dr['percent_of_forest_disturbed']:.1f}% of forest", "tone": "orange"})
        kpis.append({"label": "Probable forest loss", "value": _km2(dr["probable_forest_loss_km2"]),
                     "sub": "large radar drop (6 dB or more)", "tone": "red"})
    if vr:
        kpis.append({"label": "High vulnerability", "value": _km2(vr["area_by_class_km2"]["High"]),
                     "sub": f"{vr['percent_by_class']['High']}% of the area", "tone": "red"})
        top = max(vr["regions"], key=lambda r: r["mean_score"])
        kpis.append({"label": "Most vulnerable region", "value": top["region"],
                     "sub": f"score {top['mean_score']}/100 ({top['vulnerability']})", "tone": "red"})

    def fig(title, path, cap):
        if path and os.path.exists(path):
            figs.append({"title": title, "url": _url(path), "caption": cap})

    if fr:
        o = fr["outputs"]
        fig("Flood extent map", o.get("flood_map"),
            "Red = newly flooded land. Blue = permanent water (rivers/lakes), excluded from the flood total. Grey = radar image of land.")
        fig("Detected water (after the event)", o.get("water"),
            "Everything blue is open water after the event: permanent water plus new flooding.")
    if dr:
        o = dr["outputs"]
        fig("Forest change map", o.get("forest_change_map"),
            "Red = probable forest loss, orange = possible disturbance, green = stable forest, blue = water, grey = non-forest.")
        fig("Radar change (dB)", o.get("change_db"),
            "Red areas got darker between the two dates (backscatter decrease), which is the signal used to find forest loss.")
    if vr:
        o = vr["outputs"]
        fig("Flood vulnerability classes", o.get("classes"),
            "Red = High, yellow = Medium, green = Low vulnerability. Letters mark regions; dark blue is the river network.")
        fig("Regions ranked", o.get("ranking"),
            "Average vulnerability score of each region, highest at the top of the chart.")
        fig("What drives the score", o.get("factors"),
            "One panel per factor. Darker red = that factor makes the land more vulnerable there.")

    regions = []
    if vr:
        for r in sorted(vr["regions"], key=lambda r: -r["mean_score"]):
            regions.append({"region": r["region"], "vulnerability": r["vulnerability"],
                            "score": r["mean_score"], "high_km2": r["high_area_km2"],
                            "high_pct": r["high_percent"],
                            "drivers": ", ".join(DRIVER_TEXT[k] for k in r.get("top_drivers", []))})
    return {"demo": res["demo"], "kpis": kpis, "figures": figs, "regions": regions,
            "briefing_html": md_to_html(briefing), "ai": ai_label,
            "seconds": res["seconds"], "analyses": res["analyses"]}


def write_report(res, briefing, payload):
    """One shareable HTML file with the images embedded."""
    parts = [f"<h1>NISAR EcoFlood AI report</h1><p>Generated {datetime.now():%d %b %Y %H:%M}. "
             f"Data: {html.escape(res['ctx']['data_source'])}.</p>"]
    parts.append("<div class=k>" + "".join(
        f"<div><small>{html.escape(k['label'])}</small><b>{html.escape(k['value'])}</b>"
        f"<small>{html.escape(k['sub'])}</small></div>" for k in payload["kpis"]) + "</div>")
    parts.append("<h2>AI briefing</h2>" + md_to_html(briefing))
    for f in payload["figures"]:
        p = os.path.join(OUT_ROOT, unquote(f["url"][len("/out/"):]))
        try:
            with open(p, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            parts.append(f"<figure><img src='data:image/png;base64,{b64}'><figcaption><b>{html.escape(f['title'])}</b>"
                         f" - {html.escape(f['caption'])}</figcaption></figure>")
        except OSError:
            pass
    if payload["regions"]:
        rows = "".join(f"<tr><td>{r['region']}</td><td>{r['vulnerability']}</td><td>{r['score']}</td>"
                       f"<td>{r['high_km2']}</td><td>{html.escape(r['drivers'])}</td></tr>" for r in payload["regions"])
        parts.append("<h2>Regions</h2><table><tr><th>Region</th><th>Vulnerability</th><th>Score</th>"
                     f"<th>High area km&sup2;</th><th>Main drivers</th></tr>{rows}</table>")
    doc = ("<!doctype html><meta charset=utf-8><title>EcoFlood AI report</title><style>"
           "body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1b2430}"
           "img{max-width:100%}figure{margin:18px 0}figcaption{font-size:13px;color:#555}"
           ".k{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}"
           ".k div{border:1px solid #ddd;border-radius:8px;padding:10px}.k b{display:block;font-size:20px}"
           "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px;text-align:left}"
           "</style>" + "".join(parts))
    path = os.path.join(res["run_dir"], "EcoFlood_Report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def finish_run(res):
    """Run the AI explanation and package everything for the UI / terminal."""
    briefing, label = EXPLAINER.briefing(res["ctx"])
    payload = build_payload(res, briefing, label)
    report = write_report(res, briefing, payload)
    payload["report_url"] = _url(report)
    res["briefing"], res["payload"] = briefing, payload
    with open(os.path.join(res["run_dir"], "ai_briefing.md"), "w", encoding="utf-8") as f:
        f.write(briefing)
    with open(os.path.join(res["run_dir"], "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res["ctx"], f, indent=2, default=float)
    return res


# --------------------------------------------------------------------------- #
# Web app
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NISAR EcoFlood AI</title>
<style>
:root{--bg:#f4f7fa;--card:#fff;--ink:#1b2430;--mute:#5d6b7a;--line:#e2e8ef;--brand:#0b6e99;--brand2:#2e7d32;
--blue:#1976d2;--orange:#ef8a00;--red:#d32f2f;--grey:#6b7785}
@media (prefers-color-scheme:dark){:root{--bg:#10161d;--card:#18212b;--ink:#e8eef4;--mute:#9aa9b8;--line:#2a3745}}
*{box-sizing:border-box}body{margin:0;font-family:"Segoe UI",system-ui,Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.5}
header{background:linear-gradient(120deg,#0b6e99,#2e7d32);color:#fff;padding:22px 20px}
header h1{margin:0;font-size:26px}header p{margin:4px 0 0;opacity:.9}
main{max-width:1100px;margin:0 auto;padding:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
h2{margin:0 0 10px;font-size:18px}h3{margin:14px 0 4px;font-size:16px;color:var(--brand)}
button{font:inherit;cursor:pointer;border:0;border-radius:8px;padding:10px 16px;background:var(--brand);color:#fff;font-weight:600}
button.sec{background:transparent;color:var(--brand);border:1px solid var(--brand)}button:disabled{opacity:.5;cursor:default}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.badge{display:inline-block;font-size:12px;padding:3px 9px;border-radius:99px;background:var(--line);color:var(--mute)}
.badge.live{background:#d9f0dd;color:#1b5e20}
.demo{background:#fff4d6;color:#7a5a00;padding:8px 12px;border-radius:8px;font-size:14px;margin-bottom:12px}
details{margin-top:12px}summary{cursor:pointer;color:var(--brand);font-weight:600}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin-top:10px}
label{font-size:13px;color:var(--mute);display:block}input,select{width:100%;padding:8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.kpi{border-radius:10px;padding:12px;border:1px solid var(--line);border-left:5px solid var(--grey);background:var(--card)}
.kpi small{display:block;color:var(--mute);font-size:12px}.kpi b{font-size:21px}
.kpi.blue{border-left-color:var(--blue)}.kpi.orange{border-left-color:var(--orange)}.kpi.red{border-left-color:var(--red)}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--card)}
figure img{width:100%;display:block;background:#fff}figcaption{padding:10px;font-size:13px;color:var(--mute)}figcaption b{color:var(--ink)}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left}
.pill{padding:2px 9px;border-radius:99px;color:#fff;font-size:12px;font-weight:600}
.High{background:var(--red)}.Medium{background:#e0a800}.Low{background:#2e9b3f}
.brief p,.brief li{margin:4px 0}.brief ul{padding-left:20px;margin:4px 0}
#log{font-family:Consolas,monospace;font-size:13px;color:var(--mute);white-space:pre-wrap}
.spin{width:18px;height:18px;border:3px solid var(--line);border-top-color:var(--brand);border-radius:50%;display:inline-block;animation:r 1s linear infinite;vertical-align:middle;margin-right:8px}
@keyframes r{to{transform:rotate(360deg)}}
#chatbox{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:10px;padding:10px;margin-bottom:10px;background:var(--bg)}
.msg{margin:8px 0;padding:8px 12px;border-radius:10px;max-width:92%}.msg.u{background:var(--brand);color:#fff;margin-left:auto;width:fit-content}
.msg.a{background:var(--card);border:1px solid var(--line)}.msg small{display:block;color:var(--mute);font-size:11px;margin-top:4px}
.chips button{background:var(--line);color:var(--ink);font-weight:400;font-size:13px;padding:6px 10px}
.err{color:var(--red);font-weight:600}.hidden{display:none}
</style></head><body>
<header><h1>🌊🌳 NISAR EcoFlood <span style="font-weight:400">AI</span></h1>
<p>Flood mapping &middot; Deforestation detection &middot; Flood vulnerability &mdash; run and explained by AI</p></header>
<main>
<div class="card" id="start">
  <div class="row"><button id="demoBtn" onclick="run('demo')">▶ Run demo analysis</button>
  <span class="badge" id="aiBadge">AI: ...</span></div>
  <p style="color:var(--mute);margin:10px 0 0">One click runs flood detection, forest-change detection and vulnerability scoring,
  then the AI shows the results and explains them. The demo uses a synthetic area so you can see everything working.</p>
  <details><summary>Analyse my own satellite files</summary>
   <p style="color:var(--mute);font-size:14px">Paste full file paths (GeoTIFF or .npy), e.g. <code>C:\Users\HP\Downloads\pre.tif</code>.
   Flood needs a before/after pair. Forest uses the same pair unless you give another. DEM, rainfall and soil moisture are optional but improve the vulnerability score.</p>
   <div class="grid2">
    <div><label>Radar BEFORE (flood pair)</label><input id="pre"></div>
    <div><label>Radar AFTER (flood pair)</label><input id="post"></div>
    <div><label>Forest radar BEFORE (optional)</label><input id="before"></div>
    <div><label>Forest radar AFTER (optional)</label><input id="after"></div>
    <div><label>DEM / elevation (optional)</label><input id="dem"></div>
    <div><label>Rainfall in mm (optional)</label><input id="rainfall"></div>
    <div><label>Soil moisture (optional)</label><input id="soil_moisture"></div>
    <div><label>Pixel size in metres (only if the files have no geo-info)</label><input id="pixel_size" type="number"></div>
    <div><label>Polarisation</label><select id="pol"><option value="hv">HV (best for forest)</option><option value="hh">HH</option></select></div>
   </div>
   <p><button class="sec" onclick="run('files')">Run on my files</button></p>
  </details>
</div>

<div class="card hidden" id="progress"><span class="spin" id="spin"></span><b id="pmsg">Working...</b><div id="log"></div></div>

<div id="results" class="hidden">
  <div class="demo hidden" id="demoNote">This run uses <b>synthetic demo data</b>, so the numbers are illustrative and do not describe a real place.</div>
  <div class="card"><h2>Key numbers</h2><div class="kpis" id="kpis"></div></div>
  <div class="card"><div class="row" style="justify-content:space-between"><h2>🤖 AI briefing</h2><span class="badge live" id="aiLabel"></span></div>
   <div class="brief" id="briefing"></div></div>
  <div class="card"><h2>Maps and charts</h2><div class="figs" id="figs"></div></div>
  <div class="card" id="regCard"><h2>Regions</h2><div style="overflow:auto"><table id="regTable"></table></div></div>
  <div class="card"><h2>💬 Ask the AI</h2>
   <div id="chatbox"></div>
   <div class="chips row" id="chips"></div>
   <div class="row" style="margin-top:10px"><input id="q" placeholder="Ask anything about the results..." style="flex:1" onkeydown="if(event.key==='Enter')ask()">
   <button onclick="ask()">Ask</button></div></div>
  <div class="card"><a id="reportLink" href="#" target="_blank">⬇ Open the full shareable report</a>
   <span style="color:var(--mute)"> &nbsp;(a single HTML file with maps and the AI briefing; also saved in the output folder)</span></div>
</div>
</main>
<script>
const $=id=>document.getElementById(id);
const CHIPS=["Explain the results simply","Which region is most at risk and why?","How are forest loss and flooding linked?","How reliable are these results?","What should be done first?"];
fetch('/api/info').then(r=>r.json()).then(i=>{$('aiBadge').textContent='AI: '+i.ai;if(i.online)$('aiBadge').classList.add('live');});
async function run(mode){
  const params={};for(const id of ['pre','post','before','after','dem','rainfall','soil_moisture','pixel_size','pol'])params[id]=$(id).value;
  $('demoBtn').disabled=true;$('results').classList.add('hidden');$('progress').classList.remove('hidden');
  $('spin').style.display='inline-block';$('pmsg').textContent='Starting...';$('pmsg').className='';$('log').textContent='';
  const r=await fetch('/api/run',{method:'POST',body:JSON.stringify({mode,params})});
  if(!r.ok&&r.status!==409){showErr('Could not start');return;}poll();
}
function showErr(m){$('spin').style.display='none';$('pmsg').textContent='Problem: '+m;$('pmsg').className='err';$('demoBtn').disabled=false;}
async function poll(){
  const s=await (await fetch('/api/status')).json();
  $('log').textContent=s.log.join('\n');$('pmsg').textContent=s.message||'Working...';
  if(s.state==='running'){setTimeout(poll,700);return;}
  $('demoBtn').disabled=false;
  if(s.state==='error'){showErr(s.message);return;}
  if(s.state==='done'){$('spin').style.display='none';$('pmsg').textContent='Done in '+s.result.seconds+' s';render(s.result);}
}
function render(r){
  $('results').classList.remove('hidden');$('demoNote').classList.toggle('hidden',!r.demo);
  $('kpis').innerHTML=r.kpis.map(k=>`<div class="kpi ${k.tone}"><small>${k.label}</small><b>${k.value}</b><small>${k.sub}</small></div>`).join('');
  $('briefing').innerHTML=r.briefing_html;$('aiLabel').textContent=r.ai;
  $('figs').innerHTML=r.figures.map(f=>`<figure><a href="${f.url}" target="_blank"><img src="${f.url}" loading="lazy"></a><figcaption><b>${f.title}</b><br>${f.caption}</figcaption></figure>`).join('');
  $('regCard').classList.toggle('hidden',!r.regions.length);
  $('regTable').innerHTML='<tr><th>Region</th><th>Vulnerability</th><th>Score</th><th>High area (km²)</th><th>Main drivers</th></tr>'+
    r.regions.map(x=>`<tr><td><b>${x.region}</b></td><td><span class="pill ${x.vulnerability}">${x.vulnerability}</span></td><td>${x.score}</td><td>${x.high_km2} (${x.high_pct}%)</td><td>${x.drivers}</td></tr>`).join('');
  $('reportLink').href=r.report_url;
  $('chatbox').innerHTML='';addMsg('a','Hi! I have run the analysis. Ask me anything: why a region is high, what a number means, or what to do next.');
  $('chips').innerHTML=CHIPS.map(c=>`<button onclick="ask(this.textContent)">${c}</button>`).join('');
  $('results').scrollIntoView({behavior:'smooth'});
}
function addMsg(who,html,small){const d=document.createElement('div');d.className='msg '+who;d.innerHTML=html+(small?`<small>${small}</small>`:'');$('chatbox').appendChild(d);$('chatbox').scrollTop=1e9;return d;}
async function ask(text){
  const q=(text||$('q').value).trim();if(!q)return;$('q').value='';
  addMsg('u',q.replace(/</g,'&lt;'));const w=addMsg('a','<i>thinking...</i>');
  try{const r=await (await fetch('/api/chat',{method:'POST',body:JSON.stringify({question:q})})).json();
    w.innerHTML=r.answer_html+`<small>${r.ai}</small>`;}catch(e){w.textContent='Sorry, something went wrong.';}
  $('chatbox').scrollTop=1e9;
}
</script></body></html>
"""

STATE = {"state": "idle", "message": "", "log": [], "result": None, "ctx": None, "history": []}
LOCK = threading.Lock()


def _log(msg):
    print("  " + msg, flush=True)
    with LOCK:
        STATE["log"].append(msg)
        STATE["message"] = msg


def _worker(cfg):
    try:
        res = run_full_analysis(cfg, _log)
        res = finish_run(res)
        with LOCK:
            STATE.update(state="done", message="Done", result=res["payload"], ctx=res["ctx"], history=[])
    except Exception as e:
        traceback.print_exc()
        with LOCK:
            STATE.update(state="error", message=str(e) or e.__class__.__name__)


def start_run(cfg) -> bool:
    with LOCK:
        if STATE["state"] == "running":
            return False
        STATE.update(state="running", message="Starting...", log=[], result=None, ctx=None, history=[])
    threading.Thread(target=_worker, args=(cfg,), daemon=True).start()
    return True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=float))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/info":
            return self._json({"ai": EXPLAINER.label, "online": bool(EXPLAINER.key)})
        if path == "/api/status":
            with LOCK:
                return self._json({"state": STATE["state"], "message": STATE["message"],
                                   "log": STATE["log"], "result": STATE["result"]})
        if path.startswith("/out/"):
            full = os.path.realpath(os.path.join(OUT_ROOT, unquote(path[5:])))
            if not full.startswith(os.path.realpath(OUT_ROOT) + os.sep) or not os.path.isfile(full):
                return self._send(404, "not found", "text/plain")
            ctype = {".png": "image/png", ".html": "text/html; charset=utf-8", ".json": "application/json",
                     ".csv": "text/csv", ".md": "text/plain; charset=utf-8"}.get(
                os.path.splitext(full)[1].lower(), "application/octet-stream")
            with open(full, "rb") as f:
                return self._send(200, f.read(), ctype)
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json({"error": "bad request"}, 400)
        path = urlparse(self.path).path
        if path == "/api/run":
            cfg = {"mode": body.get("mode", "demo"), **(body.get("params") or {})}
            started = start_run(cfg)
            return self._json({"started": started}, 200 if started else 409)
        if path == "/api/chat":
            with LOCK:
                ctx, hist = STATE["ctx"], STATE["history"]
            q = str(body.get("question", "")).strip()[:1000]
            if not ctx:
                return self._json({"answer_html": "<p>Run an analysis first, then I can explain it.</p>", "ai": ""})
            ans, label = EXPLAINER.answer(q, ctx, hist)
            with LOCK:
                STATE["history"].append((q, ans))
            return self._json({"answer_html": md_to_html(ans), "ai": label})
        self._send(404, "not found", "text/plain")


def serve(port, open_browser, auto_cfg=None):
    os.makedirs(OUT_ROOT, exist_ok=True)
    srv = None
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            port = p
            break
        except OSError:
            continue
    if srv is None:
        print("[error] Could not open a local port. Close other programs or use --port 9000")
        sys.exit(1)
    url = f"http://127.0.0.1:{port}/"
    print("\n=== NISAR EcoFlood AI ===")
    print(f"Dashboard : {url}")
    print(f"AI mode   : {EXPLAINER.label}")
    print(f"Outputs   : {OUT_ROOT}")
    print("Press Ctrl+C in this terminal to stop.\n")
    if auto_cfg:
        start_run(auto_cfg)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.server_close()


# --------------------------------------------------------------------------- #
# Terminal mode
# --------------------------------------------------------------------------- #
def cli(cfg, chat):
    try:
        res = run_full_analysis(cfg, lambda m: print("  " + m, flush=True))
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(f"\n[error] {e}\n")
        sys.exit(1)
    res = finish_run(res)
    print("\n" + "=" * 70 + f"\n AI BRIEFING   [{res['payload']['ai']}]\n" + "=" * 70)
    print(re.sub(r"\*\*", "", res["briefing"]).replace("## ", "\n>> "))
    print("\nFull report :", os.path.join(res["run_dir"], "EcoFlood_Report.html"))
    print("All outputs :", res["run_dir"])
    if chat:
        print("\nAsk the AI about the results (empty line or 'exit' to quit).")
        hist = []
        while True:
            try:
                q = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            ans, label = EXPLAINER.answer(q, res["ctx"], hist)
            hist.append((q, ans))
            print(f"\nAI [{label}]\n" + re.sub(r"\*\*", "", ans).replace("## ", ">> "))


def main():
    ap = argparse.ArgumentParser(description="NISAR EcoFlood AI - one-click analysis with AI explanation")
    ap.add_argument("--cli", action="store_true", help="no browser: run and print the AI briefing")
    ap.add_argument("--chat", action="store_true", help="with --cli: ask questions in the terminal")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--ask-key", action="store_true", help="type an Anthropic API key (not saved)")
    ap.add_argument("--model", help="Claude model name (default claude-sonnet-5-5)")
    ap.add_argument("--pre"), ap.add_argument("--post")
    ap.add_argument("--before"), ap.add_argument("--after")
    ap.add_argument("--dem"), ap.add_argument("--rainfall"), ap.add_argument("--soil-moisture")
    ap.add_argument("--pixel-size", type=float)
    ap.add_argument("--pol", choices=["hv", "hh"], default="hv")
    ap.add_argument("--band", type=int, default=1)
    a = ap.parse_args()

    if a.ask_key:
        EXPLAINER.key = getpass.getpass("Anthropic API key (input hidden, not saved): ").strip()
    if a.model:
        EXPLAINER.model = a.model

    has_data = any([a.pre, a.post, a.before, a.after, a.dem, a.rainfall, a.soil_moisture])
    cfg = {"mode": "files" if has_data else "demo", "pre": a.pre, "post": a.post,
           "before": a.before, "after": a.after, "dem": a.dem, "rainfall": a.rainfall,
           "soil_moisture": a.soil_moisture, "pixel_size": a.pixel_size, "pol": a.pol, "band": a.band}

    if a.cli:
        cli(cfg, a.chat)
    else:
        serve(a.port, not a.no_browser, cfg if has_data else None)


if __name__ == "__main__":
    main()
