# Tidemark

A NASA Space Apps 2026 project for the **Dancing with the SARs** challenge. It compares two NISAR radar passes over the same place and shows where water appeared, receded or stayed, in a map with a before/after divider.

The project has two parts:

- `nisar_change.py` reads two NISAR GCOV files, finds changes, and writes image layers.
- `web/` is the viewer. It only shows what the script produced.

The files in `web/data` right now come from **synthetic** data, so you can see the app working before you touch real data. The viewer shows a yellow "Synthetic demo data" banner until you replace them.

## 1. Set up (10 minutes)

You need Python 3.10 or newer.

```bash
pip install -r requirements.txt
python nisar_change.py demo      # builds fake scenes and processes them (about 15 seconds)
python nisar_change.py serve     # opens http://localhost:8000
```

Drag the round handle on the map. If you see the viewer with a river, a flood and a yellow banner, your setup works. Use `python` instead of `python3` on Windows if needed.

If `pip install rasterio` fails on your machine, try `conda install -c conda-forge rasterio h5py`.

## 2. Pick your place and get an Earthdata token

1. Choose one place where water changed. Rivers and deltas work well. Good candidates are the Brahmaputra and Meghna basin, the Sundarbans, or the Godavari and Krishna deltas. A monsoon or flood event makes the best story.
2. Create a free account at <https://urs.earthdata.nasa.gov>.
3. After signing in, open **Generate Token** and copy the token.
4. Set it in your terminal:

```bash
# Mac or Linux
export EARTHDATA_TOKEN="paste-token-here"
# Windows PowerShell
$env:EARTHDATA_TOKEN="paste-token-here"
```

Tokens expire after some weeks, so generate a fresh one if downloads fail.

## 3. Find two passes

```bash
python nisar_change.py search --lon 90.41 --lat 23.81 --start 2025-11-01 --urls
```

Replace the longitude and latitude with your place. The output groups passes by **track and frame**. Pick two passes from the same group, ideally 12 days or more apart, with a flood or heavy rain between them. Passes from different tracks look at the ground from different angles and will not line up.

Check the size column before downloading. Files can be hundreds of megabytes or more.

If the search prints nothing useful, add `--debug` to see the raw field names, or search by hand at <https://search.asf.alaska.edu> (choose dataset NISAR, product GCOV) and copy the download links.

## 4. Download and inspect

```bash
python nisar_change.py download URL_BEFORE URL_AFTER --out data/real
python nisar_change.py inspect data/real/THE_FIRST_FILE.h5
```

`inspect` prints the layers in the file and confirms it found the `HHHH` and `HVHV` bands, the map projection, and the date. If it cannot find them, paste its output to me and we will adjust the script.

## 5. Make the change map

```bash
python nisar_change.py change \
  --before data/real/FIRST.h5 --after data/real/SECOND.h5 \
  --pol HH --bbox WEST SOUTH EAST NORTH \
  --title "Tidemark: your place name"
python nisar_change.py serve
```

- `--bbox` crops to a longitude and latitude box, for example `90.2 23.6 90.6 24.0`. Use it. It makes processing much faster and the map easier to read.
- `--looks 8` averages 8 by 8 pixels, giving about 160 m pixels. Use `--looks 4` for more detail on a small area.
- `--pol HH` is a good first choice at L-band. Try `--pol HV` too, and compare.

### Tune the water threshold

The script calls a pixel water when it is darker than **-20 dB** in HH (**-25 dB** in HV). That is a starting point, not a fact. Try these:

- `--threshold -18`, `--threshold -22`, and so on. Look at the map each time. Water should be filled in without speckle on land.
- `--threshold auto` picks a cutoff from the image histogram. It works when the area has plenty of both water and land.

Pick the value that matches what you can see in the radar image and the satellite background. Write the value and your reasoning in your submission.

## 6. Submit

- Put the project on GitHub. Keep `web/` (including `web/data`) and `nisar_change.py`. Leave out `data/` because the files are large.
- To publish the viewer, upload the `web/` folder to GitHub Pages, Netlify or Vercel. It is a static site with no server code.
- Record a 1 to 2 minute video: show the divider moving, explain what the colours mean, and state one number (for example, "new water covers 140 km² of the area imaged").
- List every resource you used, because NASA requires citations: NISAR data from ASF DAAC, Leaflet, OpenStreetMap, CARTO and Esri tiles, and the Python libraries.
- Say plainly what the tool cannot do. The viewer already lists the main limits (flooded vegetation, radar shadow, the threshold choice). Keep that honesty in your write-up.

## How the processing works

1. Read the HH (or HV) covariance layer, which is radar power on the ground.
2. Average blocks of pixels to reduce speckle noise, convert to decibels, and smooth lightly.
3. Mark pixels below the water threshold in each pass.
4. **New water**: not water before, water after, and at least 3 dB darker. **Receded water**: the reverse. **Permanent water**: water in both.
5. Drop tiny patches, reproject to latitude and longitude, count areas on the original metric grid, and write PNG layers and `meta.json`.

## Ideas if you have time

- A second location, or a second pair of dates, with a switcher in the viewer.
- Overlay a river or district boundary and report the area per district.
- Compare HH and HV results and show where they disagree.
- Add a time series: one small chart of water area across all available passes.

## Files

```
nisar_change.py      search, download, inspect, change, demo, serve
requirements.txt
web/index.html       the viewer
web/data/            layers written by nisar_change.py
web/vendor/leaflet/  map library, bundled so it works offline
```

The viewer's street map and satellite background need internet. The radar layers do not.
