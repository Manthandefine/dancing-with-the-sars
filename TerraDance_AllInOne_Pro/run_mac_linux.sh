#!/bin/bash
cd "$(dirname "$0")"
PY=python3
for v in 3.12 3.11 3.13 3.10; do
  if command -v python$v >/dev/null 2>&1; then PY=python$v; break; fi
done
echo "Using: $PY"
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo "Starting TerraDance Pro All-In-One..."
python -m streamlit run app.py
