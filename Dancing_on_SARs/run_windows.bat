@echo off
cd /d "%~dp0"
set "PY="
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=py -%%V"
  )
)
if not defined PY set "PY=python"
echo Using: %PY%
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo Starting TerraDance Pro All-In-One...
python -m streamlit run app.py
pause
