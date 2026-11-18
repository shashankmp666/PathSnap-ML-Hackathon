@echo off
cd /d "%~dp0"
echo ========================================================
echo  Starting PathSnap Distributed Backend (Laptop 1)
echo ========================================================
echo.
set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PY_EXE%" (
    set "PY_EXE=python"
)

echo [PathSnap] Using Python: %PY_EXE%
echo [PathSnap] Starting API server on http://0.0.0.0:8000 ...
echo [PathSnap] Keep this terminal open during evaluation!
echo.

"%PY_EXE%" -m uvicorn api.app:app --host 0.0.0.0 --port 8000
pause
