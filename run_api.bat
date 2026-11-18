@echo off
echo ========================================================
echo  Starting PathSnap Distributed Backend (Laptop 1)
echo ========================================================
echo.
call .venv\Scripts\activate.bat
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
pause
