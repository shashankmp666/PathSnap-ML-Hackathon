@echo off
echo ========================================================
echo  Starting PathSnap Evaluator Interface (Laptop 2)
echo ========================================================
echo.
set /p IP_ADDR="Enter Laptop 1 IP address (press Enter for localhost / single-laptop test): "
if "%IP_ADDR%"=="" set IP_ADDR=localhost

echo.
echo Opening http://%IP_ADDR%:8000 in your browser...
start "" "http://%IP_ADDR%:8000"
echo.
echo If the page shows "Connected to Laptop 1", you are ready!
echo.
pause
