@echo off
setlocal enabledelayedexpansion
rem Stops the MoneyPrinterTurbo WebUI (and its worker/child processes).
rem
rem Only targets this project's Streamlit process - matched by its command line
rem running "webui\Main.py" - so unrelated Python apps (e.g. shortsmaker) are
rem left untouched. Use this after editing code so the next launch of
rem webui.bat picks up the new version instead of running stale in-memory code.

echo ***** Looking for MoneyPrinterTurbo WebUI processes... *****

set "FOUND="
for /f %%P in ('powershell -NoProfile -ExecutionPolicy Bypass -Command ^"Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" ^| Where-Object { $_.CommandLine -and $_.CommandLine -match 'Main\.py' -and $_.CommandLine -match 'streamlit' } ^| Select-Object -ExpandProperty ProcessId^"') do (
    set "FOUND=1"
    echo   stopping PID %%P ^(and child processes^)
    taskkill /PID %%P /T /F >nul 2>nul
)

if not defined FOUND (
    echo ***** No running MoneyPrinterTurbo WebUI found. *****
) else (
    echo ***** Stopped. Relaunch with webui.bat to load the latest code. *****
)

endlocal
