@echo off
setlocal
set "PSModulePath="
chcp 65001 >nul
set "HGV_CHECK="
if /I "%~1"=="--check" set "HGV_CHECK=-CheckOnly"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0OneClick-HGV.ps1" %HGV_CHECK%
set "HGV_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %HGV_EXIT%
