@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_console.ps1" %*
if errorlevel 1 (
  echo.
  echo Startup failed. See the message above and the console logs.
  pause
  exit /b 1
)
exit /b 0
