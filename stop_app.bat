@echo off
setlocal

REM Stop Streamlit process listening on port 8501
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo [INFO] Attempted to stop app on port 8501.
pause

endlocal
