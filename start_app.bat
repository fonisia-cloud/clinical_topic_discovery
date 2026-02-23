@echo off
setlocal

REM Clinical Topic Discovery one-click launcher (no activate dependency)
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "LOG_FILE=start_app.log"

echo [%date% %time%] START > "%LOG_FILE%"
echo [INFO] Working dir: %cd%

if not exist "%VENV_PY%" (
    echo [INFO] First run: creating virtual environment...
    echo [%date% %time%] Creating venv >> "%LOG_FILE%"
    python -m venv .venv >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. See %LOG_FILE%
        pause
        exit /b 1
    )

    echo [INFO] Installing dependencies...
    echo [%date% %time%] Installing dependencies >> "%LOG_FILE%"
    "%VENV_PY%" -m pip install -r requirements.txt >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. See %LOG_FILE%
        pause
        exit /b 1
    )
) else (
    echo [INFO] Using existing environment. Skipping dependency install.
    echo [%date% %time%] Reusing existing venv >> "%LOG_FILE%"
)

echo [INFO] Starting app on http://localhost:8501
echo [%date% %time%] Launching streamlit >> "%LOG_FILE%"
start "" "http://localhost:8501"
"%VENV_PY%" -m streamlit run app.py --server.port 8501 --server.headless true >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [ERROR] App stopped unexpectedly. See %LOG_FILE%
    pause
    exit /b 1
)

endlocal
