@echo off
setlocal
title Crop Recommendation Launcher
cd /d "%~dp0"

echo Dang khoi dong Crop Recommendation...
echo Thu muc: "%CD%"
echo.

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Khong tim thay Python trong .venv\Scripts\python.exe.
    echo Vui long thiet lap moi truong theo README.md truoc khi chay.
    pause
    exit /b 1
)

echo Dang kiem tra moi truong Python...
"%PYTHON_EXE%" -c "import fastapi, uvicorn, streamlit, requests"
if errorlevel 1 (
    echo Moi truong .venv thieu thu vien hoac khong khoi dong duoc.
    echo Kiem tra moi truong va huong dan cai dat trong README.md.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import requests; r = requests.get('http://127.0.0.1:8000/health', timeout=2); raise SystemExit(0 if r.ok and 'model_loaded' in r.json() else 1)" >nul 2>&1
if errorlevel 1 (
    echo Dang mo Backend...
    start "Crop Backend" "%ComSpec%" /d /k ""%PYTHON_EXE%" -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000"
) else (
    echo Backend da chay, su dung lai dich vu hien tai.
)

"%PYTHON_EXE%" -c "import requests; r = requests.get('http://127.0.0.1:8501/_stcore/health', timeout=2); raise SystemExit(0 if r.ok and r.text.strip() == 'ok' else 1)" >nul 2>&1
if errorlevel 1 (
    echo Dang mo Frontend...
    start "Crop Frontend" "%ComSpec%" /d /k ""%PYTHON_EXE%" -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true"
) else (
    echo Frontend da chay, su dung lai dich vu hien tai.
)

echo Dang cho hai dich vu san sang...
powershell.exe -NoProfile -Command "$deadline = (Get-Date).AddSeconds(45); while ((Get-Date) -lt $deadline) { try { $apiHealth = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2; $uiHealth = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 2; if ($apiHealth.model_loaded -eq $true -and $uiHealth.StatusCode -eq 200 -and $uiHealth.Content.Trim() -eq 'ok') { exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }; Write-Host 'Khong the khoi dong day du hai dich vu sau 45 giay.'; exit 1"
if errorlevel 1 (
    echo Vui long xem loi trong cua so Backend va Frontend.
    echo Backend can nap duoc model; cac cong 8000 va 8501 can dung cho ung dung nay.
    pause
    exit /b 1
)

echo.
echo Ung dung da san sang. Dang mo trinh duyet...
echo Giao dien: http://localhost:8501
echo Tai lieu API: http://127.0.0.1:8000/docs
start "" /b "%ComSpec%" /d /c start "" "http://localhost:8501"
echo Nhan Ctrl+C trong cua so Backend va Frontend de dung dich vu.
echo Nhan phim bat ky de dong cua so khoi chay nay.
pause >nul
endlocal
exit /b 0
