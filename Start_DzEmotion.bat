@echo off
setlocal EnableDelayedExpansion
title DzEmotion - Ultimate One-Click Installer ^& Launcher
color 0b

echo ===============================================================
echo                DzEmotion Installation ^& Launcher
echo ===============================================================
echo This script will automatically download and install:
echo 1. Python (if missing)
echo 2. Ollama AI Engine (if missing)
echo 3. The Gemma 3:4b Local AI Model (if missing)
echo 4. All Python Dependencies
echo ===============================================================
echo.

:: ------------------------------------------------------------------
:: 1. CHECK AND INSTALL PYTHON
:: ------------------------------------------------------------------
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Python is not installed. Downloading Python 3.11...
    curl -L -o python_installer.exe https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe
    
    echo [INFO] Installing Python silently... 
    start /wait python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    
    echo [OK] Python installed. We will use it locally for this session.
    
    :: Try to add standard user path temporarily so script doesn't fail
    set "PATH=%LocalAppData%\Programs\Python\Python311\;%LocalAppData%\Programs\Python\Python311\Scripts\;%PATH%"
    
    :: Double check
    python --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Python was installed but still not found in PATH. 
        echo Please restart your computer and run this script again.
        pause
        exit /b
    )
) else (
    echo [OK] Python is already installed.
)


:: ------------------------------------------------------------------
:: 2. CHECK AND INSTALL OLLAMA
:: ------------------------------------------------------------------
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Ollama is not installed. Downloading Ollama Setup...
    curl -L -o OllamaSetup.exe https://ollama.com/download/OllamaSetup.exe
    
    echo [INFO] Please follow the Ollama installation window that appears.
    start /wait OllamaSetup.exe
    
    echo [OK] Ollama installed. Adding to temporary path just in case...
    set "PATH=%LocalAppData%\Programs\Ollama;%PATH%"
    
    :: Wait a few seconds for the Ollama background service to start automatically
    timeout /t 5 /nobreak >nul
) else (
    echo [OK] Ollama is already installed.
)


:: ------------------------------------------------------------------
:: 3. PULL GEMMA 3:4B MODEL
:: ------------------------------------------------------------------
echo.
echo [INFO] Checking local AI model...
:: We use pull to download if missing, or update if exists. This might take a while if new.
echo Pulling gemma3:4b from Ollama (This is a ~2.5GB download if first time)...
ollama pull gemma3:4b
if %errorlevel% neq 0 (
    echo [ERROR] Failed to pull the gemma3:4b model. Make sure Ollama is running in your taskbar!
    pause
    exit /b
)
echo [OK] Model gemma3:4b is ready.


:: ------------------------------------------------------------------
:: 4. SETUP PYTHON VIRTUAL ENVIRONMENT
:: ------------------------------------------------------------------
echo.
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating an isolated Python environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment. 
        pause
        exit /b
    )
    echo [OK] Virtual environment created successfully.
)

:: Activate the environment
call venv\Scripts\activate.bat

:: Install Requirements
echo [INFO] Checking and installing Python dependencies (this may take a few minutes)...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies from requirements.txt.
    pause
    exit /b
)
echo [OK] All dependencies are installed!
echo.


:: ------------------------------------------------------------------
:: 5. START DASHBOARD
:: ------------------------------------------------------------------
echo ===============================================================
echo Starting the DzEmotion Dashboard...
echo Press CTRL+C in this window to stop the server.
echo ===============================================================
echo.

streamlit run app.py

pause
