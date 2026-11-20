@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (set "PY=py") else (set "PY=python")
%PY% --version >nul 2>nul
if errorlevel 1 (
 echo Python 3.10+ was not found. Install Python and enable Add Python to PATH.
 pause
 exit /b 1
)
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
if not exist ".env" copy ".env.example" ".env" >nul
if not exist "backend\uploads" mkdir "backend\uploads"
echo.
echo Installation complete. Run start.bat.
pause
