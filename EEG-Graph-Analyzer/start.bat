@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run install.bat first.
 pause
 exit /b 1
)
call ".venv\Scripts\activate.bat"
echo EEG Agent: http://127.0.0.1:8000
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
