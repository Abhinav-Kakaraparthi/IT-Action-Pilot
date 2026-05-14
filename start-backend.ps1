$ErrorActionPreference = "Stop"
Set-Location "C:\Users\surya\Downloads\action-agent-app\backend"
& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8002
