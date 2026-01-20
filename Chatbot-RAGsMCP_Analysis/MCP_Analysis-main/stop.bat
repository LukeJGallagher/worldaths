@echo off
echo Starting clean shutdown of all MCP processes...

echo Finding processes on port 8002 (GitHub server)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8002') do (
  if not "%%a" == "" (
    echo Killing process with PID: %%a
    taskkill /F /PID %%a
  )
)

echo Finding processes on port 8004 (PostgreSQL server)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8004') do (
  if not "%%a" == "" (
    echo Killing process with PID: %%a
    taskkill /F /PID %%a
  )
)

echo Finding processes on port 8006 (MySQL server)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8006') do (
  if not "%%a" == "" (
    echo Killing process with PID: %%a
    taskkill /F /PID %%a
  )
)

echo Finding Streamlit processes on port 8501...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501') do (
  if not "%%a" == "" (
    echo Killing process with PID: %%a
    taskkill /F /PID %%a
  )
)

echo Killing any remaining Python processes...
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM pythonw.exe /T 2>nul

echo Waiting for processes to terminate...
timeout /t 2 /nobreak > nul

echo All MCP processes have been terminated.
echo Ports 8002, 8004, 8006, and 8501 should now be free.
echo You can now restart the MCP system. 