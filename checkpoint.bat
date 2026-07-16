@echo off
REM One-click Git checkpoint with timestamp
cd /d "A:\Pocket TTS-GPU"

REM Stage all changes
git add -A

REM Check if there are changes to commit
git diff --cached --quiet
if %errorlevel%==0 (
    echo No changes to checkpoint.
    pause
    exit /b 0
)

REM Generate timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set datetime=%%I
if "%datetime%"=="" (
    for /f %%I in ('powershell -command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"') do set timestamp=%%I
) else (
    set timestamp=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2% %datetime:~8,2%:%datetime:~10,2%:%datetime:~12,2%
)

REM Commit with timestamp
git commit -m "Checkpoint: %timestamp%"

echo.
echo Checkpoint created: %timestamp%
pause
