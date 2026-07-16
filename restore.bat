@echo off
REM Restore to last Git checkpoint
cd /d "A:\Pocket TTS-GPU"

echo ========================================
echo    Git Checkpoint Restore
echo ========================================
echo.

REM Show last 5 commits
echo Recent checkpoints:
echo ----------------------------------------
git log --oneline -5
echo ----------------------------------------
echo.

REM Confirm
set /p confirm="Restore ALL files to last checkpoint? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Restore cancelled.
    pause
    exit /b 0
)

REM Restore
git checkout HEAD -- .

echo.
echo Files restored to last checkpoint.
pause
