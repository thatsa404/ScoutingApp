@echo off
REM ---------------------------------------------------------------------------
REM Double-click to open the interactive field calibrator.
REM
REM Click a point in the VIDEO window, then the matching point in the FIELD
REM window.  u = undo last pair, r = reset everything, s or q = save and compute.
REM
REM Points already saved for this video are pre-loaded and drawn GREY; anything
REM you add this session is YELLOW.  Pass --fresh to start from nothing.
REM
REM Floor-plane features ONLY -- tape intersections, carpet seams, guardrail
REM BASE corners.  A point with any height silently wrecks the fit.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set VIDEO=GSxbsE42o5o
if not "%~1"=="" set VIDEO=%~1

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   ERROR: .venv not found in %CD%
  echo   Create it first:  uv sync
  echo.
  pause
  exit /b 1
)

REM Some packages depend on opencv-python-headless, which shares the cv2 namespace
REM with opencv-python and overwrites its binaries -- leaving a build with GUI: NONE,
REM so cv2.imshow opens nothing and the calibrator dies silently. easyocr was the
REM culprit and is now gone, but the guard stays: it is a no-op when already correct,
REM costs ~1 s, and self-heals if anything pulls the headless build back in.
".venv\Scripts\python.exe" -c "import cv2,sys; sys.exit(0 if 'WIN32UI' in cv2.getBuildInformation().split('GUI:')[1][:400] else 1)" 2>nul
if errorlevel 1 (
  echo   OpenCV has no GUI backend ^(headless build won^) -- reinstalling...
  "%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" pip install --force-reinstall --no-deps opencv-python==5.0.0.93
)

echo.
echo   Calibrating %VIDEO%
echo   video window: click a floor point.  field window: click its match.
echo   u undo   r reset   s save   q quit
echo.

".venv\Scripts\python.exe" -m rtrack.calibrate %VIDEO% --interactive --plate

echo.
echo   Done. Renders written to out\stage2\:
echo     %VIDEO%_warp.png        warped video over the field render
echo     %VIDEO%_reproject.png   field outline drawn back on the video
echo     %VIDEO%_error_map.png   metres of error per pixel of click
echo.
pause
