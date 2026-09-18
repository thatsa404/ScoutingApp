@echo off
REM ---------------------------------------------------------------------------
REM Double-click to build the curation bundles and open the viewer.
REM
REM Two bundles are produced and they are DIFFERENT TOOLS:
REM
REM   <video>_curate_frames.json   FULL-FRAME view.  One picture per moment with
REM                                every robot boxed.  Click a box, press 1-6.
REM                                Everything in a frame is a different robot, so
REM                                each team can only be used once -- the viewer
REM                                strikes out teams already taken.  This is the
REM                                one that shows field elements like the fuel
REM                                pile, because it shows EVERY detection.
REM
REM   <video>_curate.json          PER-TRACK strips.  One row of crops following
REM                                a single track through time.  Better for
REM                                deciding WHERE a track changes robot.
REM
REM Drag whichever you want onto the viewer page.  Answers are kept separately
REM per mode, so you can use both.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set VIDEO=GSxbsE42o5o
set MATCH=2026necmp_f1m3
if not "%~1"=="" set VIDEO=%~1
if not "%~2"=="" set MATCH=%~2

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   ERROR: .venv not found in %CD%
  echo   Create it with:  uv sync
  echo.
  pause
  exit /b 1
)

echo.
echo   Building curation bundles for %VIDEO% / %MATCH%
echo.

".venv\Scripts\python.exe" -m rtrack.curate %VIDEO% --match %MATCH% --frames 16 --anchors 2
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m rtrack.curate %VIDEO% --match %MATCH%
if errorlevel 1 goto fail

echo.
echo   Opening the viewer. Drag ONE of these onto the page:
echo     out\stage3\%VIDEO%_curate_frames.json   full-frame, click boxes
echo     out\stage3\%VIDEO%_curate.json          per-track crop strips
echo.
start "" "viewer\curate.html"
start "" "%CD%\out\stage3"
pause
exit /b 0

:fail
echo.
echo   BUILD FAILED -- see the error above.
echo.
pause
exit /b 1
