@echo off
REM ProjectSearchBar launcher (Windows)
REM Usage: ProjectSearchBar\launch_windows.bat

setlocal ENABLEDELAYEDEXPANSION
cd /d %~dp0

if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)

if "%PROJECTSEARCHBAR_HOST%"=="" set PROJECTSEARCHBAR_HOST=127.0.0.1
if "%PROJECTSEARCHBAR_PORT%"=="" set PROJECTSEARCHBAR_PORT=8360
if "%PROJECTSEARCHBAR_LLM_TIMEOUT%"=="" set PROJECTSEARCHBAR_LLM_TIMEOUT=90

echo Launching ProjectSearchBar at http://%PROJECTSEARCHBAR_HOST%:%PROJECTSEARCHBAR_PORT%
py -3 launch.py
