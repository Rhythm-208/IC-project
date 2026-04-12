@echo off
echo ============================================
echo  Neural Network Training Visualizer - Build
echo ============================================
echo.

gcc -o nn_vis.exe main_vis.c -lgdi32 -lm -O2 -mwindows

if %ERRORLEVEL% EQU 0 (
    echo  BUILD SUCCESSFUL!
    echo  Run:  nn_vis.exe
    echo.
) else (
    echo  BUILD FAILED!
    echo.
)
