@ECHO OFF
ECHO Current directory: 
CD

RMDIR /S /Q ..\_BUILD
MKDIR ..\_BUILD

ECHO Building projects:
CD ..\_BUILD
"C:\Program Files\CMake\bin\cmake" -G "Visual Studio 17 2022" -A "x64" ..\projects

PAUSE
REM -D CMAKE_BUILD_TYPE=Debug   --config Debug
