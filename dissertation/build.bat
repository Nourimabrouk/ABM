@echo off
REM LaTeX Build Script for PhD Dissertation
REM Automated compilation with error handling and optimization

echo Starting PhD Dissertation Build...
echo =====================================

REM Set variables
set LATEX_ENGINE="C:\Users\Nouri\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
set BIBER_ENGINE="C:\Users\Nouri\AppData\Local\Programs\MiKTeX\miktex\bin\x64\biber.exe"
set DOCUMENT=main

echo Using LaTeX Engine: %LATEX_ENGINE%

REM Clean previous builds
echo Cleaning previous build files...
if exist %DOCUMENT%.pdf del %DOCUMENT%.pdf
if exist %DOCUMENT%.aux del %DOCUMENT%.aux
if exist %DOCUMENT%.log del %DOCUMENT%.log
if exist %DOCUMENT%.bbl del %DOCUMENT%.bbl
if exist %DOCUMENT%.blg del %DOCUMENT%.blg
if exist %DOCUMENT%.toc del %DOCUMENT%.toc
if exist %DOCUMENT%.lof del %DOCUMENT%.lof
if exist %DOCUMENT%.lot del %DOCUMENT%.lot
if exist %DOCUMENT%.idx del %DOCUMENT%.idx
if exist %DOCUMENT%.ind del %DOCUMENT%.ind
if exist %DOCUMENT%.out del %DOCUMENT%.out

echo.
echo First LaTeX Pass...
%LATEX_ENGINE% -interaction=nonstopmode %DOCUMENT%.tex

echo.
echo Running Biber for Bibliography...
if exist %DOCUMENT%.bcf (
    %BIBER_ENGINE% %DOCUMENT%
) else (
    echo No .bcf file found, skipping Biber
)

echo.
echo Second LaTeX Pass...
%LATEX_ENGINE% -interaction=nonstopmode %DOCUMENT%.tex

echo.
echo Third LaTeX Pass (for cross-references)...
%LATEX_ENGINE% -interaction=nonstopmode %DOCUMENT%.tex

echo.
echo Build Process Completed!

if exist %DOCUMENT%.pdf (
    echo SUCCESS: PDF generated successfully!
    echo File location: %CD%\%DOCUMENT%.pdf
    echo Opening PDF...
    start %DOCUMENT%.pdf
) else (
    echo ERROR: PDF was not generated. Check the log file for errors.
    echo Check %DOCUMENT%.log for detailed error information
)

pause