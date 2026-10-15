@echo off
echo ========================================================
echo  Starting Plagiarism Detector Web Application...
echo ========================================================
echo.
call .venv\Scripts\activate.bat
streamlit run app.py
pause
