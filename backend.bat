@echo off
REM Activate backend venv and run backend
cd backend
call ..\.venv\Scripts\activate
python main.py
