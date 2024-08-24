@echo off
cd /d "D:\conference work\pythonProject\venv\Scripts"
call activate
cd /d "D:\conference work\pythonProject\dataset_generation"
python rotate_dataset.py
pause