@echo off
python setup.py clean --all
python setup.py build_ext --inplace
pip install -e .
pause
