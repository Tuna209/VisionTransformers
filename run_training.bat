@echo off
REM Run the simplified DETR model on AU-AIR dataset
cd %~dp0..
call .venvonly\Scripts\activate
python codebase\detr_wrapper.py --pretrained --epochs 10
pause
