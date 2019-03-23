@echo off
title ODDL Launching...
set root_path=%C:\Windows\System32%
set work_path=%C:\Users\"RMC LABS"\oddl\object_detector_app%
cd %root_path%
call C:\Users\"RMC LABS"\Anaconda3\Scripts\activate.bat oddl
cd %work_path%
C:\Users\"RMC LABS"\Anaconda3\envs\oddl\python.exe launch.py
pause