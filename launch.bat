@echo off
title ODDL Launching...
set root_path=%C:\Windows\System32%
set work_path=%C:\Users\RMCLABS\oddl\object_detector_app%
cd %root_path%
call C:\Users\RMCLABS\Anaconda3\Scripts\activate.bat oddl
cd %work_path%
C:\Users\RMCLABS\Anaconda3\envs\oddl\python.exe launch.py
