@echo off
title ODDL Launching...
cd oddl/object_detector_app
conda activate oddl
echo Virtualenv activated
echo Starting service...
python launch.py
