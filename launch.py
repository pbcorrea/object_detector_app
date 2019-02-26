#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import datetime
from subprocess import Popen, PIPE
import time

cwd = os.getcwd()
log_directory = os.path.join(cwd,'logs')
print(log_directory)
print(log_directory+'error_logs.txt')
while True:
    try:
        subprocess.call(['python','oddl.py'])
        #os.system('python object_detection_multithreading.py -strin '+ip_dir)
    except Exception as e:
        f=open(log_directory+'error_logs.txt', "a+")
        f.write('{}. Failure to execute. Error:\t{}'.format(datetime.now.strftime("%d/%m/%Y, %H:%M:%S"),str(e)))
        pass
