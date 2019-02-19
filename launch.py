#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import datetime
from subprocess import Popen, PIPE
import time

height='600'
width='800'
size = str(width+'x'+height)
quality = "40"
fps = "25.0"

ip_dir=("http://10.23.170.23/control/faststream.jpg?stream=full&preview&previewsize="
+size+"&quality="+quality+"&fps="+fps)
print(ip_dir)
log_directory = 'logs/'
while True:
    try:
        subprocess.call(['python','object_detection_multithreading.py','-strin '+ip_dir])
        #os.system('python object_detection_multithreading.py -strin '+ip_dir)
    except Exception as e:
        f=open(log_directory+'error_logs.txt', "a+")
        f.write('{}. Failure to execute. Error:\t{}'.format(datetime.now.strftime("%d/%m/%Y, %H:%M:%S"),str(e)))
        pass
