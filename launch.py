#!/usr/bin/env python
import os

height='480'
width='600'
size = str(width+'x'+height)
camera = "left"
quality = "40"
fps = "15.0"

ip_dir=("http://10.23.170.23/control/faststream.jpg?stream=full&preview&previewsize="
+size+"&quality="+quality+"&fps="+fps+"&camera="+camera)

operative = False
while operative == False:
    try:
        os.system('python object_detection_multithreading.py -strin '+ip_dir)
        operative = True
        break
    except:
        operative = False
