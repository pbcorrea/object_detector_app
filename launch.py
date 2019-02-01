#!/anaconda3/bin/python
import os

size = "640x480"
camera = "both"
quality = "40"
fps = "15.0"

ip_dir=("http://10.23.41.128/control/faststream.jpg?stream=full&preview&previewsize="
+size+"&quality="+quality+"&fps="+fps+"&camera="+camera)

operative = False
while operative == False:
    try:
        os.system('python object_detection_multithreading_time_testing.py -strin '+ip_dir)
        operative = True
        break
    except:
        operative = False
