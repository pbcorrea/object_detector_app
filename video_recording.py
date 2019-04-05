import argparse
import cv2
import json
import time
import os
import sys

import numpy as np
import subprocess as sp

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, IPVideoStream, WebcamVideoStream, draw_boxes_and_labels

from datetime import date

def record(output_name,src,length,test=False):

    if test == True:
        cap = cv2.VideoCapture(0)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')  Windows only
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #video_capture = IPVideoStream(src=stream_ip).start()
    out = cv2.VideoWriter(output_name+'.mov', fourcc, 20.0, size,True)
    start_time = time.time()
    time_elapsed = 0
    while time_elapsed < length:
        #frame = cv2.imdecode(video_capture.read(), 1)
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Recording',frame)
            out.write(frame)
        time_elapsed = time.time()-start_time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('Video written ... ')
    cap.release()
    out.release() 
    #video_capture.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    height = 720
    width = 1280
    size = str(width)+'x'+str(height)
    quality = "50"
    fps = "30.0"
    stream_ip=("http://10.23.217.103/control/faststream.jpg?stream=full&preview&previewsize="
    +size+"&quality="+quality+"&fps="+fps+"&camera=left")
    #video_capture = IPVideoStream(src=stream_ip).start()
    record('Video de prueba_1_hora',stream_ip,10000,test=True)
