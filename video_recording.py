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


print("Today's date:", today)

if __name__ == '__main__':
    fps = FPS().start()
    height = 720
    width = 1280
    size = str(width)+'x'+str(height)
    quality = "50"
    fps = "30.0"
    stream_ip=("http://10.23.217.103/control/faststream.jpg?stream=full&preview&previewsize="
    +size+"&quality="+quality+"&fps="+fps+"&camera=left")
    video_capture = IPVideoStream(src=stream_ip).start()
    fps = FPS().start()
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (height, width))
    date = date.today() 
    today = date.strftime("%d")
    while True:
        if today == '25':
            try:

        frame = cv2.imdecode(video_capture.read(), 1)
        cv2.imshow('Video', frame)
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
           
    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    out.release() 
    video_capture.stop()
    cv2.destroyAllWindows()
