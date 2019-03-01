#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2
import json
import time
import os
import sys

import numpy as np
import subprocess as sp
import tensorflow as tf

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, IPVideoStream, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util
from pyModbusTCP.client import ModbusClient

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = {1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'bicycle'}, 3: {'id': 3, 'name': 'car'},
 4: {'id': 4, 'name': 'motorcycle'}, 5: {'id': 5, 'name': 'airplane'}, 6: {'id': 6, 'name': 'bus'}, 7: {'id': 7, 'name': 'train'},
 8: {'id': 8, 'name': 'truck'}, 9: {'id': 9, 'name': 'boat'}}

def raise_alarm(connection, alarm):
    alarm_time = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())
    if alarm:
        try:
            connection.write_single_coil(0,1)
        except:
            pass
    else:
        try:
            connection.write_single_coil(0,0)
        except:
            pass

def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Filter only the needed results
    filtered_classes = ['person']

    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)

def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            pass
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))
    fps.stop()
    sess.close()

def add_warning(frame, height, width):
    cv2.line(frame, (0,int(0.5*height)), (int(width),int(0.5*height)), (0,255,255))
    cv2.line(frame, (0,int(0.75*height)), (int(width),int(0.75*height)), (0,0,255))

def alarm_condition(frame, point, height, width):
    y_threshold_warning = 0.5
    y_threshold_alarm = 0.75
    if point['ymax']>y_threshold_warning and point['ymax']<y_threshold_alarm:
        cv2.putText(frame, 'WARNING', (100,50),font, 1.5, (0,0,255), 2)
        return True
    elif point['ymax']>y_threshold_alarm:
        cv2.putText(frame, 'ALARM', (100,50),font, 1.5, (0,0,255), 2)
        return True
    else:
        return False

def display_rectangle(frame,point,height,width,text=False):
        mid_x = (point['xmax']+point['xmin'])/2
        mid_y = (point['ymax']+point['ymin'])/2
        width_aprox = round(point['xmax']-point['xmin'],1)
        height_aprox = round(point['ymax']-point['ymin'],1)
        cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                  (int(point['xmax'] * width), int(point['ymax'] * height)), color, 3)
        cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                  (int(point['xmin'] * width) + len(name[0]) * 6,
                   int(point['ymin'] * height) - 10), color, -1, cv2.LINE_AA)
        if text:
            cv2.putText(frame, 'Height: {}'.format(height_aprox*height), (int(mid_x*width),
            int(mid_y*height+15)),font, 0.5, (255,255,255), 2)
            cv2.putText(frame, 'Width: {}'.format(width_aprox*width), (int(mid_x*width),
            int(mid_y*height-15)),font, 0.5, (255,255,255), 2)

if __name__ == '__main__':
    filtered_classes = ['person']
    parser = argparse.ArgumentParser()
    parser.add_argument('-strin', '--stream-input', dest="stream_in", action='store', type=str, default=None)
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=800, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=600, help='Height of the frames in the video stream.')
    parser.add_argument('-strout','--stream-output', dest="stream_out", help='The URL to send the livestreamed object detection to.')
    args = parser.parse_args()
    height = 600
    width = 800
    size = str(width)+'x'+str(height)
    quality = "20"
    fps = "15.0"
    stream_ip=("http://10.23.170.23/control/faststream.jpg?stream=full&preview&previewsize="
    +size+"&quality="+quality+"&fps="+fps+"&camera=left")
    modbus_ip = '192.168.127.254'
    modbus_port = '502'
    connection = ModbusClient(host=modbus_ip, port=modbus_port, auto_open=True)
    connection.debug(False)
    input_q = Queue(1)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()
    video_capture = IPVideoStream(src=stream_ip).start()
    cv2.useOptimized()
    fps = FPS().start()
    alarm = False
    while True:
        try:
            try:
                frame = cv2.imdecode(video_capture.read(), 1)
            except:
                frame =  np.zeros((height,width,3), np.uint8)
                pass
            input_q.put(frame)
            raise_alarm(connection,alarm)
            font = cv2.FONT_HERSHEY_DUPLEX
            if output_q.empty():
                alarm = False
                pass  # fill up queue
            else:
                alarm = False
                data = output_q.get()
                rec_points = data['rect_points']
                class_names = data['class_names']
                class_colors = data['class_colors']
                for point, name, color in zip(rec_points, class_names, class_colors):
                    if 'person' in name[0]:
                        display_rectangle(frame,point,height,width,text=False)
                        alarm = alarm_condition(frame, point, height, width)
                    else:
                        alarm = False
                        pass
                add_warning(frame,height,width)
                cv2.imshow('ODDL - Fatality Prevention', frame)
            print('[INFO] Closing application ...')
            sys.exit(1)
            fps.update()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.stop()
        except:
            print('[INFO] Fatal error. Closing application...')
            sys.exit()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    cv2.destroyAllWindows()
    connection.write_single_coil(0,0)
