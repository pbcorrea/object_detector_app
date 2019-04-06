#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2
import json
import time
import os
import sys
import requests

import numpy as np
import subprocess as sp
import tensorflow as tf

from datetime import datetime
from queue import Queue, LifoQueue
from threading import Thread, Lock
from utils.app_utils import FPS, IPVideoStream, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util
from pyModbusTCP.client import ModbusClient

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017' 
#MODEL_NAME = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.

PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def raise_alarm(connection, alarm):
    alarm_request_ip = 'http://10.23.183.143/control/rcontrol?action=sound&soundfile=Alarm'
    alarm_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if alarm[0] == True and alarm[1] == False:
        print('[I] PRECAUCION:\t{}'.format(alarm_time))
        alarm_lock.acquire()
        requests.get(alarm_request_ip) #CONEXION ALARMA CAMARA
        connection.write_single_coil(1,1) #CONEXION LUZ INTERNA MODBUS
        time.sleep(2.7)
        connection.write_single_coil(1,0) #CONEXION LUZ INTERNA MODBUS
        alarm_lock.release()
        print('[F] PRECAUCION:\t{}'.format(alarm_time))
    elif alarm[0] == True and alarm[1] == True:
        print('[I] ALARMA:\t{}'.format(alarm_time))
        alarm_lock.acquire()
        requests.get(alarm_request_ip) #CONEXION ALARMA CAMARA
        connection.write_single_coil(1,1) #ABRIR CONEXION LUZ INTERNA MODBUS
        connection.write_single_coil(2,1) #ABRIR CONEXION CORTA-CORRIENTE MODBUS
        time.sleep(7)
        connection.write_single_coil(1,0) #CERRAR CONEXION LUZ INTERNA MODBUS
        connection.write_single_coil(2,0) #CERRAR CONEXION CORTA-CORRIENTE MODBUS
        alarm_lock.release()
        print('[F] ALARMA:\t{}'.format(alarm_time))   
    else:
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
            output_q.put(detect_objects(frame_rgb, sess, detection_graph))
        except:
            pass
    fps.stop()
    sess.close()


def add_warning(frame, height, width, text): # CAMBIAR ACÁ LOS VALORES PARA LAS LÍNEAS DE ALARMA
    yellow_line = 0.25
    red_line = 0.55
    cv2.line(frame, (0,int(yellow_line*height)), (int(width),int(yellow_line*height)), (0,255,255))
    cv2.line(frame, (0,int(red_line*height)), (int(width),int(red_line*height)), (0,0,255))
    cv2.putText(frame, text, (50,100),font, 2, (0,0,255), 2)

def alarm_condition(frame, point, height, width): # CAMBIAR ACÁ LOS VALORES PARA LAS LÍNEAS DE ALARMA
    y_threshold_warning = 0.25
    y_threshold_alarm = 0.55
    if point['ymax']>y_threshold_warning and point['ymax']<y_threshold_alarm:
        text = 'PRECAUCION'
        alarm = [True, False]
    elif point['ymax']>y_threshold_alarm:
        text = 'ALARMA'
        alarm = [True, True]
    else:
        text = ''
        alarm = [False, False]
    return alarm, text


def display_rectangle(frame,point,height,width,name,text=False):
        mid_x = (point['xmax']+point['xmin'])/2
        mid_y = (point['ymax']+point['ymin'])/2
        width_aprox = round(point['xmax']-point['xmin'],1)
        height_aprox = round(point['ymax']-point['ymin'],1)
        cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                  (int(point['xmax'] * width), int(point['ymax'] * height)), color, 3)
        cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                  (int(point['xmin'] * width) + len(name[0]) * 6,
                   int(point['ymin'] * height) - 10), color, -1, cv2.LINE_AA)
        cv2.putText(frame, name, (int(point['xmin'] * width), int(point['ymin'] * height)), font,
                            0.3, (0, 0, 0), 1)
        if text:
            cv2.putText(frame, 'Height: {}'.format(height_aprox*height), (int(mid_x*width),
            int(mid_y*height+15)),font, 0.5, (255,255,255), 2)
            cv2.putText(frame, 'Width: {}'.format(width_aprox*width), (int(mid_x*width),
            int(mid_y*height-15)),font, 0.5, (255,255,255), 2)

if __name__ == '__main__':
    filtered_classes = ['person','car',]
    height = 720
    width = 1280
    size = str(width)+'x'+str(height)
    quality = "50"
    fps = "30.0"
    stream_ip=("http://10.23.183.143/control/faststream.jpg?stream=full&preview&previewsize="
    +size+"&quality="+quality+"&fps="+fps+"&camera=left")
    modbus_ip = '192.168.127.254'
    modbus_port = '502'
    connection = ModbusClient(host=modbus_ip, port=modbus_port, auto_open=True)
    connection.debug(False)
    alarm_lock = Lock()
    input_q = Queue(1)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    t = Thread(target=worker, args=(input_q, output_q))
    t.daemon = True
    t.start()
    text = ''
    alarm = [False, False]
    frame = np.zeros((height,width,3))
    video_capture = IPVideoStream(src=stream_ip).start()
    fps = FPS().start() 
    thread_list = [] 
    while True:
        if video_capture.read().size:
            frame = cv2.imdecode(video_capture.read(), 1)
        else:
            frame = cv2.imdecode(np.zeros((height,width,3)), 1)
        input_q.put(frame)
        font = cv2.FONT_HERSHEY_DUPLEX
        if output_q.empty():
            alarm = [False, False]
            text = ''
            pass  # fill up queue
        else:
            text = ''
            data = output_q.get()
            rec_points = data['rect_points']
            class_names = data['class_names']
            class_colors = data['class_colors']
            for point, name, color in zip(rec_points, class_names, class_colors):
                if 'person' in name[0]:
                    display_rectangle(frame,point,height,width,name[0],text=False)
                    alarm, text = alarm_condition(frame, point, height, width)
                    if not alarm_lock.locked():
                        alarm_thread = Thread(target=raise_alarm,args=(connection,alarm))
                        alarm_thread.daemon = True
                        alarm_thread.start()
                        thread_list.append(alarm_thread)
                #elif 'car' in name[0]:
                #   print(name[0])
                  # display_rectangle(frame,point,height,width,text=False)
                #elif 'truck' in name[0]:
                 #   print(name[0])
                #    display_rectangle(frame,point,height,width,text=False)
                #elif 'bus' in name[0]:
                 #   print(name[0])
                  #  display_rectangle(frame,point,height,width,text=False)
                else:
                    alarm = [False, False]
                    text = ''
                    pass
            add_warning(frame,height,width,text)
            cv2.imshow('ODDL - Fatality Prevention', frame)
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
        fps.stop()

print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

video_capture.stop()
cv2.destroyAllWindows()
#connection.write_single_coil(2,0)