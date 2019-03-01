import os
import cv2
import time
import argparse
import numpy as np
import subprocess as sp
import json
import tensorflow as tf
import struct
import socket

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, IPVideoStream, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03'
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

def raise_alarm(IP, port):
    alarm_time = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())
    print('[INFO] Alarm raised at {}'.format(alarm_time))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((IP,port))
    #req = struct.pack() FALTA DEFINIR MJE PARA ACTIVAR LA ALARMA
    sock.send(req)
    sock.close()


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    time_i = time.time()
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
    # Distance calculation
    print('[INFO] Time used for object detection: {} s '.format(time.time()-time_i))
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
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = np.zeros((height,width,3), np.uint8)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))
    fps.stop()
    sess.close()

def add_warning(frame, height, width):
    red_warning = frame.copy()
    yellow_warning = frame.copy()
    cv2.rectangle(red_warning,(0,int(0.75*width)),(int(height),int(width)),(0,0,255),-1)
    cv2.rectangle(yellow_warning,(0,int(0.5*width-1)),(int(height),int(0.75*width-1)),(0,255,255),-1)
    cv2.addWeighted(red_warning, 0.5, frame, 0.5, 0, frame)
    cv2.addWeighted(yellow_warning, 0.5, frame, 0.5, 0, frame)

def alarm_condition(frame, point):
    y_threshold_warning = 0.5
    y_threshold_alarm = 0.75
    #cv2.line(frame, (0,y_threshold_warning*height), (width,y_threshold_warning*height), color = 'yellow')
    #cv2.line(frame, (0,y_threshold_alarm*height), (width,y_threshold_alarm*height), color = 'red')
    if point['ymax']>y_threshold_warning and point['ymax']<y_threshold_alarm:
        cv2.putText(frame, 'WARNING', (100,50),font, 1.5, (0,0,255), 2)
        return True
    elif point['ymax']>y_threshold_alarm:
        cv2.putText(frame, 'ALARM', (100,50),font, 1.5, (0,0,255), 2)
        ## raise_alarm(IP,port)
        return True
    else:
        return False

def display_rectangle(frame,point,height,width,text=False):
        time_i = time.time()
        mid_x = (point['xmax']+point['xmin'])/2
        mid_y = (point['ymax']+point['ymin'])/2
        width_aprox = round(point['xmax']-point['xmin'],1)
        height_aprox = round(point['ymax']-point['ymin'],1)
        cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                  (int(point['xmax'] * width), int(point['ymax'] * height)), color, 3)
        cv2.rectangle(frame, (int(point['xmin'] * width), int(point['ymin'] * height)),
                  (int(point['xmin'] * width) + len(name[0]) * 6,
                   int(point['ymin'] * height) - 10), color, -1, cv2.LINE_AA)
        print('[INFO] Time used for rectangle creation: {} s '.format(time.time()-time_i))
        if text:
            cv2.putText(frame, 'Height: {}'.format(height_aprox*height), (int(mid_x*width),
            int(mid_y*height+15)),font, 0.5, (255,255,255), 2)
            cv2.putText(frame, 'Width: {}'.format(width_aprox*width), (int(mid_x*width),
            int(mid_y*height-15)),font, 0.5, (255,255,255), 2)

if __name__ == '__main__':
    filtered_classes = ['person','car','bus','truck']
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
    height = 480
    width = 640
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    if (args.stream_in):
        print('Reading from IP')
        video_capture = IPVideoStream(src=args.stream_in).start()
    else:
        print('Reading from webcam.')
        video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()
    while t<11:
        frame = cv2.imdecode(video_capture.read(), 1)
        input_q.put(frame)
        font = cv2.FONT_HERSHEY_DUPLEX
        if output_q.empty():
            pass  # fill up queue
        else:
            t = time.time()
            data = output_q.get()
            rec_points = data['rect_points']
            class_names = data['class_names']
            class_colors = data['class_colors']
            for point, name, color in zip(rec_points, class_names, class_colors):
                if 'person' in name[0]:
                    print('Point:{:.2f},{:.2f}'.format(point['xmin'],point['xmax']))
                    display_rectangle(frame,point,height,width,text=False)
                    print('[INFO] time elapsed: {} s'.format(time.time()-t))
                    if alarm_condition(frame, point):
                        print('\a')
                        #os.system('say "warning"')
                else:
                    pass
            try:
                add_warning(frame,height,width)
            except:
                pass
            if args.stream_out:
                print('Streaming elsewhere!')
            else:
                cv2.imshow('Video', frame)
        fps.update()
            t+=1
    else:
        print('[INFO] Closing...')
        sys.exit(1)
        #print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    cv2.destroyAllWindows()