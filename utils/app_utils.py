#!/usr/bin/env python
# From http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

import struct
import six
import collections
import cv2
import datetime
import subprocess as sp
import json
import numpy
import time
import os
import requests
import socket
import sys
from threading import Thread, Event, ThreadError
from matplotlib import colors


class Alarm:
	def __init__(self):
		self.alarm_ip = 'http://10.10.10.10/control/rcontrol?action=sound&soundfile=Alarm'
		self.modbus_ip = '192.168.127.254'
		self.modbus_port = '502'
		self.connection = ModbusClient(host=modbus_ip, port=modbus_port, auto_open=True)
		self.connection.debug(False)
	
	def condition(self,frame, point, height, width): # CAMBIAR ACÁ LOS VALORES PARA LAS LÍNEAS DE ALARMA
		self.threshold_warning = 0.25
		self.threshold_alarm = 0.55
		if point['ymax']>y_threshold_warning and point['ymax']<y_threshold_alarm:
			self.text = 'PRECAUCION'
			self.alarm = [True, False]
		elif point['ymax']>y_threshold_alarm:
			self.text = 'ALARMA'
			self.alarm = [True, True]
		else:
			self.text = ''
			self.alarm = [False, False]
		return self.alarm, self.text

	def activate(self):
		if self.alarm[0] == True and self.alarm[1] == False:
			print('Iniciando alarma 1:\t{}'.format(time.time()))
			alarm_lock.acquire()
			requests.get(self.alarm_ip) #CONEXION ALARMA CAMARA
        	#connection.write_single_coil(1,1) #CONEXION LUZ INTERNA MODBUS
			time.sleep(2)
			alarm_lock.release()
			print('Terminando alarma 1:\t{}'.format(time.time()))
		elif self.alarm[0] == True and self.alarm[1] == True:
			print('Iniciando alarma 2:\t{}'.format(time.time()))
			alarm_lock.acquire()
			requests.get(self.alarm_ip) #CONEXION ALARMA CAMARA
        	#connection.write_single_coil(1,1) #CONEXION LUZ INTERNA MODBUS
        	#connection.write_single_coil(2,1) #CONEXION CORTA-CORRIENTE MODBUS
			time.sleep(2)
			alarm_lock.release()
			print('Terminando alarma 2:\t{}'.format(time.time()))  
		else:
			pass


class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self): 
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()

class IPVideoStream:
	def __init__(self, src):
		# initialize the video camera stream and read the first frame
		# from the stream
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False 
		# try connection and send data until succesful
		self.connected = False
		while not self.connected:
			try:
				self.stream = requests.get(src, stream=True, timeout=10000)
				if self.stream.status_code == 200:
					self.connected = True
					print('[INFO] Connection succesful.')
					bytes_ = bytes()
					for chunk in self.stream.iter_content(chunk_size=1024):
						bytes_+=chunk
						a = bytes_.find(b'\xff\xd8')
						b = bytes_.find(b'\xff\xd9')
						if a!=-1 and b!=-1:
							jpg = bytes_[a:b+2]
							bytes_ = bytes_[b+2:]
							self.frame = numpy.fromstring(jpg, dtype=numpy.uint8)
							print(self.frame.shape)
							self.grabbed = self.frame is not None
							break
			except requests.exceptions.ConnectionError:
				print('[INFO] Connection error. Retrying in 10 seconds...')
				self.connected = False
				time.sleep(10)
				pass

	def check_connection(url):
	    try:
	        _ = requests.get(url, timeout=60)
	        return True
	    except requests.ConnectionError:
	        print("Connection timed out")
	    return False

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		# if the thread indicator variable is set, stop the thread
		bytes_ = bytes()
		try:
			while True:# Camera connected(?)
				if self.stopped:
					return
				for chunk in self.stream.iter_content(chunk_size=1024):
					bytes_+=chunk
					a = bytes_.find(b'\xff\xd8')
					b = bytes_.find(b'\xff\xd9')
					if a!=-1 and b!=-1:
						frame_bytes = bytes_[a:b+2]
						bytes_ = bytes_[b+2:]
						jpg = numpy.fromstring(frame_bytes, dtype=numpy.uint8)
						if jpg.size:
							self.frame = jpg
							self.grabbed = self.frame is not None
							break
						else:
							self.frame = numpy.zeros(shape=(1280,720,3))
							print('Empty frame',frame_bytes)
							pass
						
		except ThreadError:
			print('ThreadError')
			self.stopped = True

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


def standard_colors():
	colors = [
		'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
		'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
		'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
		'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
		'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
		'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
		'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
		'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
		'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
		'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
		'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
		'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
		'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
		'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
		'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
		'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
		'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
		'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
		'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
		'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
		'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
		'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
		'WhiteSmoke', 'Yellow', 'YellowGreen'
	]
	return colors


def color_name_to_rgb():
	colors_rgb = []
	for key, value in colors.cnames.items():
		colors_rgb.append((key, struct.unpack('BBB', bytes.fromhex(value.replace('#', '')))))
	return dict(colors_rgb)


def draw_boxes_and_labels(
		boxes,
		classes,
		scores,
		category_index,
		instance_masks=None,
		keypoints=None,
		max_boxes_to_draw=20,
		min_score_thresh=.5,
		agnostic_mode=False):
	"""Returns boxes coordinates, class names and colors

	Args:
		boxes: a numpy array of shape [N, 4]
		classes: a numpy array of shape [N]
		scores: a numpy array of shape [N] or None.  If scores=None, then
		this function assumes that the boxes to be plotted are groundtruth
		boxes and plot all boxes as black with no classes or scores.
		category_index: a dict containing category dictionaries (each holding
		category index `id` and category name `name`) keyed by category indices.
		instance_masks: a numpy array of shape [N, image_height, image_width], can
		be None
		keypoints: a numpy array of shape [N, num_keypoints, 2], can
		be None
		max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
		all boxes.
		min_score_thresh: minimum score threshold for a box to be visualized
		agnostic_mode: boolean (default: False) controlling whether to evaluate in
		class-agnostic mode or not.  This mode will display scores but ignore
		classes.
	"""
	# Create a display string (and color) for every box location, group any boxes
	# that correspond to the same location.
	box_to_display_str_map = collections.defaultdict(list)
	box_to_color_map = collections.defaultdict(str)
	box_to_instance_masks_map = {}
	box_to_keypoints_map = collections.defaultdict(list)
	if not max_boxes_to_draw:
		max_boxes_to_draw = boxes.shape[0]
	for i in range(min(max_boxes_to_draw, boxes.shape[0])):
		if scores is None or scores[i] > min_score_thresh:
			box = tuple(boxes[i].tolist())
			if instance_masks is not None:
				box_to_instance_masks_map[box] = instance_masks[i]
			if keypoints is not None:
				box_to_keypoints_map[box].extend(keypoints[i])
			if scores is None:
				box_to_color_map[box] = 'black'
			else:
				if not agnostic_mode:
					if classes[i] in category_index.keys():
						class_name = category_index[classes[i]]['name']
					else:
						class_name = 'N/A'
					display_str = '{}: {}%'.format(
						class_name,
						int(100 * scores[i]))
				else:
					display_str = 'score: {}%'.format(int(100 * scores[i]))
				box_to_display_str_map[box].append(display_str)
				if agnostic_mode:
					box_to_color_map[box] = 'DarkOrange'
				else:
					box_to_color_map[box] = standard_colors()[
						classes[i] % len(standard_colors())]

	# Store all the coordinates of the boxes, class names and colors
	color_rgb = color_name_to_rgb()
	rect_points = []
	class_names = []
	class_colors = []
	for box, color in six.iteritems(box_to_color_map):
		ymin, xmin, ymax, xmax = box
		rect_points.append(dict(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax))
		class_names.append(box_to_display_str_map[box])
		class_colors.append(color_rgb[color.lower()])
	return rect_points, class_names, class_colors
