#! /usr/bin/env python

import rospy
import cv2
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

#NN imports
import math
import re
import random
from sklearn.metrics import confusion_matrix

from collections import Counter
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join

#training CNN
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras import backend

import tensorflow as tf

import sys

sess1 = tf.Session()
graph1 = tf.get_default_graph()
set_session(sess1)

#load NN for stall
stall_NN = load_model("/home/fizzer/ros_ws/src/my_controller/src/NN_object_stall_good")
stall_recognized = False

# Stall NN helper

classes_stall = np.array([])
for i in range(1,9):
  classes_stall = np.append(classes_stall, i)

comparison_img = "/home/fizzer/ros_ws/src/my_controller/pictures/cropped_close_ups/black_far_P_GOOD.png" 
dist_scale_front = 0.8 # 0.7
positive_match = 4 # 4

# Cropping variables
X_crop_left = 0
X_crop_right = 640 #640
Y_crop_top = 400
Y_crop_bottom = 720

# centroid
X_centroid_list = []
Y_centroid_list = []
centroid_avg_error = 25
prev_x = 0
prev_y = 0
prev_match = False
prev_prev_match = False
found_match = False

#detecting if blue car seen
blue_car_detected = False
predict_array = np.array([0,0,0,0,0,0])

frame_stall_trace = None

sim_time_seconds = 0

def cropImage(width_start, width_end, height_start, height_end, frame):
	return frame[height_start:height_end,width_start:width_end,0:3]

def stall_num_detect(image_path, title, robot_frame, dist_scale):
	global X_centroid_list
	global Y_centroid_list
	global prev_x
	global prev_y
	global prev_match
	global prev_prev_match
	global found_match
	global frame_stall_trace

	found_match = False

	reserved_frame = np.copy(robot_frame)

	sift = cv2.xfeatures2d.SIFT_create()
	frame = cropImage(X_crop_left,X_crop_right,Y_crop_top,Y_crop_bottom,robot_frame)
	original_frame = frame

	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	# trying to detect black
	frame_threshold = cv2.inRange(hsv,(0,0,0),(180,255,30))
	frame = frame_threshold
	#cv2.imshow("HSV",frame)
	#cv2.waitKey(1)

	# roslaunch my_controller SIFT_stall_num.launch
	# ./run_sim.sh -vpg

	# Camera Robot Frame - get keypoints and descriptors
	kp_frame, desc_frame = sift.detectAndCompute(frame, None)
	
	# Our comparison image - get keypoints and descriptors
	img = cv2.imread(image_path, 0)
	kp_image, desc_image = sift.detectAndCompute(img, None)
	
	# Feature Matching
	index_params = dict(algorithm=0, trees=5)
	search_params = dict()
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(desc_image, desc_frame, k=2)

	good_points = []
	for m, n in matches:
		# to avoid many False results, take descriptors that have short distances between them
		# play with the dist_scale constant: 0.6, 0.7, 0.8 - potential values
		if m.distance < dist_scale * n.distance:
			good_points.append(m)

	img_matches = cv2.drawMatches(img, kp_image, frame, kp_frame, good_points, frame)
	#cv2.imshow("Matches", img_matches)
	#cv2.waitKey(1)

	match = False
	if len(good_points) > positive_match:
		match = True		

		query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
		train_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

		# matrix shows object from its perspective?
		matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
		matches_mask = mask.ravel().tolist()        # extract points from mask and put into a list

		# Perspective transforms, helps with homography
		h, w = img.shape        # height and width of original image

		# points gets h and w of image. does not work with int32, but float32 works
		pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
		dst = cv2.perspectiveTransform(pts, matrix)
		
		point_0 = dst[0]
		point_1 = dst[1]
		point_2 = dst[2]
		point_3 = dst[3]

		x = []
		y = []

		i = 0
		while i < 4:
			x.append(dst[i,0,0])
			y.append(dst[i,0,1])
			i = i + 1

		x_centroid = int(sum(x)/4.0)
		y_centroid = int(sum(y)/4.0)

		if len(X_centroid_list) == 0:
			# first list entry
			X_centroid_list.append(x_centroid)
			Y_centroid_list.append(y_centroid)
		else:
			if abs(prev_x-x_centroid) < centroid_avg_error and abs(prev_y-y_centroid) < centroid_avg_error:
				# detecting similar location
				x_centroid_avg = int(sum(X_centroid_list)/len(X_centroid_list))
				y_centroid_avg = int(sum(Y_centroid_list)/len(Y_centroid_list))
				if abs(x_centroid-x_centroid_avg) < centroid_avg_error and abs(y_centroid-y_centroid_avg) < centroid_avg_error:
					# around location of ongoing centroid point
					X_centroid_list.append(x_centroid)
					Y_centroid_list.append(y_centroid)
					#print(" ")
					#print("avgs: " + str(x_centroid_avg) + " " + str(y_centroid_avg))
				else:
					# new and last value are close but far from avg, therefore the robot probably moved a lot, recalculate centroid at new location
					new_list1 = []
					new_list2 = []
					X_centroid_list = new_list1
					Y_centroid_list = new_list2
					X_centroid_list.append(prev_x)
					X_centroid_list.append(x_centroid)
					Y_centroid_list.append(prev_y)
					Y_centroid_list.append(y_centroid)
			else:
				# nothing happens this run, we don't know if the robot moved or got a noisy result
				nothing = 0

		prev_x = x_centroid
		prev_y = y_centroid

		#print("vals: " + str(x_centroid) + " " + str(y_centroid))
		#print(" ")
		#print("number of good points: " + str(len(good_points)))
		
		# roslaunch my_controller SIFT_stall_num.launch

	else:
		match = False

	# we have found the centroid of P, will now draw a box to the right of the centroid to try to capture the stall number
	if len(X_centroid_list) != 0 and match and prev_match and prev_prev_match:
		found_match = True
		x_centroid_avg = int(sum(X_centroid_list)/len(X_centroid_list))
		y_centroid_avg = int(sum(Y_centroid_list)/len(Y_centroid_list))

		# values i used to capture data
		x_move = 35
		y_move = -45
		box_capture_height = 100
		box_capture_width = 100

		# draw rectangle function requires the top-left and bottom-right corner
		top_left_x = x_centroid_avg + X_crop_left + x_move
		top_left_y = y_centroid_avg + Y_crop_top + y_move
		bottom_right_x = top_left_x + box_capture_width
		bottom_right_y = top_left_y + box_capture_height

		stall_num_trace = cv2.rectangle(robot_frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,0,255),3)
		#cv2.imshow("trace",stall_num_trace)
		#cv2.waitKey(1)

		# Convert BGR to HSV
		hsv = cv2.cvtColor(reserved_frame,cv2.COLOR_BGR2HSV)

		# trying to detect black
		frame_stall_trace = cv2.inRange(hsv,(0,0,0),(180,255,30))

		#print(frame_threshold.shape)
		frame_stall_trace = frame_stall_trace[top_left_y:bottom_right_y,top_left_x:bottom_right_x]
		#cv2.imshow("real stall", frame_threshold)
		#cv2.waitKey(1)

		# roslaunch my_controller SIFT_stall_num.launch
	
	prev_prev_match = prev_match
	prev_match = match
	return found_match

def callback_blue_car(b_car_detected):
	global blue_car_detected
	#extract original string from data
	blue_car_detected = str(b_car_detected.data)
	#print("Blue car detected in SIFT_stall_num")

def callback_image(data):
	global blue_car_detected
	global frame_stall_trace
	if blue_car_detected == "blue car detected":
		cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
		stall_detected = stall_num_detect(comparison_img, "stall number", cv_image, dist_scale_front)
		#print("***************Ken's NN output: " + str(stall_detected))
		if stall_detected:
			global sess1
			global graph1
			global stall_recognized
			# directly run the NN on the image: frame_stall_trace
			hsv_crop_stall = frame_stall_trace.reshape(1,frame_stall_trace.shape[0],frame_stall_trace.shape[1],1)
			img = cv2.cvtColor(hsv_crop_stall[0], cv2.COLOR_GRAY2RGB)
			img_aug = np.expand_dims(img, axis=0)
			#Thanks to Grace for finding this solution: https://github.com/tensorflow/tensorflow/issues/28287
			with graph1.as_default():
				set_session(sess1)
				y_predict = stall_NN.predict(img_aug)[0]
			predicted_character = map_stall_prediction_to_char(y_predict)
			#print("predicted stall: " + str(predicted_character))

			if str(predicted_character) == "7.0" or str(predicted_character) == "8.0":
				# we got a bad reading, 
				blue_car_detected = "blue car detected"
			else:
				predicted_character = int(predicted_character)
				predict_array[predicted_character-1] = predict_array[predicted_character-1] + 1
				# max returns the max value of the array
				if np.max(predict_array) > 4:
					# publish the predicted character
					max_char = np.max(predict_array)
					i = 0
					for term in predict_array:
						if term == max_char:
							large_index = i
						i = i + 1
					character = classes_stall[large_index]

					stall_img_pub.publish(str(character))
					#time.sleep(0.05)
					blue_car_detected = False

					for j in range(0,6):
						predict_array[j] = 0
				else:
					blue_car_detected = "blue car detected"
			
			#stall_num = bridge.cv2_to_imgmsg(frame_stall_trace, encoding="passthrough")
			# publishes a cropped, hsv image 
			#stall_img_pub.publish(stall_num)
			#print("$$$$$$$$$$$$$$$$Ken's NN output: " + str(predicted_character))

'''
# ./run_sim.sh -vpg
# roslaunch my_controller run_comp.launch
# ./score_tracker.py
'''
def map_stall_prediction_to_char(y_predict):
	y_predicted_max = np.max(y_predict)
	index_predicted = np.where(y_predict == y_predicted_max)
	character = classes_stall[index_predicted]
	return character[0]


def callback_time(data):
	global sim_time_seconds
	# gets current time in seconds
	sim_time_seconds = int(str(data)[16:21])

# ROS setup stuff below
rospy.init_node('detect_car_node')
bridge = CvBridge()
velocity_pub = rospy.Publisher('/R1/cmd_vel',Twist,queue_size = 1)
#stall_img_pub = rospy.Publisher('/sim_stall',Image,queue_size=1)
stall_img_pub = rospy.Publisher('/sim_stall',String,queue_size=1)
move = Twist()
image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callback_image) 
time_sub = rospy.Subscriber('/clock',String,callback_time)
blue_car_sub = rospy.Subscriber('/blue_car_detection',String,callback_blue_car)
rospy.sleep(1) # wait 1 second to let things start up
rospy.spin()