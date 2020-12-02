#! /usr/bin/env python

import rospy
import cv2
import time
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

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

from collections import Counter 

#PID VARIABLES:

middle_screen_margin = 10
nominal_speed = 0.1
#nominal_speed = 0.05
max_turn_speed = 1

img_width = 0
previous_error = 0

d_history = [0,0,0,0,0]

control_robot = False
prev_control = False # added by Ken

#DETECTION INTERUPT VARIABLES:

blue_car_detected = False
blue_running_time = 0.0
first_pass = True

time_since_red_line = 0.0
red_line_detected = False
red_line_count = 0
crossing_crosswalk = False

#license_plate_NN._make_predict_function()
sess1 = tf.Session()
graph1 = tf.get_default_graph()
#graph1 = tf.Graph()
set_session(sess1)

#load NN for license_plates
#license_plate_NN = load_model("/home/fizzer/ros_ws/src/my_controller/src/NN_object_license_plate_opt2")
license_plate_NN = load_model("/home/fizzer/ros_ws/src/my_controller/src/NN_object_opt2")
license_plate_recognized = False
stall_recognized = False
plate_not_published = True
license_plate_location = ""
plate_number = ""

num_samples = 0
sample_set = []

stall_number = ""

stationary_count = 0
count = 2
num_times_inched_forward = 0

def get_center(img):
	y_val = 700
	height, width, shape = img.shape #720, 1280, 3
	global img_width
	img_width = width
	sub_img = img[y_val][:][:]

	grey_Lth = 75
	grey_Uth = 95

	started = False
	check_range = 5
	stop_index = None
	start_index = None
	midpoint_road = None

	noise_list = []

	for i in range(width):
		b = sub_img[i][0]
		g = sub_img[i][1]
		r = sub_img[i][2]

		if (b > grey_Lth and b < grey_Uth and 
		g > grey_Lth and g < grey_Uth and 
		r > grey_Lth and r < grey_Uth):

			if not started:
				b_sum = 0
				g_sum = 0
				r_sum = 0
				if i < (width - check_range):
					for j in range(check_range):
						b_sum+=sub_img[i+j][0]
						g_sum+=sub_img[i+j][1]
						r_sum+=sub_img[i+j][2]
					b_avg = b_sum / check_range
					g_avg = g_sum / check_range
					r_avg = r_sum / check_range

					if (b_avg > grey_Lth and b_avg < grey_Uth and 
					g_avg > grey_Lth and g_avg < grey_Uth and
					r_avg > grey_Lth and r_avg < grey_Uth):
						start_index = i
						started = True
				else:
					for j in range(width - i):
						b_sum+=sub_img[i+j][0]
						g_sum+=sub_img[i+j][1]
						r_sum+=sub_img[i+j][2]
					b_avg = b_sum / check_range
					g_avg = g_sum / check_range
					r_avg = r_sum / check_range

					if (b_avg > grey_Lth and b_avg < grey_Uth and 
					g_avg > grey_Lth and g_avg < grey_Uth and
					r_avg > grey_Lth and r_avg < grey_Uth):
						start_index = i
						started = True
		else:
			if started:
				b_sum = 0
				g_sum = 0
				r_sum = 0
				if i < (width - check_range):
					for j in range(check_range):
						b_sum+=sub_img[i+j][0]
						g_sum+=sub_img[i+j][1]
						r_sum+=sub_img[i+j][2]
					b_avg = b_sum / check_range
					g_avg = g_sum / check_range
					r_avg = r_sum / check_range

					if not(b_avg > grey_Lth and b_avg < grey_Uth and 
					g_avg > grey_Lth and g_avg < grey_Uth and
					r_avg > grey_Lth and r_avg < grey_Uth):
						stop_index = i
						started = False
						break
				else:
					for j in range(width - i):
						b_sum+=sub_img[i+j][0]
						g_sum+=sub_img[i+j][1]
						r_sum+=sub_img[i+j][2]
					b_avg = b_sum / check_range
					g_avg = g_sum / check_range
					r_avg = r_sum / check_range

					if not(b_avg > grey_Lth and b_avg < grey_Uth and 
					g_avg > grey_Lth and g_avg < grey_Uth and
					r_avg > grey_Lth and r_avg < grey_Uth):
						stop_index = i
						started = False
						break

	if started:
		# if didnt stop
		stop_index = width-1

	if stop_index is None or start_index is None:
		road_detected = False
		midpoint_road = None
	else:
		midpoint_road = int((stop_index + start_index) / 2)
		road_detected = True

	return midpoint_road, road_detected

def follow_line(midpoint_road,road_detected):

	global middle_screen_margin
	global nominal_speed
	global max_turn_speed

	global img_width
	global previous_error

	global d_history

	p_gain = 1
	d_gain = 0.1
	#i_gain = 0

	'''
	if road_detected:
		print(midpoint_road)
	else:
		print("road not detected")

	'''

	if road_detected:

		error = (img_width / 2 - midpoint_road)/10
		#print("error: ", error)

		if abs(error) <= middle_screen_margin:
			move.linear.x = nominal_speed
			move.angular.z = 0
			driving_straight = True
		else:
			driving_straight = False
			calibrated_error = (-1)*error*(-1)
			calibrated_prev_error = (-1)*previous_error
			#do this since angular.z turns right for negative vals

			#move.angular.z > 0 turns left

			#compute d
			d = calibrated_error - calibrated_prev_error
			#add new d to history
			index = -1
			d_sum = 0
			while (index > -1*(len(d_history))):
				temp = d_history[index-1]
				d_history[index] = temp
				index-=1
				d_sum += temp

			d_history[0] = d
			d_sum += d
			d_avg = d_sum / len(d_history)

			#PID for speed
			speed = p_gain*calibrated_error + d_gain*d_avg
			if abs(speed) > max_turn_speed:
				if speed > 0:
					speed = max_turn_speed
				else:
					speed = max_turn_speed*(-1)
			move.angular.z = speed
			move.linear.x = nominal_speed

		previous_error = error

	else:
		if previous_error > 0:
			move.angular.z = (-1)*max_turn_speed
			previous_error = previous_error
			#turn right
		elif previous_error < 0:
			#turn left
			move.angular.z = max_turn_speed
			previous_error = previous_error
		else:
			#randomly road disappears
			move.angular.z = max_turn_speed
			previous_error = previous_error

		previous_error = img_width / 2 #give some max value when randomly turning fast
		driving_straight = False

	return driving_straight

def detect_red_line(original_image):

	Y_LEVEL = 700

	lower_red = np.array([0,50,50])
	upper_red = np.array([10,255,255])

	hsv_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
	height, width, shape = hsv_img.shape
	sub_img = hsv_img[Y_LEVEL][:][:]


	h = sub_img[int(width/2)][0]
	s = sub_img[int(width/2)][1]
	v = sub_img[int(width/2)][2]

	if (h >= lower_red[0] and h <= upper_red[0] and
		s >= lower_red[1] and v >= lower_red[2]):
		red_line_detected = True
	else:
		red_line_detected = False
	return red_line_detected


def detect_blue_car(original_image,blue_detected_previous):
	#sub_img = hsv_img[400][:][:]
	#height, width, shape = img.shape #720, 1280, 3

	#blue_car_b = 121
	#blue_car_g = 19
	#blue_car_r = 19

	global time_detected_blue_car
	global blue_running_time
	global first_pass
	global license_plate_recognized
	global stall_recognized
	global plate_not_published
	global num_times_inched_forward
	global recently_crossed

	TIME_LIMIT = 1.5

	lower_blue = np.array([110,50,50])
	upper_blue = np.array([130,255,255])

	hsv_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

	# Threshold the HSV image to get only blue colors
	#mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
	# Bitwise-AND mask and original image
	#only_blue_left_img = cv2.bitwise_and(original_image,original_image, mask= mask)

	if blue_detected_previous:
		STARTING_Y_PIXEL = 500
		STARTING_X_PIXEL = 0
		#reset the crosswalk code when blue car is detected
		recently_crossed = False

	else:

		STARTING_Y_PIXEL = 500
		STARTING_X_PIXEL = 150
		license_plate_recognized = False
		stall_recognized = False
		plate_not_published = True
		num_times_inched_forward = 0

	sub_img = hsv_img[STARTING_Y_PIXEL][:][:]
	height, width, shape = hsv_img.shape #720, 1280, 3

	blue_car_count = 0
	blue_car_detected = False

	for i in range(STARTING_X_PIXEL,int(width/2)):
		h = sub_img[i][0]
		s = sub_img[i][1]
		v = sub_img[i][2]
		'''
		if (b > blue_car_b - 10 and b < blue_car_b + 10 and
			g > blue_car_g - 10 and g < blue_car_g + 10 and
			r > blue_car_r - 10 and r < blue_car_r + 10):
		'''
		if (h >= lower_blue[0] and h <= upper_blue[0] and
			s >= lower_blue[1] and v >= lower_blue[2]):

			blue_car_count+=1
			if blue_car_count == 2:
				blue_car_detected = True

	if blue_car_detected == False and blue_detected_previous == True:
		#extends the robot driving straight just a little farther.
		if first_pass:
			time_detected_blue_car = rospy.get_time()
			first_pass = False

		time_difference = rospy.get_time() - time_detected_blue_car
		blue_running_time += time_difference
		time_detected_blue_car = rospy.get_time()

		if blue_running_time > TIME_LIMIT:
			blue_car_detected = False #should cause next iteration to skip this loop since blue_detected_previous will be False
			blue_running_time = 0.0
			first_pass = True
		else:
			blue_car_detected = True
			#print(blue_running_time)

	'''if blue_car_detected == True and blue_detected_previous == False:
		start_time = rospy.get_time()
		while rospy.get_time() - start_time < 0.08:
			move.angular.z = 0
			move.linear.x = nominal_speed'''
	#may need to tweak this


	return blue_car_detected

#**************************************************************
# Stall NN helper

classes_stall = np.array([])
for i in range(1,9):
  classes_stall = np.append(classes_stall, i)

#License Plate NN functions

# Generate classes
A_asci = 65
Z_asci = 90
classes = np.array([])
for i in range(Z_asci-A_asci+1):
  character = chr(A_asci+i)
  classes = np.append(classes, character)
for i in range(10):
  classes = np.append(classes, i)

#OVERALL PLATE DIMENSIONS CONSTANTS
RESIZE_WIDTH = 320 #must be multiple of 4
RESIZE_HEIGHT = 120

resize_width = RESIZE_WIDTH
resize_height = RESIZE_HEIGHT
split = RESIZE_WIDTH/4

INITIAL_RESIZE_WIDTH = 75
INITIAL_RESIZE_HEIGHT = 25

def split_images(imgset0,training_flag):

  #final overall plate dimensions
  resize_width = RESIZE_WIDTH
  resize_height = RESIZE_HEIGHT

  split = resize_width / 4
  #plate = imgset0[0]

  #put all the letters in one big array
  #put that plate array into a bigger array
  first_plate = True
  for plate in imgset0:
	#Resize images
	#Found this function from https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
	resized_plate = cv2.resize(plate, (resize_width, resize_height))
	resized_plate = cv2.cvtColor(resized_plate,cv2.COLOR_BGR2RGB) #convert image colour back to what it usually is.
	LL = resized_plate[:, 0:int(split)]
	LC = resized_plate[:, int(split):int(split*2)]
	RC = resized_plate[:, int(split*2):int(split*3)]
	RR = resized_plate[:, int(split*3):int(split*4)]
	if first_plate:
	  X_dataset = np.stack((LL,LC,RC,RR))
	  first_plate = False
	else:
	  X_dataset = np.vstack((X_dataset,LL.reshape(1,int(resize_height),int(split),3),
								  LC.reshape(1,int(resize_height),int(split),3),
								  RC.reshape(1,int(resize_height),int(split),3),
								  RR.reshape(1,int(resize_height),int(split),3)))
  return X_dataset

def mapPredictionToCharacter(y_predict):
	#maps NN predictions to the numbers based on the max probability.
	y_predicted_max = np.max(y_predict)
	index_predicted = np.where(y_predict == y_predicted_max)
	character = classes[index_predicted]
	return character[0]

#**************************************************************

#**************************************************************


def callback_control(cmd):
	#control function that will determine when to get the robot to use PID control
	global control_robot
	global comp_start_time

	#extract original string from data
	cmd = str(cmd.data)
	#print(cmd)
	
	#flag for telling robot when to start and stop being controlled
	if cmd == "start":
		control_robot = True
		comp_start_time = rospy.get_time()
		#print(control_robot)
	elif cmd == "stop":
		control_robot = False
		#print(control_robot)
	else:
		print("skipped")

def callback_image(img):

	global control_robot
	global comp_start_time
	global blue_car_detected
	global driving_straight
	global license_plate_recognized
	global stall_recognized
	global plate_not_published
	global license_plate_location
	global count
	global stationary_count
	global num_times_inched_forward
	global predicted_plate_number
	global stall_number
	global time_since_red_line
	global red_line_detected
	global red_line_count
	global crossing_crosswalk
	global sample_set
	#print(control_robot)

	if control_robot:

		if rospy.get_time() - comp_start_time < 1:
			#**************************************************
			#NOTE: THESE VALUES ARE FOR A nominal_speed of 0.06
			#**************************************************
			move.angular.z = 0.0
			move.linear.x = 0.3
			time_since_red_line = rospy.get_time()
		elif rospy.get_time() - comp_start_time < 2.0:


			move.angular.z = 1
			move.linear.x = 0
			time_since_red_line = rospy.get_time()
			#gets robot to turn left once competition starts

		#this should only execute once competition starts

		else:
			#first off, blue_car_detected = False
			#it wil just do PID using follow_line
			#only when during PID, the car is driving straight, the blue car is scanned for
			#once blue car is detected, then it will stop
			#it will wait until a license plate has been decoded.
			#once the license plate and stall is decoded, it publishes it to the score tracker
			#then it sets a flag so that the score tracker doesnt run again while the blue car is still being detected
			#

			cv_image = bridge.imgmsg_to_cv2(img, "bgr8")#image robot sees

			if rospy.get_time() - time_since_red_line > 0.1:
				red_line_detected = detect_red_line(cv_image)
				#print("made it in here")
				#print(red_line_detected)
				#if red_line_count == 5:
				if red_line_detected and not crossing_crosswalk:
					#print("*****************************************")
					red_line_pub.publish("True")
					#red_line_count+=1
				else:
					red_line_detected = False
				#red_line_count+=1
				time_since_red_line = rospy.get_time()

			#print(blue_car_detected)
			if blue_car_detected:
				move.angular.z = 0
				if license_plate_recognized and stall_recognized:
					stationary_count = 0
					move.linear.x = nominal_speed
					if plate_not_published:
						plate_to_score_tracker_pub.publish("FineLine,golden," + str(stall_number) + "," + predicted_plate_number)
						plate_not_published = False
						#count+=1
				else:
					move.linear.x = 0
					stationary_count+=1
					if stationary_count > 50:
						move.linear.x = 0.1
						num_times_inched_forward+=1
						if num_times_inched_forward > 5:
							move.linear.x = nominal_speed
							if len(sample_set) != 0:
								sample_set_counter = Counter(sample_set)
								predicted_plate_number = sample_set_counter.most_common(1)[0][0]
								print("output***************************************", predicted_plate_number)
								sample_set = []
								plate_to_score_tracker_pub.publish("FineLine,golden," + str(stall_number) + "," + predicted_plate_number)
								license_plate_recognized = True
								plate_not_published = False
						else:
							stationary_count = 0
					blue_car_pub.publish("blue car detected")

			elif red_line_detected:
				move.angular.z = 0
				move.linear.x = 0

			else:
				midpoint_road, road_detected = get_center(img=cv_image) #gets index of center of road			

				driving_straight = follow_line(midpoint_road,road_detected) #does PID control of robot

			if driving_straight:
				blue_car_detected = detect_blue_car(cv_image,blue_car_detected)
			else:
				blue_car_detected = False

		#prev_control = True
		#velocity_pub.publish(move)
	else:
		#this should only execute once competition ends... for now
		move.angular.z = 0
		move.linear.x = 0

	#print(move.angular.z)
	if not crossing_crosswalk:
		velocity_pub.publish(move)

numbers = ['0','1','2','3','4','5','6','7','8','9']
letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def callback_plate_NN(plate):
	global sess1
	global graph1
	global license_plate_recognized
	global plate_number
	global num_samples
	global sample_set
	global predicted_plate_number
	plate_number = ""
	cv_plate = bridge.imgmsg_to_cv2(plate, desired_encoding='passthrough')
	cv_plate = cv_plate.reshape(1,cv_plate.shape[0],cv_plate.shape[1],3)
	split_plate = split_images(cv_plate,training_flag=False)
	for j in range(len(split_plate)):
		grey = cv2.cvtColor(split_plate[j], cv2.COLOR_BGR2GRAY).reshape(split_plate[j].shape[0],split_plate[j].shape[1],1)
		img_aug = np.expand_dims(grey, axis=0)
		#Thanks to Grace for finding this solution: https://github.com/tensorflow/tensorflow/issues/28287
		with graph1.as_default():
			set_session(sess1)
			y_predict = license_plate_NN.predict(img_aug)[0]
		predicted_character = mapPredictionToCharacter(y_predict)
		if j == 0 or j == 1:
			if predicted_character in numbers:
				if predicted_character == '1':
					predicted_character = 'I'
				elif predicted_character == '0':
					predicted_character = 'O'
				elif predicted_character == '2':
					predicted_character = 'Z'
				elif predicted_character == '6':
					predicted_character = 'G'
		elif j == 2 or j == 3:
			if predicted_character in letters:
				if predicted_character == 'O':
					predicted_character = '0'
				elif predicted_character == 'Q':
					predicted_character = '0'
				elif predicted_character == 'D':
					predicted_character = '0'
				elif predicted_character == 'Z':
					predicted_character = '2'
				elif predicted_character == 'I':
					predicted_character = '1'
				elif predicted_character == 'J':
					predicted_character = '1'
				elif predicted_character == 'G':
					predicted_character = '6'


		plate_number += str(predicted_character)
	#print(plate_number)
	#predicted_plate_number = plate_number
	#license_plate_recognized = True
	#stash the license plates and wait until the stash hits 20. once it hits 20, then take
	if num_samples < 15:
		sample_set.append(plate_number)
		print(sample_set)
		#license_plate_recognized = False
		num_samples+=1
	else:
		num_samples = 0
		license_plate_recognized = True
		sample_set_counter = Counter(sample_set)
		predicted_plate_number = sample_set_counter.most_common(1)[0][0]
		print("output***************************************", predicted_plate_number)
		sample_set = []

	#can add something like this to test whether its actually 

def callback_stall_NN(guess):
	global stall_recognized
	global stall_number
	stall_recognized = True
	#extract original string from data
	stall_number = str(guess.data)[0]
	print("predicted stall: " + str(stall_number))
	# later on, do the if statement with the 7 and the 8

# ./run_sim.sh -vpg
# roslaunch my_controller run_comp.launch
# ./score_tracker.py

def callback_crosswalk(safe_to_cross):
	global recently_crossed
	global crossing_crosswalk
	global red_line_detected
	safe_to_cross = str(safe_to_cross.data)

	if not recently_crossed:
		if safe_to_cross == "True":
			move.angular.z = 0
			move.linear.x = 0.3
			red_line_detected = False
			crossing_crosswalk = True
			velocity_pub.publish(move)
			recently_crossed = True
			red_line_count = 0
			time.sleep(2.5)
			crossing_crosswalk = False
			#print("safe")
		else:
			#print("unsafe")
			move.angular.z = 0
			move.linear.x = 0
			velocity_pub.publish(move)

rospy.init_node('control_node')
bridge = CvBridge()
velocity_pub = rospy.Publisher('/R1/cmd_vel',Twist,queue_size = 1)
blue_car_pub = rospy.Publisher('/blue_car_detection',String,queue_size = 1)
plate_to_score_tracker_pub = rospy.Publisher('/license_plate',String,queue_size = 1)
red_line_pub = rospy.Publisher('/red_line',String,queue_size = 1)
move = Twist()
image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callback_image)  #/rrbot/camera1/image_raw', Image,callback_image)
control_sub = rospy.Subscriber('/control',String,callback_control)
license_plate_sub = rospy.Subscriber('/sim_license_plates',Image,callback_plate_NN)
pedestrian_sub = rospy.Subscriber('/pedestrian',String,callback_crosswalk)
#license_plate_sub = rospy.Subscriber('/sim_stall',Image,callback_stall_NN)
stall_sub = rospy.Subscriber('/sim_stall',String,callback_stall_NN)
rospy.spin()
