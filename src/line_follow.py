#! /usr/bin/env python

import rospy
import cv2
import time
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

#PID VARIABLES:

middle_screen_margin = 10
nominal_speed = 0.1
max_turn_speed = 1

img_width = 0
previous_error = 0

d_history = [0,0,0,0,0]

control_robot = False

#DETECTION INTERUPT VARIABLES:

blue_car_detected = False
blue_running_time = 0.0
first_pass = True

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


def detect_blue_car(original_image,blue_detected_previous):
	#sub_img = hsv_img[400][:][:]
	#height, width, shape = img.shape #720, 1280, 3

	#blue_car_b = 121
	#blue_car_g = 19
	#blue_car_r = 19

	global time_detected_blue_car
	global blue_running_time
	global first_pass
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

	else:

		STARTING_Y_PIXEL = 500
		STARTING_X_PIXEL = 150

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


	return blue_car_detected

def callback_control(cmd):

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

	#print(control_robot)

	if control_robot:

		if rospy.get_time() - comp_start_time < 1:
			#**************************************************
			#NOTE: THESE VALUES ARE FOR A nominal_speed of 0.06
			#**************************************************
			move.angular.z = 0.0
			move.linear.x = 0.3
		elif rospy.get_time() - comp_start_time < 2.0:


			move.angular.z = 1
			move.linear.x = 0
			#gets robot to turn left once competition starts

		#this should only execute once competition starts

		else:

			cv_image = bridge.imgmsg_to_cv2(img, "bgr8")#image robot sees
			#print(blue_car_detected)
			if blue_car_detected:
				move.angular.z = 0
				move.linear.x = nominal_speed

			else:
				midpoint_road, road_detected = get_center(img=cv_image) #gets index of center of road			

				driving_straight = follow_line(midpoint_road,road_detected) #does PID control of robot

			if driving_straight:
				blue_car_detected = detect_blue_car(cv_image,blue_car_detected)
			else:
				blue_car_detected = False

	else:
		#this should only execute once competition ends
		move.angular.z = 0
		move.linear.x = 0

	#print(move.angular.z)
	velocity_pub.publish(move)


rospy.init_node('control_node')
bridge = CvBridge()
velocity_pub = rospy.Publisher('/R1/cmd_vel',Twist,queue_size = 1)
move = Twist()
image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callback_image)  #/rrbot/camera1/image_raw', Image,callback_image)
control_sub = rospy.Subscriber('/control',String,callback_control)
rospy.spin()
