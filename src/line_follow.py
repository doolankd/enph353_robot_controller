#! /usr/bin/env python

import rospy
import cv2
import time

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

#PID VARIABLES:

middle_screen_margin = 50
nominal_speed = 0.06
max_turn_speed = 1

img_width = 0
previous_error = 0

d_history = [0,0,0,0,0]

control_robot = False

#DETECTION INTERUPT VARIABLES:

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

		error = img_width / 2 - midpoint_road

		if abs(error) <= middle_screen_margin:
			move.linear.x = nominal_speed
			move.angular.z = 0
	
		else:
			calibrated_error = (-1)*error*(-1)
			calibrated_prev_error = (-1)*previous_error
			#do this since angular.z turns right for negative vals

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

def callback_control(cmd):

	global control_robot

	cmd = str(cmd.data)
	print(cmd)
	
	if cmd == "start":
		control_robot = True
		#print(control_robot)
	elif cmd == "stop":
		control_robot = False
		#print(control_robot)
	else:
		print("skipped")

def callback_image(img):

	global control_robot

	#print(control_robot)

	if control_robot:
		#print("in here")

		cv_image = bridge.imgmsg_to_cv2(img, "bgr8") #image robot sees
		midpoint_road, road_detected = get_center(img=cv_image)

		follow_line(midpoint_road,road_detected)

	else:
		move.angular.z = 0
		move.linear.x = 0

	velocity_pub.publish(move)


rospy.init_node('control_node')
bridge = CvBridge()
velocity_pub = rospy.Publisher('/R1/cmd_vel',Twist,queue_size = 1)
move = Twist()
image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callback_image)  #/rrbot/camera1/image_raw', Image,callback_image)
control_sub = rospy.Subscriber('/control',String,callback_control)
rospy.spin()
