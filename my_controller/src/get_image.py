#! /usr/bin/env python

import rospy
import cv2
import time

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

x_goal_coor = 400
x_grace = 50
y_goal_coor =  400
y_grace = 20
FALSE = 0
road_threshold = 110
center_of_mass = x_goal_coor
last_move = 0 							
# 0 for straight, -1 for right, 1 for left
linear_vel = 0.12
linear_vel_low = 0.030
angular_vel = 0.55

def callback_image(data):
	cv_image = bridge.imgmsg_to_cv2(data, "bgr8") 		
	(rows,cols,channels) = cv_image.shape
	print("rows: " + str(rows))
	print("cols: " + str(cols))

	move.angular.z = angular_vel
	move.linear.x = linear_vel_low
	velocity_pub.publish(move)
	
	'''

	#convert the current frame to grayscale
	grey_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

	# traverse the lower rows of pixels and find the road
	list_x = []
	list_y = []
	k = 750
	while k < 800:
		j = 0
	  	while j < 800:
	    		if (grey_frame[k,j]) > road_threshold:
	      			grey_frame[k,j] = 0
	   		else:
	      			list_x.append(j)
	      			list_y.append(k)
	    		j += 1
	  	k += 1
		  
	# x calculations
	sum_x = 0
	for x in list_x:
		sum_x += x
	if len(list_x) == 0:
		center_of_mass_x = FALSE
	else:
		center_of_mass_x = sum_x / len(list_x)

	if center_of_mass_x == FALSE:
		if last_move == 0:
			move.linear.x = linear_vel
			velocity_pub.publish(move)
		if last_move == -1:
			move.angular.z = -angular_vel
			velocity_pub.publish(move)
		if last_move == 1:
			move.angular.z = angular_vel
			velocity_pub.publish(move)

		#do the angular velocity command to spin
		move.angular.z = -0.6
		velocity_pub.publish(move)

	else: 
		#we have been able to detect the road
		x_difference = x_goal_coor - center_of_mass_x
		if x_difference < x_grace: 
			if x_difference > -x_grace:
				#go straight
				move.angular.z = 0
				move.linear.x = linear_vel
				last_move = 0
				velocity_pub.publish(move)
			else:
				#turn right
				move.angular.z = -angular_vel
				move.linear.x = linear_vel_low
				last_move = -1
				velocity_pub.publish(move)
		else:
			#turn left
			move.angular.z = angular_vel
			move.linear.x = linear_vel_low
			last_move = 1
			velocity_pub.publish(move)
	'''

rospy.init_node('comp_image_read_node')
bridge = CvBridge()
velocity_pub = rospy.Publisher('/R1/cmd_vel',Twist,queue_size = 1)
move = Twist()
image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callback_image)  #/rrbot/camera1/image_raw', Image,callback_image)
rospy.spin()






