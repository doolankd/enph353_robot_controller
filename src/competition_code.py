#! /usr/bin/env python

import rospy
import cv2 as cv2
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

FALSE = 0
TRUE = 1

start_time = 60 # simulation seconds
end_time = start_time + 60*1 # only for testing purposes, this 1 should be a 4 for 4 sim minutes
time_seconds = 0
run_once = 0
angular_vel = 0
prev_time = 0

# function included for funsies
def callback_image(data):
	cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

def callback_time(data):
	# allows us to change global variables
	global time_seconds
	global run_once
	global angular_vel
	global prev_time
	
	# gets current time in seconds
	time_seconds = int(str(data)[16:21])

	# begin competition
	if time_seconds == start_time and run_once == 0:
		#angular_vel = 2
		license_publishing("0","XR58") # the license plate digits are filler, don't mean anything
		run_once = 1
		control_pub.publish("start")

	# test sending license plates
	if time_seconds == start_time + 30 and run_once == 1:
		license_publishing("1","CH42")
		run_once = 0

	# end competition
	if time_seconds == end_time and run_once == 0:
		#angular_vel = 0
		control_pub.publish("stop")
		license_publishing("-1","XR58") # the license plate digits are filler, don't mean anything
		run_once = 1

	# print speed every second
	if time_seconds != prev_time:
		print(time_seconds)
		prev_time = time_seconds

	#move.angular.z = angular_vel
	#velocity_pub.publish(move)

'''
The string you send must contain the following comma separated values:
team ID: max 8 characters (no spaces)
team password: max 8 characters (no spaces)
license plate location: int (0 to 8); 0 is a special case - see below
license plate id: 4 characters (no spaces)

example:'TeamRed,multi21,4,XR58'
'''
def license_publishing(license_plate_location,plate_number):
	license_plate_pub.publish("FineLine,golden," + license_plate_location + "," + plate_number)


'''
some shortcuts

roslaunch my_controller competition_code.launch

cd ~/ros_ws/src/2020T1_competition/enph353/enph353_utils/scripts
./run_sim.sh -vpg
./score_tracker.py
'''

# ROS setup stuff below
rospy.init_node('comp_begin_node')
bridge = CvBridge()
image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callback_image)
time_sub = rospy.Subscriber('/clock',String,callback_time)
license_plate_pub = rospy.Publisher('/license_plate',String,queue_size = 1)
control_pub = rospy.Publisher('/control',String,queue_size = 1)
velocity_pub = rospy.Publisher('/R1/cmd_vel',Twist,queue_size = 1)
move = Twist()
rospy.spin()