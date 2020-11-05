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

FALSE = 0
TRUE = 1

image_path_cross_walk_r_pov = "/home/fizzer/ros_ws/src/my_controller/pictures/perspecitve_cross_walk.jpg"
image_path_birds_eye = "/home/fizzer/ros_ws/src/my_controller/pictures/top_cross_walk.jpg"
image_path_pedestrian = "/home/fizzer/ros_ws/src/my_controller/pictures/pedestrian.jpg"
dist_scale_r_pov = 0.93
dist_scale_birds_eye = 0.83
dist_scale_pedestrian = 1.0

def homography(image_path, title, robot_frame, dist_scale):
	homog = FALSE
	sift = cv2.xfeatures2d.SIFT_create()
	frame = robot_frame

	# Camera Robot Frame
	kp_frame, desc_frame = sift.detectAndCompute(frame, None)
	
	# Our comparison image
	img = cv2.imread(image_path, 0)
	kp_image, desc_image = sift.detectAndCompute(img, None)
	img = cv2.drawKeypoints(img, kp_image,img)

	# Feature Matching
	index_params = dict(algorithm=0, trees=5)
	search_params = dict()
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(desc_image, desc_frame, k=2)

	good_points = []
	for m, n in matches:
		# to avoid many false results, take descriptors that have short distances between them
		# play with this constant in front of n.distance: 0.6, 0.8
		if m.distance < dist_scale * n.distance:
			good_points.append(m)

	# Homography
	# if we find at least 4 matches, we will draw homography - arbitrary
	if len(good_points) > 4:
		print(title + ": homography: " + str(len(good_points)))
		homog = TRUE
		
	else:
		print(title + ": no homography: " + str(len(good_points)))
		homog = FALSE

	return homog

def callback_image(data):
	cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

	# first recognize birds-eye view of crosswalk
	homog1 = homography(image_path_birds_eye, "birds-eye", cv_image, dist_scale_birds_eye)
	# robot pov detection
	homog2 = homography(image_path_cross_walk_r_pov, "robot pov", cv_image, dist_scale_r_pov)
	# pedestrian
	# pedestrian detection is really hard unless you're about to collide
	#homography(image_path_pedestrian, "pedestrian", cv_image, dist_scale_pedestrian)

	if (homog1 == TRUE) and (homog2 == TRUE):
		print("Detect Crosswalk")

	'''
	move.angular.z = angular_vel
	move.linear.x = linear_vel_low
	velocity_pub.publish(move)
	'''

# ROS setup stuff below
rospy.init_node('comp_image_read_node')
bridge = CvBridge()
velocity_pub = rospy.Publisher('/R1/cmd_vel',Twist,queue_size = 1)
move = Twist()
image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callback_image)  #/rrbot/camera1/image_raw', Image,callback_image)
rospy.spin()
