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

FALSE = 0
TRUE = 1

image_back = "/home/fizzer/ros_ws/src/my_controller/pictures/cropped_close_ups/back_semi_white.jpg"
image_front = "/home/fizzer/ros_ws/src/my_controller/pictures/cropped_close_ups/front_semi_white.jpg"
dist_scale_back = 0.85 # 0.7
dist_scale_front = 0.85 # 0.7
positive_match = 25 # 6

# roslaunch my_controller SIFT_ped.launch

detect_ped_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
detect_thresh = 8
count = 0

# Cropping variables
X1 = 550
X2 = 750
Y1 = 400
Y2 = 600

### Need to add publisher and subscriber for integration purposes
# publisher to tell other programs the status of the pedestrian
# subscriber to determine when to start running this algorithm program

def cropImage(width_start, width_end, height_start, height_end, frame):
	return frame[height_start:height_end,width_start:width_end,0:3]

def pedestrian_ID(image_path, title, robot_frame, dist_scale):
	match = FALSE
	sift = cv2.xfeatures2d.SIFT_create()
	frame = cropImage(X1,X2,Y1,Y2,robot_frame)

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
		# to avoid many false results, take descriptors that have short distances between them
		# play with the dist_scale constant: 0.6, 0.7, 0.8 - potential values
		if m.distance < dist_scale * n.distance:
			good_points.append(m)

	img_matches = cv2.drawMatches(img, kp_image, frame, kp_frame, good_points, frame)
	cv2.imshow("Matches", img_matches)
	cv2.waitKey(1)

	# pedestrian_ID - the line of *'s are arbitrary, just a quick check that we've identified the pedestrian in the terminal
	if len(good_points) > positive_match:
		#print(title + ": ******************* pedestrian_ID: " + str(len(good_points)))
		match = TRUE		
	else:
		#print(title + ": no pedestrian_ID: " + str(len(good_points)))
		match = FALSE

	return match

def callback_image(data):
	global detect_ped_list
	global count
	cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
	#frame = cropImage(X1,X2,Y1,Y2,cv_image)

	# try recognizing backside of pedestrian 
	matches1 = pedestrian_ID(image_back, "back", cv_image, dist_scale_back)
	# try recognizing backside of pedestrian 
	matches2 = pedestrian_ID(image_front, "front", cv_image, dist_scale_front)

	if matches1 or matches2:
		count = count + 1
		detect_ped_list = []
	else:
		detect_ped_list.append(FALSE)

	if len(detect_ped_list) < detect_thresh:
		print("***********************detected!")
	else:
		print("not detected")

	#print("back: " + str(matches1))
	#print("front: " + str(matches2))

# ROS setup stuff below
rospy.init_node('comp_image_read_node')
bridge = CvBridge()
velocity_pub = rospy.Publisher('/R1/cmd_vel',Twist,queue_size = 1)
move = Twist()
image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callback_image) 
rospy.spin()

'''
# Stuff I did not use but may be helpful

# drawing keypoints on image
#img = cv2.drawKeypoints(img, kp_image,img)
#cv2.imshow("img keypoints", img)
#cv2.waitKey(1)

# homography stuff that I never used
		query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
		train_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

		# matrix shows object from its perspective?
		matrix, mask = cv2.findpedestrian_ID(query_pts, train_pts, cv2.RANSAC, 5.0)
		matches_mask = mask.ravel().tolist()        # extract points from mask and put into a list

		# Perspective transforms, helps with pedestrian_ID
		h, w = img.shape        # height and width of original image
		#print(h)
		#print(w)

		pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)    # points gets h and w of image. does not work with int32, but float32 works
		dst = cv2.perspectiveTransform(pts, matrix)

		# convert to an integer for pixel pointers, (you can't point to a decimal of a pixel)
		# True is for "closing the lines"
		# next is the colour we select, in bgr, we have selected blue
		# thickness = 3
		pedestrian_ID = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

		cv2.imshow("pedestrian_ID", pedestrian_ID)
'''