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

image_front = "/home/fizzer/Pictures/hsv_lp/cropped/black_far_P.png" 
dist_scale_front = 0.7 # 0.7
positive_match = 4 # 6

# Cropping variables
X_crop_left = 100
X_crop_right = 640
Y_crop_top = 400
Y_crop_bottom = 720

# centroid
X_centroid_list = []
Y_centroid_list = []
centroid_avg_error = 30
prev_x = 0
prev_y = 0

def cropImage(width_start, width_end, height_start, height_end, frame):
	return frame[height_start:height_end,width_start:width_end,0:3]

def license_plate_detect(image_path, title, robot_frame, dist_scale):
	global X_centroid_list
	global Y_centroid_list
	global prev_x
	global prev_y

	match = FALSE
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

	#'''	

	# roslaunch my_controller SIFT_license_plate.launch

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

	#'''
	good_points = []
	for m, n in matches:
		# to avoid many false results, take descriptors that have short distances between them
		# play with the dist_scale constant: 0.6, 0.7, 0.8 - potential values
		if m.distance < dist_scale * n.distance:
			good_points.append(m)

	img_matches = cv2.drawMatches(img, kp_image, frame, kp_frame, good_points, frame)
	cv2.imshow("Matches", img_matches)
	cv2.waitKey(1)

	enter_main = FALSE
	if len(good_points) > positive_match:
		enter_main = TRUE
		match = TRUE		

		query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
		train_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

		# matrix shows object from its perspective?
		matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
		matches_mask = mask.ravel().tolist()        # extract points from mask and put into a list

		# Perspective transforms, helps with homography
		h, w = img.shape        # height and width of original image

		pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)    # points gets h and w of image. does not work with int32, but float32 works
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
					print(" ")
					print(str(x_centroid_avg) + " " + str(y_centroid_avg))

				else:
					# new and last value are close but far from avg, therefore the robot probably moved a lot, recalculate centroid at new location
					new_list1 = []
					new_list2 = []
					X_centroid_list = new_list1
					Y_centroid_list = new_list2
			else:
				# nothing happens this run, we don't know if the robot moved or got a noisy result
				nothing = 0

		prev_x = x_centroid
		prev_y = y_centroid

		print(str(x_centroid) + " " + str(y_centroid))
		
		centroid_frame = cv2.circle(original_frame,(x_centroid,y_centroid),6,(0,0,255),-1)
		#cv2.imshow("centroid",centroid_frame)
		#cv2.waitKey(1)
		
		# convert to an integer for pixel pointers, (you can't point to a decimal of a pixel)
		# True is for "closing the lines"
		# next is the colour we select, in rgb, we have selected red
		# thickness = 3
		homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

		#cv2.imshow("Homography", homography)
		#cv2.waitKey(1)
	else:
		match = FALSE

	# we got the centroid of the p homography
	# will now try to capture the license plate using the centroid of P
	# remember to use the offsets from our earlier crop

	if len(X_centroid_list) != 0:
		x_centroid_avg = int(sum(X_centroid_list)/len(X_centroid_list))
		y_centroid_avg = int(sum(Y_centroid_list)/len(Y_centroid_list))

		x_move_left = -30
		y_move_down = 45
		license_plate_height = 30
		license_plate_width = 130
		# draw rectangle function requires the top-left and bottom-right corner
		top_left_x = x_centroid_avg + X_crop_left + x_move_left
		top_left_y = y_centroid_avg + Y_crop_top + y_move_down
		bottom_right_x = top_left_x + license_plate_width
		bottom_right_y = top_left_y + license_plate_height

		license_plate_trace = cv2.rectangle(robot_frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,0,255),3)
		cv2.imshow("trace",license_plate_trace)
		cv2.waitKey(1)
	
	#'''
	return match
	#return 5

def callback_image(data):
	cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
	#frame = cropImage(X1,X2,Y1,Y2,cv_image)
	matches1 = license_plate_detect(image_front, "front", cv_image, dist_scale_front)


# ROS setup stuff below
rospy.init_node('detect_car_node')
bridge = CvBridge()
velocity_pub = rospy.Publisher('/R1/cmd_vel',Twist,queue_size = 1)
move = Twist()
image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callback_image) 
rospy.sleep(1)
rospy.spin()