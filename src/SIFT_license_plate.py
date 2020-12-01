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

FALSE = False
TRUE = True

comparison_img = "/home/fizzer/ros_ws/src/my_controller/pictures/cropped_close_ups/black_far_P_GOOD.png" 
dist_scale_front = 0.8 # 0.7
positive_match = 4 

# Cropping variables
X_crop_left = 0
X_crop_right = 640
Y_crop_top = 400
Y_crop_bottom = 720

# centroid
X_centroid_list = []
Y_centroid_list = []
centroid_avg_error = 25
prev_x = 0
prev_y = 0
prev_match = FALSE
prev_prev_match = FALSE
found_match = False
license_plate_frame = None

sim_time_seconds = 0

#detecting if blue car seen
blue_car_detected = False

def cropImage(width_start, width_end, height_start, height_end, frame):
	return frame[height_start:height_end,width_start:width_end,0:3]

def license_plate_detect(image_path, title, robot_frame, dist_scale):
	global X_centroid_list
	global Y_centroid_list
	global prev_x
	global prev_y
	global prev_match
	global prev_prev_match
	global found_match
	global license_plate_frame

	found_match = False

	sift = cv2.xfeatures2d.SIFT_create()
	frame = cropImage(X_crop_left,X_crop_right,Y_crop_top,Y_crop_bottom,robot_frame)
	original_frame = frame

	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	# detect black in input frame
	frame_threshold = cv2.inRange(hsv,(0,0,0),(180,255,30))
	frame = frame_threshold
	#cv2.imshow("HSV",frame)
	#cv2.waitKey(1)

	# roslaunch my_controller SIFT_license_plate.launch
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

	# save to list
	good_points = []
	for m, n in matches:
		# to avoid many false results, play with the dist_scale constant: 0.6, 0.7, 0.8 
		if m.distance < dist_scale * n.distance:
			good_points.append(m)

	img_matches = cv2.drawMatches(img, kp_image, frame, kp_frame, good_points, frame)
	'''cv2.imshow("Matches", img_matches)
	cv2.waitKey(1)'''

	match = FALSE
	if len(good_points) > positive_match:
		match = TRUE		

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
		
		# homography square corner points
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

		# take good centroid points - accounting fo error
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
					print("avgs: " + str(x_centroid_avg) + " " + str(y_centroid_avg))

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

		print("vals: " + str(x_centroid) + " " + str(y_centroid))
		print(" ")
		print("number of good points: " + str(len(good_points)))
		
		# draws the centroid onto a frame - EVENTUALLY GET RID OF THIS
		#centroid_frame = cv2.circle(original_frame,(x_centroid,y_centroid),6,(0,0,255),-1)

		# points gets h and w of image. does not work with int32, but float32 works
		pts = np.float32([[0, 0], [0, h + 50], [w + 50, h + 50], [w + 50, 0]]).reshape(-1, 1, 2)   
		dst = cv2.perspectiveTransform(pts, matrix)
		# homography of P
		homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

	else:
		match = FALSE

	# once we have populated the centroid list, will now capture license plate and write to a file
	if len(X_centroid_list) != 0 and match and prev_match and prev_prev_match:
		found_match = True
		# new idea
		# real frame x,y centroid coordinates
		x_centroid_avg = int(sum(X_centroid_list)/len(X_centroid_list)) + X_crop_left
		y_centroid_avg = int(sum(Y_centroid_list)/len(Y_centroid_list)) + Y_crop_top

		# box range below x,y centroid coordinates
		box_width = 200 # 180
		box_height = 80

		# where we want to start capturing data from
		box_upper_left_corner_X = x_centroid_avg - 55
		box_upper_left_corner_Y = y_centroid_avg + 10

		box_frame = robot_frame[box_upper_left_corner_Y:box_height+box_upper_left_corner_Y,box_upper_left_corner_X:box_width+box_upper_left_corner_X,0:3].copy()
		box_frame_clean = robot_frame[box_upper_left_corner_Y:box_height+box_upper_left_corner_Y,box_upper_left_corner_X:box_width+box_upper_left_corner_X,0:3].copy()

		# transform image space to HSV - use Miti's script (HSV_thresh_script.py) to find good hsv values
		hsv_box_frame = cv2.cvtColor(box_frame, cv2.COLOR_BGR2HSV)
		low_hsv = np.array([100,29,0])			#45,107
		high_hsv = np.array([109,53,183])
		gray_mask = cv2.inRange(hsv_box_frame,low_hsv,high_hsv)
		edges = cv2.Canny(gray_mask,75,150) #75, 150

		# roslaunch my_controller SIFT_license_plate.launch
		# ./run_sim.sh -vpg
		# cd ~/ros_ws/src/2020T1_competition/enph353/enph353_utils/scripts

		# since the screen is black, and only white for edges
		indices = np.where(edges != [0])
		coordinates = zip(indices[1], indices[0])

		# Find left and right most coordinate (only looking at x_value)
		# i think its (x,y)
		rand_x, rand_y = coordinates[0]
		left_most_x = rand_x
		left_most_point = coordinates[0]
		right_most_x = rand_x
		right_most_point = coordinates[0]
		for point in coordinates:
			current_x = point[0]

			if current_x > right_most_x:
				right_most_x = current_x
				right_most_point = point

			if current_x < left_most_x:
				left_most_x = current_x
				left_most_point = point

		# captured the top line!
		character_height = 25 # how much we'll drop down to get the bottom of the license plate
		x_shift = 10 #5 # we'll move this much to the left and right of the captured line

		top_left_point = (left_most_point[0]-x_shift,left_most_point[1])
		top_right_point = (right_most_point[0]+x_shift,right_most_point[1])
		
		y_bottom_left = top_left_point[1] + character_height
		x_bottom_left = top_left_point[0]
		y_bottom_right = top_right_point[1] + character_height
		x_bottom_right = top_right_point[0]
		bottom_left_point = (x_bottom_left,y_bottom_left)
		bottom_right_point = (x_bottom_right,y_bottom_right)

		# drawing the lines on the cropped image
		line_frame = cv2.line(box_frame,top_left_point,top_right_point,(0,255,0),3)
		line_frame = cv2.line(box_frame,bottom_left_point,bottom_right_point,(0,255,0),3)
		#cv2.imshow("lines", line_frame)
		#cv2.waitKey(1)

		# get all the points again
		T_R_X, T_R_Y = top_right_point
		B_R_X, B_R_Y = bottom_right_point
		T_L_X, T_L_Y = top_left_point
		B_L_X, B_L_Y = bottom_left_point

		# from my design team code, get max width and height between points
		widthA = np.sqrt(((B_R_X - B_L_X) ** 2) + ((B_R_Y - B_L_Y) ** 2))
		widthB = np.sqrt(((T_R_X - T_L_X) ** 2) + ((T_R_Y - T_L_Y) ** 2))
		maxWidth = min(int(widthA), int(widthB))

		heightA = np.sqrt(((T_R_X - B_R_X) ** 2) + ((T_R_Y - B_R_Y) ** 2))
		heightB = np.sqrt(((T_L_X - B_L_X) ** 2) + ((T_L_Y - B_L_Y) ** 2))
		maxHeight = min(int(heightA), int(heightB))

		# preparing for perspective transform
		dst = np.float32([
			[maxWidth, 0],
			[maxWidth, maxHeight],
			[0, 0],
			[0, maxHeight]], dtype = "float32")

		original_lines = np.float32([
			[T_R_X, T_R_Y],
			[B_R_X, B_R_Y],
			[T_L_X, T_L_Y],
			[B_L_X, B_L_Y]])

		# M is the transform matrix: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/ 
		# comutes perspective transform given max height and width, and the transform matrix, M
		M = cv2.getPerspectiveTransform(original_lines, dst)
		license_plate_frame = cv2.warpPerspective(box_frame_clean, M, (maxWidth, maxHeight))
		'''cv2.imshow("license plate", license_plate_frame)
		cv2.waitKey(1)'''

		# roslaunch my_controller SIFT_license_plate.launch

	prev_prev_match = prev_match
	prev_match = match
	return found_match

def callback_blue_car(b_car_detected):
	global blue_car_detected
	#extract original string from data
	blue_car_detected = str(b_car_detected.data)

def callback_image(data):
	global blue_car_detected
	if blue_car_detected == "blue car detected":
		cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
		plate_detected = license_plate_detect(comparison_img, "front", cv_image, dist_scale_front)
		print(plate_detected)
		if plate_detected:
			license_plate = bridge.cv2_to_imgmsg(license_plate_frame, encoding="passthrough")
			license_plate_img_pub.publish(license_plate)
		blue_car_detected = False


def callback_time(data):
	global sim_time_seconds
	# gets current time in seconds
	sim_time_seconds = int(str(data)[16:21])

# ROS setup stuff below
rospy.init_node('detect_car_node')
bridge = CvBridge()
velocity_pub = rospy.Publisher('/R1/cmd_vel',Twist,queue_size = 1)
license_plate_img_pub = rospy.Publisher('/sim_license_plates',Image,queue_size=1)
move = Twist()
blue_car_sub = rospy.Subscriber('/blue_car_detection',String,callback_blue_car)
image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callback_image) 
time_sub = rospy.Subscriber('/clock',String,callback_time)
rospy.sleep(1) # wait 1 second to let things start up
rospy.spin()