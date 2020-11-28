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

FALSE = 0
TRUE = 1

image_front = "/home/fizzer/ros_ws/src/my_controller/pictures/cropped_close_ups/black_far_P_GOOD.png" 
write_location = "/home/fizzer/NN_pics/" #"/home/fizzer/ros_ws/src/my_controller/pictures/NN_Training/"
files_written = 0
files_written_max = 100
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
prev_match = FALSE

sim_time_seconds = 0

def cropImage(width_start, width_end, height_start, height_end, frame):
	return frame[height_start:height_end,width_start:width_end,0:3]

def license_plate_detect(image_path, title, robot_frame, dist_scale):
	global X_centroid_list
	global Y_centroid_list
	global prev_x
	global prev_y
	global prev_match
	global files_written

	reserved_frame = np.copy(robot_frame)

	sift = cv2.xfeatures2d.SIFT_create()
	frame = cropImage(X_crop_left,X_crop_right,Y_crop_top,Y_crop_bottom,robot_frame)
	original_frame = frame

	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	# trying to detect black
	frame_threshold = cv2.inRange(hsv,(0,0,0),(180,255,30))
	frame = frame_threshold
	cv2.imshow("HSV",frame)
	cv2.waitKey(1)

	#cv2.imshow("normal",robot_frame)
	#cv2.waitKey(1)	

	# roslaunch my_controller SIFT_stall_num_gather_data.launch
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
		# to avoid many false results, take descriptors that have short distances between them
		# play with the dist_scale constant: 0.6, 0.7, 0.8 - potential values
		if m.distance < dist_scale * n.distance:
			good_points.append(m)

	img_matches = cv2.drawMatches(img, kp_image, frame, kp_frame, good_points, frame)
	cv2.imshow("Matches", img_matches)
	cv2.waitKey(1)

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
					X_centroid_list.append(prev_x)
					X_centroid_list.append(x_centroid)
					Y_centroid_list.append(prev_y)
					Y_centroid_list.append(y_centroid)
			else:
				# nothing happens this run, we don't know if the robot moved or got a noisy result
				nothing = 0

		prev_x = x_centroid
		prev_y = y_centroid

		print(str(x_centroid) + " " + str(y_centroid))
		print(" ")
		print("number of good points: " + str(len(good_points)))
		
		centroid_frame = cv2.circle(original_frame,(x_centroid,y_centroid),6,(0,0,255),-1)
		#cv2.imshow("centroid",centroid_frame)
		#cv2.waitKey(1)

		pts = np.float32([[0, 0], [0, h + 50], [w + 50, h + 50], [w + 50, 0]]).reshape(-1, 1, 2)    # points gets h and w of image. does not work with int32, but float32 works
		dst = cv2.perspectiveTransform(pts, matrix)
		homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
		# Homography of P
		#cv2.imshow("Homography", homography)
		#cv2.waitKey(1)

		# new idea - afterwards, compute all this outside this big loop, so that we can deal with averages
		# real frame x,y centroid coordinates
		x_centroid_avg = int(sum(X_centroid_list)/len(X_centroid_list)) + X_crop_left
		y_centroid_avg = int(sum(Y_centroid_list)/len(Y_centroid_list)) + Y_crop_top

		#roslaunch my_controller SIFT_stall_num_gather_data.launch


		'''

		# box range below x,y centroid coordinates
		box_width = 200 # 180
		box_height = 80

		# where we want to start capturing data from
		box_upper_left_corner_X = x_centroid_avg - 55
		box_upper_left_corner_Y = y_centroid_avg + 10

		box_frame = robot_frame[box_upper_left_corner_Y:box_height+box_upper_left_corner_Y,box_upper_left_corner_X:box_width+box_upper_left_corner_X,0:3].copy()
		box_frame_clean = robot_frame[box_upper_left_corner_Y:box_height+box_upper_left_corner_Y,box_upper_left_corner_X:box_width+box_upper_left_corner_X,0:3].copy()
		#cv2.imshow("crop", box_frame)
		#cv2.waitKey(1)
		#print(box_frame[50,50,0]) #105
		#print(box_frame[45,90,0]) #105
		#gray_box_frame = cv2.cvtColor(box_frame, cv2.COLOR_BGR2GRAY)
		#cv2.imshow("gray", gray_box_frame)
		#cv2.waitKey(1)

		hsv_box_frame = cv2.cvtColor(box_frame, cv2.COLOR_BGR2HSV)
		# using 50 - 110 captures the British columbia and the flag of the plate
		low_hsv = np.array([100,29,0])			#45,107
		high_hsv = np.array([109,53,183])
		gray_mask = cv2.inRange(hsv_box_frame,low_hsv,high_hsv)
		edges = cv2.Canny(gray_mask,75,150) #75, 150

		# roslaunch my_controller SIFT_license_plate.launch
		# ./run_sim.sh -vpg
		# cd ~/ros_ws/src/2020T1_competition/enph353/enph353_utils/scripts

		cv2.imshow("edges", edges)
		cv2.waitKey(1)
		#print(edges)

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

		#print("left: " + str(left_most_point))
		#print("right: " + str(right_most_point))
		#print(indices[0])
		#print(coordinates)

		# captured the top line!
		character_height = 25
		x_shift = 10 #5

		top_left_point = (left_most_point[0]-x_shift,left_most_point[1])
		top_right_point = (right_most_point[0]+x_shift,right_most_point[1])
		
		y_bottom_left = top_left_point[1] + character_height
		x_bottom_left = top_left_point[0]
		y_bottom_right = top_right_point[1] + character_height
		x_bottom_right = top_right_point[0]
		bottom_left_point = (x_bottom_left,y_bottom_left)
		bottom_right_point = (x_bottom_right,y_bottom_right)

		line_frame = cv2.line(box_frame,top_left_point,top_right_point,(0,255,0),3)
		line_frame = cv2.line(box_frame,bottom_left_point,bottom_right_point,(0,255,0),3)
		cv2.imshow("lines", line_frame)
		cv2.waitKey(1)

		T_R_X, T_R_Y = top_right_point
		B_R_X, B_R_Y = bottom_right_point
		T_L_X, T_L_Y = top_left_point
		B_L_X, B_L_Y = bottom_left_point

		# from my design team code
		widthA = np.sqrt(((B_R_X - B_L_X) ** 2) + ((B_R_Y - B_L_Y) ** 2))
		widthB = np.sqrt(((T_R_X - T_L_X) ** 2) + ((T_R_Y - T_L_Y) ** 2))
		maxWidth = min(int(widthA), int(widthB))

		heightA = np.sqrt(((T_R_X - B_R_X) ** 2) + ((T_R_Y - B_R_Y) ** 2))
		heightB = np.sqrt(((T_L_X - B_L_X) ** 2) + ((T_L_Y - B_L_Y) ** 2))
		maxHeight = min(int(heightA), int(heightB))

		# code I copied immediately below, subtracted 1 from maxWidth and maxHeight
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
		M = cv2.getPerspectiveTransform(original_lines, dst)
		license_plate_frame = cv2.warpPerspective(box_frame_clean, M, (maxWidth, maxHeight))

		cv2.imshow("license plate", license_plate_frame)
		cv2.waitKey(1)
		
		# roslaunch my_controller SIFT_license_plate.launch
		'''

	else:
		match = FALSE

	# we have found the centroid of P, will now draw a box to the right of the centroid to try to capture the stall number
	if len(X_centroid_list) != 0 and match and prev_match:
		x_centroid_avg = int(sum(X_centroid_list)/len(X_centroid_list))
		y_centroid_avg = int(sum(Y_centroid_list)/len(Y_centroid_list))

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
		frame_threshold = cv2.inRange(hsv,(0,0,0),(180,255,30))

		print(frame_threshold.shape)
		frame_threshold = frame_threshold[top_left_y:bottom_right_y,top_left_x:bottom_right_x]
		cv2.imshow("real stall", frame_threshold)
		cv2.waitKey(1)

		# roslaunch my_controller SIFT_stall_num_gather_data.launch

		if files_written < files_written_max:
			cv2.imwrite(write_location + "{}_5".format(sim_time_seconds) + ".png", frame_threshold)
			files_written = files_written + 1

		print("files_written: " + str(files_written))

		# wait for the image to be processed
		rospy.sleep(1)

		#license_plate_trace = cv2.rectangle(robot_frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,0,255),3)
		#cv2.imshow("trace",license_plate_trace)
		#cv2.waitKey(1)
	#else:
		#cv2.imshow("trace",robot_frame)
		#cv2.waitKey(1)


	# we got the centroid of the p homography
	# will now try to capture the license plate using the centroid of P
	# remember to use the offsets from our earlier crop

	'''
	if len(X_centroid_list) != 0 and match and prev_match:
		
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
		
		# M is the transform matrix: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/ 
		M = cv2.getPerspectiveTransform(original_lines, dst)
		license_plate_frame = cv2.warpPerspective(box_frame_clean, M, (maxWidth, maxHeight))

		if files_written < files_written_max:
			cv2.imwrite(write_location + "{}".format(sim_time_seconds) + ".jpg", license_plate_frame)
			files_written = files_written + 1

		cv2.imshow("license plate", license_plate_frame)
		cv2.waitKey(1)
		print("files_written: " + str(files_written))

		# wait for the image to be processed
		rospy.sleep(1)

		#license_plate_trace = cv2.rectangle(robot_frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,0,255),3)
		#cv2.imshow("trace",license_plate_trace)
		#cv2.waitKey(1)
	#else:
		#cv2.imshow("trace",robot_frame)
		#cv2.waitKey(1)

	'''
	
	prev_match = match
	#return match
	return 5

def callback_image(data):
	cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
	matches1 = license_plate_detect(image_front, "front", cv_image, dist_scale_front)

def callback_time(data):
	global sim_time_seconds
	# gets current time in seconds
	sim_time_seconds = int(str(data)[16:21])


# ROS setup stuff below
rospy.init_node('detect_car_node')
bridge = CvBridge()
velocity_pub = rospy.Publisher('/R1/cmd_vel',Twist,queue_size = 1)
move = Twist()
image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image,callback_image) 
time_sub = rospy.Subscriber('/clock',String,callback_time)
rospy.sleep(1) # wait 1 second to let things start up
rospy.spin()