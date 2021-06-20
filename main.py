
import numpy as np
import imutils
import time
import cv2
import os
from tracker import Tracker
from typing import List
from collections import deque
from random import randrange



#create queue for printing trajectory on screen
tracker = None
trajectory = deque(maxlen=100)

#variables to detect change possession
change_possession_right_flag = False
change_possession_left_flag = False
change_possession = -1

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.abspath('../yolo/yolov3.weights')
configPath = os.path.abspath('../yolo/yolov3.cfg')

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# initialize the video stream, pointer to output video file, and
# frame dimensions, video inside material folder
cap = cv2.VideoCapture('../material/CV_basket.mp4')
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(cap.get(prop))
	print('[INFO] ', total ,' total frames in video')
except:
	print('[INFO] could not determine # of frames in video')
	print('[INFO] no approx. completion time can be provided')
	total = -1



while True:
	
	ret, frame = cap.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not ret:
		break
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

    # loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:

			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > 0.5:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype('int')
				
				
				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs	
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				
				
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

	# if tracker is None create a new one, select randomlly a player to track
	# passing its bounding box
	if len(idxs) > 0:
		if tracker is None:
			tracker = Tracker(frame=frame, bounding_box=boxes[randrange(len(boxes))])
		
	#create red bounding box for tracked person
	coordinates = tracker.track(frame=frame)

	if coordinates is not None:
		(x_t, y_t, w_t, h_t) = coordinates
		center_base = (int(x_t + w_t/2), int(y_t + h_t))
		trajectory.appendleft(center_base)

		#plot a red line for the trajectory
		for i in range(1, len(trajectory)):
			if trajectory[i-1] is None or trajectory[i] is None:
				continue
			cv2.line(frame, trajectory[i-1], trajectory[i], color=(0,0,255), thickness=2)
		cv2.rectangle(frame, (x_t, y_t), (x_t + w_t, y_t + h_t), color=(0,0,255), thickness=3)
		cv2.putText(frame, 'Tracked', (x_t, y_t - 5),  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale= 0.5, color = (0,0,255), thickness= 2)
	else:
		tracker = None

	# check the number of detection and if > 4 for having
	# a better view of a change possession. If all the detection
	# are below or above the center of the frame a the change possession
	# counter will be incremented
	if len(idxs) > 4:

		if change_possession_right_flag == False:
			player_on_right = 0
			for i in idxs.flatten():
				point = boxes[i][0]+(boxes[i][2]//2)
				if point  >= W//2:
					player_on_right += 1
			

			if player_on_right == len(idxs):
				change_possession +=1
				change_possession_right_flag = True
				change_possession_left_flag = False

		if  change_possession_left_flag == False:
			player_on_left = 0
			for i in idxs.flatten():
				point = boxes[i][0]+(boxes[i][2]//2)		
				if point <= W//2:
					player_on_left += 1
		
			if player_on_left == len(idxs):
				change_possession +=1
				change_possession_left_flag = True
				change_possession_right_flag = False
	
	
	#plotting on frame detected players with bounding boxes
	if len(idxs) > 0:
		for i in idxs.flatten():

			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0,255, 0), thickness=2)
			
	#put information on frame, number of people and possession change
	cv2.putText(frame, 
                	'Number of people detected : ' + str(len(idxs)), 
                	(20, 40), 
                	cv2.FONT_HERSHEY_SIMPLEX, 1, 
                	(0, 255, 255), 
                	2, 
                	cv2.LINE_4)

	if change_possession == -1:
		text_change_possetion = '0'
	else:
		text_change_possetion = str(change_possession)

	cv2.putText(frame, 
                'Change possession : ' + text_change_possetion, 
                (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
	

	#create a video writer to save the basic video analysis
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*'MPEG')
		writer = cv2.VideoWriter('../output/output.avi', fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print('[INFO] single frame took ', elap ,'seconds')
			print('[INFO] estimated total time to finish: ', (elap * total)//60, 'minutes')

	writer.write(frame)

writer.release()
cap.release()

