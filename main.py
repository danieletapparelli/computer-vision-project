# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from tracker2 import TrackerL
from typing import List



net = cv2.dnn.readNetFromDarknet('../yolo/yolov3.cfg', '../yolo/yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)


ln = net.getLayerNames()
print(len(ln), ln)


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.abspath('../yolo/coco.names')
print(labelsPath)
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
#COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
#	dtype="uint8")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.abspath('../yolo/yolov3.weights')
print("Weight path : "+ weightsPath)
configPath = os.path.abspath('../yolo/yolov3.cfg')
print("Config path : "+ configPath)
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture('../material/CV_basket.mp4')
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1
# loop over frames from the video file stream

tracker_flag = False
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
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
				(centerX, centerY, width, height) = box.astype("int")
				
				
				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates,
				# confidences, and class IDs
				
				boxes.append([x, y, int(width), int(height)])
				#kalman = boxes.append(confidence)
			

				confidences.append(float(confidence))
				classIDs.append(classID)
				
				
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
		0.3)

	print("BOXES", boxes)
	if tracker_flag == False:

		tracker = TrackerL(frame=frame, bounding_box=boxes[0])
		tracker_flag = True
	
	if tracker_flag == True:
		(x1,y1,w1,h1) = tracker.track(frame=frame)
		cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), color=(0,0,254), thickness=3)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			
	

			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# print([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])
			# x1, y1, x2, y2 = get_corner_coordinates([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])
			# kalman = np.array([x1, y1, x2, y2, 1])
			# print("Kalmann : "+str(kalman))
			# print("Shape of the array = ",np.shape(kalman))
			# draw a bounding box rectangle and label on the frame
			#color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i]) 
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
			
			

			font = cv2.FONT_HERSHEY_SIMPLEX
			
			cv2.putText(frame, 
                'Change possetion : ', 
                (100, 100), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
	if len(idxs)>3:
		cv2.putText(frame, 
                	'Number of people detected : '+str(len(idxs)), 
                	(50, 50), 
                	font, 1, 
                	(0, 255, 255), 
                	2, 
                	cv2.LINE_4)
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*'MPEG')
		writer = cv2.VideoWriter('merda.avi', fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))
	# write the output frame to disk
	writer.write(frame)
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

