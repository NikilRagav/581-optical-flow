import cv2
import numpy as np

#this probably loads as BGR

#return a numFramesxHxWx3 matrix
def loadVideo(filename, numFrames):
	vid = cv2.VideoCapture(filename)
	status, img = vid.read()
	cnt = 0
	frame_data = []
	while status and cnt < numFrames:
		frame_data.append(img)
		status, img = vid.read()
		cnt += 1