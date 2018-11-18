import cv2
import numpy as np

#return a 10xHxWx3 matrix
def loadVideo(filename):
	vid = cv2.VideoCapture(filename)
	status, img = vid.read()
	cnt = 0
	frame_data = []
	while status and cnt < 10:
		frame_data.append(img)
		status, img = vid.read()
		cnt += 1