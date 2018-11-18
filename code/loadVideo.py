import cv2
import numpy as np

#this probably loads as BGR

#return a 10xHxWx3 matrix
def loadVideo(filename, start, numFrames):
	vid = cv2.VideoCapture(filename)
	status, img = vid.read()
	frame_data = []
	length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

	start_frame = start
	if start is not 0:
		start_frame -= 1
	vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

	cnt = 0
	while status and cnt < numFrames:
		print(cnt)
		print(img)
		frame_data.append(img)
		status, img = vid.read()
		cnt += 1
	return np.array(frame_data), length