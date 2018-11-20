import cv2
import numpy as np
from getBoundingBox import getBoundingBox
from getFeatures import 
import matplotlib.pyplot as plt

if __name__ == '__main__':
  # setup video capture
	cap = cv2.VideoCapture("input_videos/Easy.mp4")
	ret,img1 = cap.read()
	ret,img2 = cap.read()
	cap.release()

	maxCorners = 20
	qualityLevel = 0.01
	minDistance = 8

	bbox_list = []
	bbox_pts = []
	bbox_list, bbox_pts, new_img = getBoundingBox(img1, bbox_list, bbox_pts)
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	startXs, startYs, _ = getFeatures(img1_gray, bbox_list, maxCorners, qualityLevel, minDistance)
	newXs = startXs + 5
	newYs = startYs + 7
	print(startXs, startYs)
	print(newXs, newYs)
	#nnewXs, nnewYs = estimateAllTranslation(newXs, newYs, img2, img3)
	#print(len(newXs[0]))
	#print(len(newYs[0]))

	plt.figure()
	plt.imshow(img2)
	#plt.plot(newYs, newXs, 'w+')
	plt.axis('off')
	plt.show()