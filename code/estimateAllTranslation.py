import cv2
import numpy as np
from getBoundingBox import getBoundingBox
from getFeatures import getFeatures
from estimateFeatureTranslation import estimateFeatureTranslation
import matplotlib.pyplot as plt

def estimateAllTranslation(startXs, startYs, img1, img2):

	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.double)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.double)

	[numFeatures, numObj] = startXs.shape
	sobel_kernel = 5

	newXs = np.zeros(startXs.shape)
	newYs = np.zeros(startXs.shape)

	Ix = cv2.Sobel(img1_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	Iy = cv2.Sobel(img1_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

	for i in range(numFeatures):
		for j in range(numObj):
			startX = startXs[i][j]
			startY = startYs[i][j]
			newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, img1_gray, img2_gray)
			newXs[i][j] = newX
			newYs[i][j] = newY

	return newXs, newYs

	
if __name__ == '__main__':
	# setup video capture
	cap = cv2.VideoCapture("input_videos/Easy.mp4")
	ret,img1 = cap.read()
	ret,img2 = cap.read()
	ret,img3 = cap.read()
	cap.release()

	maxCorners = 20
	qualityLevel = 0.01
	minDistance = 8

	bbox_list = []
	bbox_pts = []
	bbox_list, bbox_pts, new_img = getBoundingBox(img1, bbox_list, bbox_pts)
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	startXs, startYs, _ = getFeatures(img1_gray, bbox_list, maxCorners, qualityLevel, minDistance)
	newXs, newYs = estimateAllTranslation(startXs, startYs, img1, img2)
	nnewXs, nnewYs = estimateAllTranslation(newXs, newYs, img2, img3)
	print(len(nnewXs[0]))
	print(len(nnewYs[0]))

	plt.figure()
	plt.imshow(img3)
	plt.plot(nnewXs, nnewYs, 'r+')
	plt.axis('off')
	plt.show()