import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from numpy.linalg import lstsq
from getBoundingBox import getBoundingBox
from getFeatures import getFeatures

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):

	window_size = 10
	  
	[H, W] = img1.shape
	It = img2 - img1

	window_ymin = max(0, startY-window_size/2)
	window_xmin = max(0, startX-window_size/2)
	window_ymax = min(H-1, startY+window_size/2)
	window_xmax = min(W-1, startX+window_size/2)

	inter_Ix = interp2d(np.arange(0, W), np.arange(0, H), Ix)
	inter_Iy = interp2d(np.arange(0, W), np.arange(0, H), Iy)
	inter_It = interp2d(np.arange(0, W), np.arange(0, H), It)

	window_Ix = np.zeros((np.round(window_ymax - window_ymin).astype(int), np.round(window_xmax - window_xmin).astype(int)))
	window_Iy = np.zeros((np.round(window_ymax - window_ymin).astype(int), np.round(window_xmax - window_xmin).astype(int)))
	window_It = np.zeros((np.round(window_ymax - window_ymin).astype(int), np.round(window_xmax - window_xmin).astype(int)))

	for i in range(np.round(window_ymax - window_ymin).astype(int)):
		for j in range(np.round(window_xmax - window_xmin).astype(int)):
			window_Ix[i][j] = inter_Ix(window_xmin + j, window_ymin + i)
			window_Iy[i][j] = inter_Iy(window_xmin + j, window_ymin + i)
			window_It[i][j] = inter_It(window_xmin + j, window_ymin + i)

	# window_Ix = inter_Ix[window_ymin:window_ymax, window_xmin:window_xmax]
	# window_Iy = inter_Iy[window_ymin:window_ymax, window_xmin:window_xmax]
	# window_It = inter_It[window_ymin:window_ymax, window_xmin:window_xmax]

	A = np.zeros((2, 2))
	A[0][0] = np.sum(np.multiply(window_Ix, window_Ix))
	A[0][1] = np.sum(np.multiply(window_Ix, window_Iy))
	A[1][0] = np.sum(np.multiply(window_Ix, window_Iy))
	A[1][1] = np.sum(np.multiply(window_Iy, window_Iy))

	B = np.zeros((2, 1))
	B[0][0] = 0 - np.sum(np.multiply(window_Ix, window_It))
	B[1][0] = 0 - np.sum(np.multiply(window_Iy, window_It))

	A_inv = np.linalg.inv(A)
	u, v = A_inv.dot(B)


	newX = startX + u
	newY = startY + v
	return newX, newY

if __name__ == '__main__':
	# setup video capture
	cap = cv2.VideoCapture("input_videos/Easy.mp4")
	ret,img1 = cap.read()
	ret,img2 = cap.read()
	cap.release()

	img1 = np.array(img1)
	img2 = np.array(img2)

	maxCorners = 20
	qualityLevel = 0.01
	minDistance = 8

	bbox_list = []
	bbox_pts = []
	bbox_list, bbox_pts, new_img = getBoundingBox(img1, bbox_list, bbox_pts)
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	x, y, _ = getFeatures(img1_gray, bbox_list, maxCorners, qualityLevel, minDistance)
	Ix, Iy = np.gradient(img1_gray)
	Ix = np.array(Ix)
	Iy = np.array(Iy)

	newXs = []
	newYs = []
	for i in range(len(x)):
		for j in range(len(x[i])):
		  startX = x[i][j]
		  startY = y[i][j]
		  newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, img1_gray, img2_gray)
		  newXs.append(newX)
		  newYs.append(newY)

	print(len(newXs))
	print(len(newYs))

	plt.figure()
	plt.imshow(img1_gray, cmap='gray')
	plt.plot(x, y, 'r+')
	plt.axis('off')

	plt.figure()
	plt.imshow(img2_gray, cmap='gray')
	plt.plot(newXs, newYs, 'r+')
	plt.axis('off')
	plt.show()