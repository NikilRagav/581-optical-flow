import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from numpy.linalg import lstsq

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):

	window_size = 10

	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.double)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.double)
	  
	[H, W] = img1_gray.shape
	It = img2_gray - img1_gray

	window_ymin = max(0, startY-window_size/2)
	window_xmin = max(0, startX-window_size/2)
	window_ymax = min(H-1, startY+window_size/2)
	window_xmax = min(W-1, startX+window_size/2)

	inter_Ix = interp2d(np.arange(0, W), np.arange(0, H), Ix)
	inter_Iy = interp2d(np.arange(0, W), np.arange(0, H), Iy)
	inter_It = interp2d(np.arange(0, W), np.arange(0, H), It)

	Ix_window = inter_Ix[window_ymin:window_ymax, window_xmin:window_xmax]
	Iy_window = inter_Iy[window_ymin:window_ymax, window_xmin:window_xmax]
	It_window = inter_It[window_ymin:window_ymax, window_xmin:window_xmax]

	A = np.zeros((2, 2))
	A[0][0] = np.sum(np.multiply(Ix_window, Ix_window))
	A[0][1] = np.sum(np.multiply(Ix_window, Iy_window))
	A[1][0] = np.sum(np.multiply(Ix_window, Iy_window))
	A[1][1] = np.sum(np.multiply(Iy_window, Iy_window))

	B = np.zeros((2, 1))
	B[0][0] = 0 - np.sum(np.multiply(Ix_window, It_window))
	B[1][0] = 0 - np.sum(np.multiply(Iy_window, It_window))

	A_inv = np.linalg.inv(A)
	u, v = A_inv.dot(B)


	newX = startX + u
	newY = startY + v
	return newX, newY
