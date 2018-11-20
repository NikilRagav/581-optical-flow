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