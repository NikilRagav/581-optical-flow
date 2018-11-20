import cv2
import numpy as np
from getBoundingBox import getBoundingBox
from getFeatures import getFeatures

def displayFeatures(filename):
	numObjects = 2
	maxCorners = 20
	qualityLevel = 0.01
	minDistance = 8

	im = cv2.imread(filename)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	bbox_list = []
	bbox_pts = []
	for i in range(0, numObjects):
		bbox_list, bbox_pts, newImage = getBoundingBox(im, bbox_list, bbox_pts)	
	print(bbox_pts)
	x, y, _ = getFeatures(im, bbox_list, maxCorners, qualityLevel, minDistance)
	print(x)
	print(y)

if __name__ == '__main__':
    displayFeatures("test.jpg")
