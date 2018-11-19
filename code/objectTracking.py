import cv2
import numpy as np
from loadVideo import loadVideo
from getBoundingBox import getBoundingBox
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from estimateFeatureTranslation import estimateFeatureTranslation
from applyGeometricTransformation import applyGeometricTransformation

def objectTracking():
	numObjects = 2
	maxCorners = 20
	qualityLevel = 0.01
	minDistance = 8
	filename = "Easy.mp4"
	numFrames = 2

	#load video
	frameData, length, h, w, fps = loadVideo(filename, 0, numFrames)

	#output video
	output = np.zeros([numFrames, h, w, 3])

	#get bounding boxes
	init_img = frameData[0, :, :, :]
	init_img_gray = cv2.cvtColor(frameData[0, :, :, :], cv2.COLOR_BGR2GRAY)
	bbox_list = []
	for i in range(0, numObjects):
		bbox_list, init_img = getBoundingBox(init_img, bbox_list)
	startXs, startYs, _ = getFeatures(init_img_gray, bbox_list, maxCorners, qualityLevel, minDistance)

	output[0, :, :, :] = init_img
	cnt = 0
	i = 0

	while i < length-1:

		if cnt == numFrames-1:
			frameData, _, _, _, _ = loadVideo(filename, cnt+1, numFrames)
			startXs, startYs, _ = getFeatures(init_img_gray, bbox_list, maxCorners, qualityLevel, minDistance)
			cnt = 0
		
		img1 = frameData[i, :, :, :]
		img2 = frameData[i+1, :, :, :]
		img1_gray = cv2.cvtColor(frameData[i, :, :, :], cv2.COLOR_BGR2GRAY)
		img2_gray = cv2.cvtColor(frameData[i+1, :, :, :], cv2.COLOR_BGR2GRAY)
		[newXs, newYs] = estimateAllTranslation(startXs, startYs, img)
		[Xs, Ys, newbbox] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)

		output[frame+1,:,:,:] = img2

		startXs = newXs
		startYs = newYs
		cnt += 1

if __name__ == '__main__':
    objectTracking()