import cv2
import numpy as np
from loadVideo import loadVideo
from getBoundingBox import getBoundingBox
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from estimateFeatureTranslation import estimateFeatureTranslation
from applyGeometricTransformation import applyGeometricTransformation

def plotBbox(img, new_bbox):
	new_bbox.shape
	for bbox_pts in new_bbox:
		x0 = bbox_pts[0][0]
		y0 = bbox_pts[0][1]
		x_dist = bbox_pts[3][0] - bbox_pts[0][0]
		y_dist = bbox_pts[3][1] - bbox_pts[0][1]
		new_img = cv2.rectangle(im, (x0, y0), (x_dist, y_dist), (0, 255, 0), 2)

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
	bbox_pts = []
	for i in range(0, numObjects):
		bbox_list, bbox_pts, init_img = getBoundingBox(init_img, bbox_list, bbox_pts)
	startXs, startYs, _ = getFeatures(init_img_gray, bbox_list, maxCorners, qualityLevel, minDistance)

	output[0, :, :, :] = init_img
	cnt = 0
	i = 0
	curr_bbox = bbox_pts

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
		[Xs, Ys, new_bbox] = applyGeometricTransformation(startXs, startYs, newXs, newYs, curr_bbox)

		plotBbox(img2, new_bbox)

		output[frame+1,:,:,:] = img2

		startXs = newXs
		startYs = newYs
		cnt += 1

if __name__ == '__main__':
    objectTracking()