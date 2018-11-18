import cv2
import numpy as np

def getBoundingBox(img):
	bbox = cv2.selectROI(im)

	# imCrop = im[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
	# cv2.imshow("Image", imCrop)
    # cv2.waitKey(0)
	return bbox