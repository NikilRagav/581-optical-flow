import cv2
import numpy as np

def getBoundingBox(im, bbox_list):
	bbox = cv2.selectROI(im)
	bbox_list.append(bbox)
	new_img = np.copy(im)
	new_img = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
	#cv2.imwrite("tester_output.jpg", new_img)

	# imCrop = im[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
	# cv2.imshow("Image", imCrop)
    # cv2.waitKey(0)
	return bbox_list, new_img