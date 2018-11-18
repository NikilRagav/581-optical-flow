import cv2
import numpy as np

def displayFeatures(filename):
	im = cv2.imread(filename)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	bbox = getBoundingBox(im)
	maxCorners = 20
	qualityLevel = 0.01
	minDistance = 8
	x, y = getFeatures(im, bbox, maxCorners, qualityLevel, minDistance)
	print(x)
	print(y)

def getBoundingBox(im):
	bbox = cv2.selectROI(im)

	new_img = np.copy(im)
	new_img = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
	#cv2.imwrite("tester_output.jpg", new_img)

	# imCrop = im[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
	# cv2.imshow("Image", imCrop)
    # cv2.waitKey(0)
	return bbox, new_img

def getFeatures(img, bbox, maxCorners, qualityLevel, minDistance):

	first = True
	for x0, y0, w, h in bbox:
		window = img[y0:y0+h, x0:x0+w]
		corners = np.int0(cv2.goodFeaturesToTrack(window, maxCorners, qualityLevel, minDistance))

		x_f = [corner[0][0]+x0 for corner in corners]
		y_f = [corner[0][1]+y0 for corner in corners]

		x_f = np.vstack(np.pad(x_f, (0, maxCorners-len(x_f)), 'constant', constant_values=-1))
		y_f = np.vstack(np.pad(y_f, (0, maxCorners-len(y_f)), 'constant', constant_values=-1))

		if first:
			x = x_f
			y = y_f
			first = False
		else:
			x = np.append(x, x_f, axis=1)
			y = np.append(y, y_f, axis=1)

		return x, y

if __name__ == '__main__':
    displayFeatures("test.jpg")
