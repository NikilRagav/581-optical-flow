import cv2
import numpy as np

# returns NxF matrix with N row coordinates of the features across F objects
# returns NxF matrix with N col coordinates of the features across F objects

#if there are fewer features than the maxCorners (N), then pad with the coordinates of the first feature
def getFeatures(img, bbox, maxCorners, qualityLevel, minDistance):

	first = True
	for (x0, y0, w, h) in bbox:
		window = img[y0:y0+h, x0:x0+w].astype(np.uint8)
		corners = np.int0(cv2.goodFeaturesToTrack(window, maxCorners, qualityLevel, minDistance))

		x_f = [corner[0][0]+x0 for corner in corners]
		y_f = [corner[0][1]+y0 for corner in corners]

		x_f = np.vstack(np.pad(x_f, (0, maxCorners-len(x_f)), 'constant', constant_values=x_f[0]))
		y_f = np.vstack(np.pad(y_f, (0, maxCorners-len(y_f)), 'constant', constant_values=y_f[0]))

		if first:
			x = x_f
			y = y_f
			first = False
		else:
			x = np.append(x, x_f, axis=1)
			y = np.append(y, y_f, axis=1)

	return x, y, np.concatenate((y,x), axis=1)
