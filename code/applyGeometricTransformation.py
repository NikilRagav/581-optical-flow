import cv2
import numpy as np
import skimage.transform

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):

	maxDistance = 4
	[numFeatures, numObjects] = startXs.shape
	Xs = []
	Ys = []
	newbbox = np.full(np.array(bbox).shape, fill_value=-1)

	for i in range(numObjects):
		startX = startXs[:, i]
		startY = startYs[:, i]
		newX = newXs[:, i]
		newY = newYs[:, i]

		dist = ((startX - newX)**2 + (startY - newY)**2)

		startX = startX[dist <= maxDistance**2]
		startY = startY[dist <= maxDistance**2]
		newX = newX[dist <= maxDistance**2]
		newY = newY[dist <= maxDistance**2]

		trans = skimage.transform.SimilarityTransform()
		start = np.matrix.transpose(np.vstack((startX, startY)))
		new = np.matrix.transpose(np.vstack((newX, newY)))
		trans.estimate(start, new)

		if not Xs:
			Xs = newX
		else:
			np.append(Xs, newX)

		if not Ys:
			Ys = newY
		else:
			np.append(Ys, newY)

		startBox = np.vstack((np.matrix.transpose(np.array(bbox[i,:,:])), np.ones(4)))
		newBox = np.dot(trans.params, startBox)
		newbbox[i,:,:] = np.matrix.transpose(newBox[0:2,:])

		return Xs, Ys, newbbox


		



