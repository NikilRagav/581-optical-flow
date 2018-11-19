import cv2
import numpy as np
from scipy import signal
import sklearn.preprocessing

def estimateAllTranslation(startXs, startYs, img1, img2):

kernel_x = sklearn.preprocessing.normalize(np.array([[1, -1, 0]]))
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.double)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.double)
	
fx = signal.convolve2d(img1_gray, kernel_x, 'same')
fy = signal.convolve2d(img2_gray, kernel_y, 'same')
