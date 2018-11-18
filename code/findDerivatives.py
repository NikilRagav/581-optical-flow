'''
  File name: findDerivatives.py
  Author: Nikil Ragav
  Date created: 
'''

'''
  File clarification:
    Compute gradient information of the input grayscale image
    - Input I_gray: H x W matrix as image
    - Output Mag: H x W matrix represents the magnitude of derivatives
    - Output Magx: H x W matrix represents the magnitude of derivatives along x-axis
    - Output Magy: H x W matrix represents the magnitude of derivatives along y-axis
    - Output Ori: H x W matrix represents the orientation of derivatives
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from PIL import Image

def findDerivatives(I_gray):
  # TODO: your code here
  
  #vertical sobel is derivative in horizontal (x)
  vs = np.asarray([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
  MagX = signal.convolve(I_gray,vs,"same")

  #horizontal sobel is derivative in vertical (y)
  hs =  np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  MagY = signal.convolve(I_gray,hs,"same")
  
  #full derivative is just sum (not fxy)
  Mag = np.hypot(MagY,MagX)
  
  #the resulting angles are in radians (from -pi to pi)
  Ori = np.arctan2(MagY,MagX)
  
  
  return [Mag, MagX, MagY, Ori]