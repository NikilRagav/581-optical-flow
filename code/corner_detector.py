# -*- coding: utf-8 -*-

'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line,
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
import numpy as np
from skimage import feature
def corner_detector(img):
  return feature.corner_shi_tomasi(img)
