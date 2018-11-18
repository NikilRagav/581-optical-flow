'''
flow_runner.py
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from scipy import sparse
from scipy import interpolate
from skimage import feature
from PIL import Image
import imageio

from findDerivatives import findDerivatives
from loadVideo import loadVideo

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def flow_runner():
    '''
    It probably makes sense to do this 10 frames at a time.
    That way, we don't store huge amounts of data in RAM.

    (we should keep the last frame of every set of 10 so that we can 
     make sure we track the same feature across each set of frames)

    take in the video 
    as a numFramesxHxWxnumColors np array
    where the first frame of the set is the last frame of the previous
    (in the case of the first time, the first frame is the first frame in the video)
    
    compute the grayscale version
    numFramesxHxW

    compute Ix (convolve with vertical sobel and broadcast)
    numFramesxHxW

    compute Iy (horizontal sobel and broadcast)
    numFramesxHxW

    compute It
    (numFrames-1)xHxW


    Find corners in each frame
    numFramesxnumCornersx2
    Find corresponding points between each pair of frames
    (numFrames-1)xnumCorners
    (boolean matrix)



    For each frame
    for each 11x11 patch 
    [[
        sum(Ix*Ix)      sum(Ix*Iy)
        sum(Ix*Iy)      sum(Iy*Iy)
    ]]

    *
    [[
        u
        v
    ]]

    =

    -[[
        sum(Ix*It)
        sum(Iy*It)
    ]]

    We can collect all of the A and b for all the windows and solve all the u v at once
    Because solve() broadcasts in 3D

    A  must be (numWindows, 2, 2)
    b  must be (numWindows, 2, 1)
    uv will be (numWindows, 2, 1)

    uv = np.linalg.solve(A, b)

    we'll have to iteratively do this a few times

    Let's load 10 frames at a time so we don't take up too much ram and also so that we can calculate flow correctly 
    even if objects leave the frame etc
    '''

    #filepaths for input and output



    #number of frames to calculate at a time
    numFrames = 10
    currentFrame = 0

    #open output video file

    current_frames, totalFrames = loadVideo(input, currentFrame, numFrames)
    #H, W come from the video file itself
    numFrames, H,W = current_frames.shape[0], current_frames.shape[1], current_frames.shape[2]

    #let user draw bounding box on frame[0]
    # for now we'll just do the tracking on the whole frame

    #constants for corner detection
    maxCorners = 50
    qualityLevel = .05
    minDistance = np.min(H,W)/windowsize

    while (currentFrame + numFrames - 1) < totalFrames:
        #get grayscale
        frames_gray = rgb2gray(current_frames)

        #get all derivatives

        #unfortunately convolve2d doesn't broacast in 3D
        frames_Ix = np.zeros((numFrames, H, W, 3))
        frames_Iy = np.zeros((numFrames, H, W, 3))

        for i in range(numFrames):
            __, frames_Ix[i], frames_Iy[i], __ = findDerivatives(frames_gray[i])

        frames_It = current_frames[1:] - current_frames[0:-1]

        #get features
        #goddamnit please vectorize APIs!!
        '''
        corner_list = np.empty( (numFrames), dtype='object')
        for i in range(numFrames):
            corner_list[i] = feature.corner_peaks( feature.corner_shi_tomasi(frames_gray[i]) )
        '''
        corner_list = np.zeros( (numFrames, maxCorners, 2) )
        for i in range(numFrames):
            corner_list[i] = cv2.goodFeaturesToTrack(frames_gray[i], maxCorners, qualityLevel, minDistance)

        #remove points that are outside the bounding box for this frame
        # for now, we'll just do the whole frame

        #do actual calc

        #at the end of the actual calc,
        # we should have a numFramesxnumFeaturesx2 vector with u v values for each feature in each frame

        #get the vectors to draw
        #throw out the vectors for image points that were originally not in the image
        # if for a particular frame we ended up with fewer than our maximum number of features

        #calculating new bounding box
        #take the old features that were within the bounding box
        #apply the u,v transformation for them
        #do ransac on them to estimate a similarity transform for the bounding box
        #do a similarity transform on the old box coords
        #draw new box at transformed coords

        #append to output video

        currentFrame += numFrames
        current_frames, __ = loadVideo(input, currentFrame+numFrames, numFrames)
        numFrames = current_frames.shape[0]
    
    return 0