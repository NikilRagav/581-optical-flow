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
import os
import imageio

from findDerivatives import findDerivatives
from loadVideo import loadVideo
from getFeatures import getFeatures

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def bgr2gray(bgr):
  return np.dot(bgr[...,:3], [0.114,0.587,0.299])

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
    in_folder = "input_videos/"
    out_folder = "input_videos/"
    filename = "Easy.mp4"

    in_video_path = os.path.join(in_folder,filename)
    out_video_path = os.path.join(out_folder,filename)

    #number of frames to calculate at a time
    numFrames = 10
    currentFrame = 0


    current_frames, totalFrames, fps = loadVideo(in_video_path, currentFrame, numFrames)
    #H, W come from the video file itself
    numFrames, H,W = current_frames.shape[0], current_frames.shape[1], current_frames.shape[2]

    #open output video file
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (W,H))

    #let user draw bounding box on frame[0]
    # for now we'll just do the tracking on the whole frame

    #constants for corner detection
    maxFeatures = 50
    qualityLevel = .05
    minDistance = (8/360)*H #keep same ratio of 8 pixel distance for a 360p video regardless of resolution

    windowsize = 9
    half_window = np.floor(windowsize/2)

    while (currentFrame + numFrames - 1) < totalFrames:
        #get grayscale
        frames_gray = bgr2gray(current_frames)

        # #pad all frames by floor(windowsize/2) along all sides
        # frames_gray_padded = np.pad(frames_gray, ( (half_window,half_window), (half_window,half_window) ), mode='reflect')

        #get all derivatives

        #unfortunately convolve2d doesn't broacast in 3D
        frames_Ix = np.zeros((numFrames, H, W))
        frames_Iy = np.zeros((numFrames, H, W))

        for i in range(numFrames):
            __, frames_Ix[i], frames_Iy[i], __ = findDerivatives(frames_gray[i])

        frames_It = frames_gray[1:] - frames_gray[0:-1]

        #get features
        #goddamnit please vectorize APIs!!
        feature_list = np.zeros( (numFrames, maxFeatures, 2) )
        for i in range(numFrames):
            __, __, feature_list[i] = getFeatures(frames_gray[i], (0,0,W,H), maxFeatures, qualityLevel, minDistance)
        # for now, we'll just do the whole frame
        zaxis = np.rollaxis( np.rollaxis( np.outer(np.ones((1,maxFeatures,1)), np.arange(3) )[np.newaxis, :], 2, 0), 2, 1)
        feature_list = np.concatenate( (zaxis, feature_list), axis=2)
        #now it is z,x,y coordinates (where z is frame number)

        #do actual calc across all frames
        uv = np.zeros( (numFrames-1, maxFeatures, 2))

        #windows is a maxFeatures x 1 array
        summation_kernel = np.ones( (windowsize,windowsize) )
        frames_summed_Ix = np.zeros_like(frames_gray)
        frames_summed_Iy = np.zeros_like(frames_gray)
        frames_summed_Ix_squared = np.zeros_like(frames_gray)
        frames_summed_Iy_squared = np.zeros_like(frames_gray)
        frames_summed_Ix_Iy = np.zeros_like(frames_gray)
        for i in range(numFrames):
            frames_summed_Ix[i] = signal.convolve2d(frames_Ix, summation_kernel, mode='same', boundary='symm')
            frames_summed_Iy[i] = signal.convolve2d(frames_Iy, summation_kernel, mode='same', boundary='symm')
        frames_summed_Ix_squared = frames_summed_Ix * frames_summed_Ix
        frames_summed_Iy_squared = frames_summed_Iy * frames_summed_Iy
        frames_summed_Ix_Iy = frames_summed_Ix * frames_summed_Iy


        #at the end of the actual calc,
        # we should have a (numFrames-1)xmaxFeaturesx2 vector with u v values for each feature in each frame

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
        for frame in current_frames:
            out.write(frame)

        currentFrame += numFrames
        current_frames, __ = loadVideo(filename, currentFrame+numFrames, numFrames)
        numFrames = current_frames.shape[0]

    #setup output video
    #easy fps = 29.97
    #med fps = 30.01
    #hard fps = 30.33
    # imageio.mimsave(filename+"_output.mp4", output_frames, format="MP4", fps="29")
    out.release()
    cv2.destroyAllWindows()

    return 0