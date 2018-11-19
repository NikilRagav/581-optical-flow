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

#CONSTANTS
numFrames = 10 #number of frames to calculate at a time

#constants for corner detection
maxFeatures = 50
qualityLevel = .05
minDistance = (8/360)*H #keep same ratio of 8 pixel distance for a 360p video regardless of resolution

windowsize = 9
half_window = np.floor(windowsize/2)

maxIterations = 5 #stanford paper says 5 should be enough http://robots.stanford.edu/cs223b04/algo_tracking.pdf 
minAccuracy = .01 #sandipan suggests this is enough


def flow_runner():
    #filepaths for input and output
    in_folder = "input_videos/"
    out_folder = "input_videos/"
    filename = "Easy.mp4"

    in_video_path = os.path.join(in_folder,filename)
    out_video_path = os.path.join(out_folder,filename)



    currentFrame = 0
    current_frames, totalFrames, fps = loadVideo(in_video_path, currentFrame, numFrames)
    #H, W come from the video file itself
    numFrames, H,W = current_frames.shape[0], current_frames.shape[1], current_frames.shape[2]

    #open output video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (W,H))

    #let user draw bounding box on frame[0]
    # for now we'll just do the tracking on the whole frame

    

    while (currentFrame + numFrames - 1) < totalFrames:
        #get grayscale - doing it in the get Features func instead since it was throwing errors
        frames_gray = bgr2gray(current_frames) #?

        
        #get derivatives

        #unfortunately convolve2d doesn't broadcast in 3D
        frames_Ix = np.zeros((numFrames-1, H, W))
        frames_Iy = np.zeros((numFrames-1, H, W))

        for i in range(numFrames-1):
            __, frames_Ix[i], frames_Iy[i], __ = findDerivatives(frames_gray[i])

        #get features
        #goddamnit please vectorize APIs!!

        feature_list = np.zeros( (numFrames-1, maxFeatures, 2) )
        for i in range(numFrames-1):
            __, __, feature_list[i] = getFeatures(frames_gray[i], (0,0,W,H), maxFeatures, qualityLevel, minDistance)
        # for now, we'll just do the whole frame
        zaxis = np.rollaxis( np.rollaxis( np.outer(np.ones((1,maxFeatures,1)), np.arange(numFrames-1) )[np.newaxis, :], 2, 0), 2, 1)
        feature_list = np.concatenate( (zaxis, feature_list), axis=2)
        #now it is z,x,y coordinates (where z is frame number)

        #do actual calc across all frames
        uv = np.zeros( (numFrames-1, maxFeatures, 2, 1))

        #calculate the sum of derivatives in the windows around each feature point
        summation_kernel = np.ones( (windowsize,windowsize) )
        
        #these are never used
        #frames_Ix_summed = np.zeros_like(frames_Ix)
        #frames_Iy_summed = np.zeros_like(frames_Ix)
        #frames_It_summed = np.zeros_like(frames_Ix)

        frames_Ix_Ix_summed = np.zeros_like(frames_Ix)
        frames_Iy_Iy_summed = np.zeros_like(frames_Ix)
        frames_Ix_Iy_summed = np.zeros_like(frames_Ix)
        frames_Ix_It_summed = np.zeros_like(frames_Ix)
        frames_Iy_It_summed = np.zeros_like(frames_Ix)
        for i in range(numFrames-1):
            #these aren't ever used
            #frames_Ix_summed[i] = signal.convolve2d(frames_Ix, summation_kernel, mode='same', boundary='symm')
            #frames_Iy_summed[i] = signal.convolve2d(frames_Iy, summation_kernel, mode='same', boundary='symm')
            #frames_It_summed[i] = signal.convolve2d(frames_It, summation_kernel, mode='same', boundary='symm')

            frames_Ix_Ix_summed[i] = signal.convolve2d(frames_Ix*frames_Ix, summation_kernel, mode='same', boundary='symm')
            frames_Iy_Iy_summed[i] = signal.convolve2d(frames_Iy*frames_Iy, summation_kernel, mode='same', boundary='symm')
            frames_Ix_Iy_summed[i] = signal.convolve2d(frames_Ix*frames_Iy, summation_kernel, mode='same', boundary='symm')

        #create the A matrix and b vector at each feature point
        '''
        for each feature point in each frame 
        Take the all the points in the window around the feature point
        Window:
        [[
            pt  pt     pt    pt  pt
            pt  pt     pt    pt  pt
            pt  pt  feature  pt  pt
            pt  pt     pt    pt  pt
            pt  pt     pt    pt  pt
        ]]


        A * uv = -b
        [[
            sum( Ix(pointInWindow) * Ix(pointInWindow) )      sum( Ix(pointInWindow) * Iy(pointInWindow) )
            sum( Ix(pointInWindow) * Iy(pointInWindow) )      sum( Iy(pointInWindow) * Iy(pointInWindow) )
        ]]

        *
        [[
            u
            v
        ]]

        =

        -[[
            sum( Ix(pointInWindow) * It(pointInWindow+u) )
            sum( Iy(pointInWindow) * It(pointInWindow+v) )
        ]]
        where It = frame_gray[i+1][pointInWindowCoords+uv] - frame_gray[i][pointInWindowCoords]
        '''
        #A matrix need only be created once
        #b matrix needs to be created every iteration of the solver with interpolation

        A = np.zeros( (numFrames-1),maxFeatures,2,2 )
        b = np.zeros( ((numFrames-1),maxFeatures,2,1) )
        uv = np.zeros( ((numFrames-1),maxFeatures,2,1) )

        #I want every pair of frameNum and featureNum to index A,b,uv correctly
        '''
        frameNum    featureNum
        0           0
        0           1
        ...         ...
        0           maxFeatures-1
        1           0
        1           1
        ...         ...
        1           maxFeatures-1
        ...         ...
        numFrames-1 maxFeatures-1
        '''

        '''

        This was unnecessary
        frameFeatPairs = np.concatenate( ( (np.arange((numFrames-1)*maxFeatures)/(maxFeatures)).astype(int).reshape(-1,1),
                                           (np.arange((numFrames-1)*maxFeatures)%(maxFeatures)).astype(int).reshape(-1,1)
                                         ), axis=1 )
        '''

        A[ :, :, 0,0 ] = frames_Ix_Ix_summed[feature_list[0], feature_list[1], feature_list[2]]
        A[ :, :, 0,1 ] = frames_Ix_Iy_summed[feature_list[0], feature_list[1], feature_list[2]]
        A[ :, :, 1,0 ] = frames_Ix_Iy_summed[feature_list[0], feature_list[1], feature_list[2]]
        A[ :, :, 1,1 ] = frames_Iy_Iy_summed[feature_list[0], feature_list[1], feature_list[2]]
        

        #run till we've reached maxIterations or till our answer is good enough
        for i in range(maxIterations):
            if np.min( abs(np.linalg.eigvals(A)) ) >= minAccuracy:
                #keep calculating uv

                #calculate It (making sure to interpolate)



            #if all the error terms are less than our threshold, all of our uv vectors are close enough to correct

        uv.reshape( (numFrames-1, maxFeatures, 2) )

        '''
        output of calc: (numFrames-1)xmaxFeaturesx2
        xdisplacement
        ydisplacement
        '''

        #at the end of the actual calc,
        # we should have a (numFrames-1)xmaxFeaturesx2 vector with u v values for each feature in each frame
        for i in range(numFrames-1):
            for j in range(maxFeatures):
                u = displacement[i][j][0]
                v = displacement[i][j][1]
                x = feature_list()

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

if __name__ == '__main__':
    flow_runner()