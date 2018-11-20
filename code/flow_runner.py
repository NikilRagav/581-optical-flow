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
from interp import interp2b

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def bgr2gray(bgr):
  return np.dot(bgr[...,:3], [0.114,0.587,0.299])


def flow_runner(numFrames, maxFeatures, qualityLevel, minDistance, windowsize, half_window, maxIterations, minAccuracy):
    #filepaths for input and output
    in_folder = "input_videos/"
    out_folder = "input_videos/"
    filename = "Easy.mp4"

    in_video_path = os.path.join(in_folder,filename)
    out_video_path = os.path.join(out_folder,filename)



    currentFrame = 0
    current_frames, totalFrames, H, W, fps = loadVideo(in_video_path, currentFrame, numFrames)
    #H, W come from the video file itself
    numFrames = current_frames.shape[0]
    minDistance /=H #keep same ratio of 8 pixel distance / 360p of resolution video regardless of resolution

    '''
    #open output video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (W,H))
    '''

    #let user draw bounding box on frame[0]
    # for now we'll just do the tracking on the whole frame


    while (currentFrame + numFrames - 1) < totalFrames:
        #GET GRAYSCALE
        frames_gray = bgr2gray(current_frames).astype(int)

        
        #GET DERIVATIVES

        #unfortunately convolve2d doesn't broadcast in 3D
        frames_Ix = np.zeros((numFrames-1, H, W))
        frames_Iy = np.zeros((numFrames-1, H, W))

        for i in range(numFrames-1):
            __, frames_Ix[i], frames_Iy[i], __ = findDerivatives(frames_gray[i])


        #GET FEATURES
        
        #goddamnit please vectorize APIs!!

        feature_list = np.zeros( (numFrames-1, maxFeatures, 2) )
        for i in range(numFrames-1):
            __, __, feature_list[i] = getFeatures(frames_gray[i], [(0,0,W,H)], maxFeatures, qualityLevel, minDistance)
        
        # for now, we'll just do the whole frame
        
        frames_coords = np.rollaxis( np.rollaxis( np.outer(np.ones((1,maxFeatures,1)), np.arange(numFrames-1) )[np.newaxis, :], 2, 0), 2, 1)
        feature_coords = np.rollaxis( np.rollaxis( np.outer(np.ones((1,(numFrames-1),1)), np.arange(maxFeatures) )[np.newaxis, :], 2, 0), 2, 0)
        feature_list = np.concatenate( (frames_coords, feature_coords, feature_list), axis=2)
        #now it is a (numFrames-1) x maxFeatures x 4
        #now it is w z y x coordinates (where w is frame number, z is feature number)
        feature_list = feature_list.astype(int)

        #include all the window points around the featurePoint
        #I want a (numFrames-1) x maxFeatures*windowsize*windowsize x 4
        
        window = np.zeros( (windowsize, 4) )
        window[...,-1] = np.arange(-half_window, half_window+1)
        #window is windowsize, 4

        window_perp = np.rollaxis(window[...,np.newaxis],-1,-2)
        window_perp = window_perp[...,[0,1,-1,-2]]
        #window_perp is windowsize, 1, 4 

        window_overall = (window + window_perp).reshape(-1,4)
        #window_overall becomes windowsize, windowsize, 4 -> windowsize*windowsize, 4

        #feature_windows stacks all the coords into a (numFrames-1)*maxFeatures, 4 array
        feature_windows = feature_list.reshape(-1,4)
        window_overall = np.rollaxis(window_overall[...,np.newaxis],-1,-2)
        #window overall becomes windowsize*windowsize, 1, 4

        #feature_windows becomes  windowsize*windowsize, (numFrames-1)*maxFeatures, 4
        #                      -> windowsize*windowsize*(numFrames-1)*maxFeatures, 4
        feature_windows = (feature_windows + window_overall).reshape(-1,4)
        feature_windows = feature_windows.reshape( (numFrames-1), -1, 4 )
        feature_windows = feature_windows.astype(int)


        #CALCULATE FLOW ACROSS ALL FRAMES

        #Calculate the sum of derivatives in the windows around each feature point
        summation_kernel = np.ones( (windowsize,windowsize) )
        

        frames_Ix_Ix_summed = np.zeros_like(frames_Ix)
        frames_Iy_Iy_summed = np.zeros_like(frames_Ix)
        frames_Ix_Iy_summed = np.zeros_like(frames_Ix)
        frames_Ix_It_summed = np.zeros_like(frames_Ix)
        frames_Iy_It_summed = np.zeros_like(frames_Ix)
        for i in range(numFrames-1):

            frames_Ix_Ix_summed[i] = signal.convolve2d(frames_Ix[i]*frames_Ix[i], summation_kernel, mode='same', boundary='symm')
            frames_Iy_Iy_summed[i] = signal.convolve2d(frames_Iy[i]*frames_Iy[i], summation_kernel, mode='same', boundary='symm')
            frames_Ix_Iy_summed[i] = signal.convolve2d(frames_Ix[i]*frames_Iy[i], summation_kernel, mode='same', boundary='symm')

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
            u <- x displacement
            v <- y displacement
        ]]

        =

        -[[
            sum( Ix(pointInWindow) * It(pointInWindow+u,v) )
            sum( Iy(pointInWindow) * It(pointInWindow+u,v) )
        ]]
        where It = frame_gray[i+1][pointInWindowCoords+uv] - frame_gray[i][pointInWindowCoords]
        '''
        #A matrix need only be created once
        #b matrix needs to be created every iteration of the solver with interpolation

        A = np.zeros(  ((numFrames-1),maxFeatures,2,2) )
        b = np.zeros(  ((numFrames-1),maxFeatures,2,1) )
        uv = np.zeros( ((numFrames-1),maxFeatures,2,1) )

        uv_window_points = ( uv[:,:,np.newaxis]+np.zeros((windowsize*windowsize, 2, 1)) ).reshape(numFrames-1, -1, 2,1)

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

        A[ :, :, 0,0 ] = frames_Ix_Ix_summed[feature_list[ ..., 0],
                                             feature_list[ ..., -2],
                                             feature_list[ ..., -1]]

        A[ :, :, 0,1 ] = frames_Ix_Iy_summed[feature_list[ ..., 0],
                                             feature_list[ ..., -2],
                                             feature_list[ ..., -1]]

        A[ :, :, 1,0 ] = frames_Ix_Iy_summed[feature_list[ ..., 0],
                                             feature_list[ ..., -2],
                                             feature_list[ ..., -1]]

        A[ :, :, 1,1 ] = frames_Iy_Iy_summed[feature_list[ ..., 0],
                                             feature_list[ ..., -2],
                                             feature_list[ ..., -1]]

        # I_window_points is (numFrames-1)*maxFeatures*windowsize*windowsize, 1
        # the pixel value at each image at each window point
        # includes only frames[:-1]
        I_window_points = frames_gray[(feature_windows[ ..., 0]).reshape(-1,1),
                                      (feature_windows[ ..., -2]).reshape(-1,1),
                                      (feature_windows[ ..., -1]).reshape(-1,1)]

        #run till we've reached maxIterations or till our answer is good enough
        for i in range(maxIterations):
            #calculate It (making sure to interpolate)
            
            #J is the pixelvalues of each window point at their projected locations in the next frame
            #skip the first frame values
            #  J is a (numFrames-1)*maxFeatures*windowsize*windowsize, 1
            J = interp2b(frames_gray[1:], (feature_windows[ ..., 0]).reshape(-1,1),
                                          (feature_windows[ ..., -2]+uv_window_points[...,1,0]).reshape(-1,1),
                                          (feature_windows[ ..., -1]+uv_window_points[...,0,0]).reshape(-1,1))
            
            It = J - I_window_points
            #It = It.reshape(numFrames-1,maxFeatures,-1,1)

            # place all the It values in frames so we can do convolution
            frames_It = np.zeros_like( frames_Ix )
            frames_It[(feature_windows[ ..., 0]).reshape(-1,1),
                      (feature_windows[ ..., -2]).reshape(-1,1),
                      (feature_windows[ ..., -1]).reshape(-1,1)] = It
            for i in range(numFrames-1):
                frames_Ix_It_summed[i] = signal.convolve2d(frames_Ix[i]*frames_It[i], summation_kernel, mode='same', boundary='symm')
                frames_Iy_It_summed[i] = signal.convolve2d(frames_Iy[i]*frames_It[i], summation_kernel, mode='same', boundary='symm')

            b[ :, :, 0,0 ] = frames_Ix_It_summed[feature_list[ ..., 0],
                                                 feature_list[ ..., -2],
                                                 feature_list[ ..., -1]]
            b[ :, :, 1,0 ] = frames_Iy_It_summed[feature_list[ ..., 0],
                                                 feature_list[ ..., -2],
                                                 feature_list[ ..., -1]]

            nu = np.linalg.solve(A,-b)
            uv += nu

            #propagate uv to all points in the window
            uv_perp = np.zeros( (numFrames-1, windowsize*windowsize, maxFeatures, 2, 1) )
            uv_big = uv[:, np.newaxis, ...] + uv_perp
            uv_window_points = ( np.rollaxis(uv_big, -4,-2) ).reshape(numFrames-1, -1, 2, 1)

            if np.max( np.linalg.norm(nu, axis=-2) ) <= minAccuracy:
                break

            #if all the error terms are less than our threshold, all of our uv vectors are close enough to correct

        uv = uv.reshape( (numFrames-1, maxFeatures, 2) )

        '''
        output of calc: (numFrames-1)xmaxFeaturesx2
        xdisplacement
        ydisplacement
        '''

        #at the end of the actual calc,
        # we should have a (numFrames-1)xmaxFeaturesx2 vector with u v values for each feature in each frame
        
        #we know the starting points for each feature. We know the displacement for each point. This is the ending points.
        #uv = np.round(uv).astype(int)
        #flip uv to vu so that it lines up with y,x in the old features list
        vu = np.zeros_like(feature_list, dtype=float)
        vu[...,[-2,-1]] = uv * 10 #make the vector arrows longer so we can see
        new_feature_list = (feature_list + vu).astype(int)

        #get the vectors to draw
        frames_out = np.copy(current_frames[:-1])

        for i in range(numFrames-1):
            for j in range(maxFeatures):
                cv2.line( frames_out[i], (feature_list[i,j,-1],feature_list[i,j,-2]), 
                                         (new_feature_list[i,j,-1],new_feature_list[i,j,-2]),
                                         (123,243,233), 1 )
        for i in range(numFrames-1):
            plt.imshow(frames_out[i,][...,[2,1,0]])

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


if __name__ == "__main__":
    #CONSTANTS
    numFrames = 10 #number of frames to calculate at a time

    #constants for corner detection
    maxFeatures = 50
    qualityLevel = .05
    minDistance = (8/360) #keep same ratio of 8 pixel distance for a 360p video regardless of resolution

    windowsize = 9
    half_window = np.floor(windowsize/2)

    maxIterations = 5 #stanford paper says 5 should be enough http://robots.stanford.edu/cs223b04/algo_tracking.pdf 
    minAccuracy = .01 #sandipan suggests this is enough

    flow_runner(numFrames, maxFeatures, qualityLevel, minDistance, windowsize, half_window, maxIterations, minAccuracy)