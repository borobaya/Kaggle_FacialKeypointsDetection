# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:38:21 2015

@author: Muhammed Onu Miah

Experimenting on detecting facial features

This file finds the computed coordinates of the trained facial features

"""

# Import packages
import numpy as np
import pandas as pd
# Other
import os
import gc; gc.enable()
import time

# --------------------------------- Parameters --------------------------------

w = None
train_ratio = None
train_max = None
voice_enabled = None

def setParams(params):
    global w, train_ratio, train_max, voice_enabled
    w = params['w']
    train_ratio = params['train_ratio']
    train_max = params['train_max']
    voice_enabled = params['voice_enabled']

def clearParams():
    global w, train_ratio, train_max, voice_enabled
    w = None
    train_ratio = None
    train_max = None
    voice_enabled = None

# ---------------------------------- Data I/O ---------------------------------

windowTrainXcoords = None
windowTestXcoords = None
windowTrainZ = None
windowTestZ = None
faceTrainZ = None
faceTestZ = None

def loadTrain():
    global windowTrainXcoords, windowTrainZ
    windowTrainXcoords = np.load('Cache/WindowTrainXcoords.npy')
    windowTrainZ = np.load('Cache/WindowTrainZ.npy')

def loadTest():
    global windowTestXcoords, windowTestZ
    windowTestXcoords = np.load('Cache/WindowTestXcoords.npy')
    windowTestZ = np.load('Cache/WindowTestZ.npy')

def saveTrain():
    np.save("Cache/FaceTrainZ.npy", faceTrainZ)

def saveTest():
    np.save("Cache/FaceTestZ.npy", faceTestZ)

def clearTrain():
    global windowTrainXcoords, windowTrainZ, faceTrainZ
    windowTrainXcoords = None
    windowTrainZ = None
    faceTrainZ = None

def clearTest():
    global windowTestXcoords, windowTestZ, faceTestZ
    windowTestXcoords = None
    windowTestZ = None
    faceTestZ = None

# ------------------------------- External Code -------------------------------

# https://github.com/joferkington/oost_paper_code/blob/master/utilities.py
# http://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    if med_abs_deviation==0:
        return diff>0

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

# ---------------------------------- Methods ----------------------------------

def findFaceTrainCoords():
    global faceTrainZ
    indexes = np.unique(windowTrainXcoords[:,0])
    faceTrainZ = np.zeros([indexes.size, 30])
    
    for i in indexes:
        imageEntries = windowTrainXcoords[:,0]==i
        imageCoords = windowTrainXcoords[imageEntries,1:]
        
        for j in xrange(15):
            matches = windowTrainZ[imageEntries, j]
            matchCoords = imageCoords[matches]
            
            # Left side of face
            if j in [0,2,3,6,7]:
                leftMatched = matchCoords[:,0]>w/2
                leftCoords = matchCoords[leftMatched,:]
                if sum(leftMatched)>0:
                    matchCoords = leftCoords

            # Right side of face
            if j in [1,4,5,8,9]:
                rightMatched = matchCoords[:,0]<w/2
                rightCoords = matchCoords[rightMatched,:]
                if sum(rightMatched)>0:
                    matchCoords = rightCoords
            
            if matchCoords.size:
                # Remove obvious outliers
                outliers = is_outlier(matchCoords)
                if sum(outliers)>0 and sum(outliers)<matchCoords.shape[0]:
                    matchCoords = matchCoords[~outliers,:]
                
                coords = matchCoords.mean(0)
                faceTrainZ[i, j*2:j*2+2] = coords

def findFaceTestCoords():
    global faceTestZ
    indexes = np.unique(windowTestXcoords[:,0])
    faceTestZ = np.zeros([indexes.size, 30])
    
    for i in indexes:
        imageEntries = windowTestXcoords[:,0]==i
        imageCoords = windowTestXcoords[imageEntries,1:]
        
        for j in xrange(15):
            matches = windowTestZ[imageEntries, j]
            matchCoords = imageCoords[matches]
            
            # Left side of face
            if j in [0,2,3,6,7]:
                leftMatched = matchCoords[:,0]>w/2
                leftCoords = matchCoords[leftMatched,:]
                if sum(leftMatched)>0:
                    matchCoords = leftCoords

            # Right side of face
            if j in [1,4,5,8,9]:
                rightMatched = matchCoords[:,0]<w/2
                rightCoords = matchCoords[rightMatched,:]
                if sum(rightMatched)>0:
                    matchCoords = rightCoords
            
            if matchCoords.size:
                # Remove obvious outliers
                outliers = is_outlier(matchCoords)
                if sum(outliers)>0 and sum(outliers)<matchCoords.shape[0]:
                    matchCoords = matchCoords[~outliers,:]
                
                coords = matchCoords.mean(0)
                faceTestZ[i, j*2:j*2+2] = coords

def computeErr():
    faceTrainY = pd.DataFrame.from_csv('Cache/FaceTrainY.csv')
    faceTrainErr = faceTrainY-faceTrainZ
    errDist = np.zeros([faceTrainErr.shape[0], 15], dtype=np.float)
    for i in xrange(faceTrainErr.shape[0]):
        for j in xrange(15):
            xy = [ faceTrainErr.values[i,j*2], faceTrainErr.values[i,j*2+1] ]
            dist = np.linalg.norm(xy)
            errDist[i,j] = dist
    np.savetxt("Cache/FaceTrainZerrDist.csv", errDist, fmt="%.1f")
    return errDist

def computeAvgErr(errDist):
    # Calculate Error
    n = int(errDist.shape[0] * train_ratio)
    if n>train_max:
        n = train_max
    errDistTrain = errDist[:n,:]
    errDistTest = errDist[n:,:]
    avgErrDistTrain = errDistTrain[~np.isnan(errDistTrain)].mean()
    avgErrDistTest = errDistTest[~np.isnan(errDistTest)].mean()
    print "Average Train Error:", avgErrDistTrain
    print "Average Test Error:", avgErrDistTest
    # Save
    f = open('Cache/avgErr.txt', 'wb')
    f.write("Average Train Error: "+`avgErrDistTrain`+"\n")
    f.write("Average Test Error: "+`avgErrDistTest`+"\n")
    f.close()
    return avgErrDistTest

# -----------------------------------------------------------------------------

def run(params):
    start = time.time()
    setParams(params)
    # -------------------------------------------------------------------------
    print "--- Feature Coordinates ---"
    
    print "Finding matched face coordinates..."
    
    print "Training set..."
    loadTrain()
    findFaceTrainCoords()
    errDist = computeErr()
    avgErrDistTest = computeAvgErr(errDist)
    saveTrain()
    clearTrain()
    
    print "Test set..."
    loadTest()
    findFaceTestCoords()    
    saveTest()
    clearTest()
    
    # ------------------------------- Clean Up --------------------------------
    
    m, s = divmod((time.time() - start), 60)
    print "Time taken to run:", int(m), "minutes", round(s,3), "seconds"
    
    # Sound when completed
    if voice_enabled:
        os.system('say "Facial coordinates computed"')
        os.system('say "Average deviation of '+`round(avgErrDistTest,1)`+' pixels"')
    clearParams()
    gc.collect() # is this doing anything?

if __name__ == '__main__':
    run({'w':96,'train_ratio': 0.666,'train_max': 1000,'voice_enabled':False})
