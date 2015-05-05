# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:51:13 2015

@author: Muhammed Onu Miah

Experimenting on detecting facial features

This file creates a training set of sliding windows from face images

"""

# Import packages
import numpy as np
import pandas as pd
# Other
import os
import gc; gc.enable()
import time
import math

# --------------------------------- Parameters --------------------------------

w = None
window_size = None
shrink_by = None
pos_tolerance = None
window_radius = None
voice_enabled = None

def setParams(params):
    global w, window_size, shrink_by, pos_tolerance, window_radius, voice_enabled
    w = params['w']
    window_size = params['window_size']
    shrink_by = params['shrink_by']
    pos_tolerance = params['pos_tolerance']
    window_radius = (window_size-1)*0.5
    voice_enabled = params['voice_enabled']

def clearParams():
    global w, window_size, shrink_by, pos_tolerance, window_radius, voice_enabled
    w = None
    window_size = None
    shrink_by = None
    pos_tolerance = None
    window_radius = None
    voice_enabled = None

# ---------------------------------- Data I/O ---------------------------------

faceTrainX = None
faceTrainY = None
faceTestX = None
windowTrainX = None
windowTrainXcoords = None
windowTrainY = None
windowTestX = None
windowTestXcoords = None

def loadTrain():
    global faceTrainX, faceTrainY, windowTrainX, windowTrainXcoords, windowTrainY
    faceTrainX = np.load('Cache/FaceTrainX.npy')
    faceTrainY = pd.DataFrame.from_csv('Cache/FaceTrainY.csv')
    
    windowTrainX = []
    windowTrainXcoords = []
    windowTrainY = []

def loadTest():
    global faceTestX, windowTestX, windowTestXcoords
    faceTestX = np.load('Cache/FaceTestX.npy')
    
    windowTestX = []
    windowTestXcoords = []

def saveTrain():
    np.save('Cache/WindowTrainX.npy', windowTrainX)
    np.save('Cache/WindowTrainXcoords.npy', windowTrainXcoords)
    np.save('Cache/WindowTrainY.npy', windowTrainY)

def saveTest():
    np.save('Cache/WindowTestX.npy', windowTestX)
    np.save('Cache/WindowTestXcoords.npy', windowTestXcoords)

def clearTrain():
    global faceTrainX, faceTrainY, windowTrainX, windowTrainXcoords, windowTrainY
    faceTrainX = None
    faceTrainY = None
    windowTrainX = None
    windowTrainXcoords = None
    windowTrainY = None

def clearTest():
    global faceTestX, windowTestX, windowTestXcoords
    faceTestX = None
    windowTestX = None
    windowTestXcoords = None

# ---------------------------------- Methods ----------------------------------

def addWindowToTraining(window, y, xpos, ypos):
    global windowTrainX, windowTrainY
    windowTrainX.append(window.flatten())
    xpos = xpos*shrink_by # Account for images being resized
    ypos = ypos*shrink_by # Account for images being resized
    tol = pos_tolerance*shrink_by
    # Figure out training labels
    labels = np.zeros((y.size/2), dtype=np.bool)
    if abs(xpos-y.left_eye_center_x)<=tol and abs(ypos-y.left_eye_center_y)<=tol:
        labels[0] = True
    if abs(xpos-y.right_eye_center_x)<=tol and abs(ypos-y.right_eye_center_y)<=tol:
        labels[1] = True
    if abs(xpos-y.left_eye_inner_corner_x)<=tol and abs(ypos-y.left_eye_inner_corner_y)<=tol:
        labels[2] = True
    if abs(xpos-y.left_eye_outer_corner_x)<=tol and abs(ypos-y.left_eye_outer_corner_y)<=tol:
        labels[3] = True
    if abs(xpos-y.right_eye_inner_corner_x)<=tol and abs(ypos-y.right_eye_inner_corner_y)<=tol:
        labels[4] = True
    if abs(xpos-y.right_eye_outer_corner_x)<=tol and abs(ypos-y.right_eye_outer_corner_y)<=tol:
        labels[5] = True
    if abs(xpos-y.left_eyebrow_inner_end_x)<=tol and abs(ypos-y.left_eyebrow_inner_end_y)<=tol:
        labels[6] = True
    if abs(xpos-y.left_eyebrow_outer_end_x)<=tol and abs(ypos-y.left_eyebrow_outer_end_y)<=tol:
        labels[7] = True
    if abs(xpos-y.right_eyebrow_inner_end_x)<=tol and abs(ypos-y.right_eyebrow_inner_end_y)<=tol:
        labels[8] = True
    if abs(xpos-y.right_eyebrow_outer_end_x)<=tol and abs(ypos-y.right_eyebrow_outer_end_y)<=tol:
        labels[9] = True
    if abs(xpos-y.nose_tip_x)<=tol and abs(ypos-y.nose_tip_y)<=tol:
        labels[10] = True
    if abs(xpos-y.mouth_left_corner_x)<=tol and abs(ypos-y.mouth_left_corner_y)<=tol:
        labels[11] = True
    if abs(xpos-y.mouth_right_corner_x)<=tol and abs(ypos-y.mouth_right_corner_y)<=tol:
        labels[12] = True
    if abs(xpos-y.mouth_center_top_lip_x)<=tol and abs(ypos-y.mouth_center_top_lip_y)<=tol:
        labels[13] = True
    if abs(xpos-y.mouth_center_bottom_lip_x)<=tol and abs(ypos-y.mouth_center_bottom_lip_y)<=tol:
        labels[14] = True
    windowTrainY.append(labels)

def addWindowToTesting(window):
    global windowTestX
    windowTestX.append(window.flatten())

def getImageCroppedCoords(y, shape):
    x1 = np.nanmin([ \
        y.left_eye_center_x, y.right_eye_center_x, \
        y.left_eye_inner_corner_x, y.left_eye_outer_corner_x, \
        y.right_eye_inner_corner_x, y.right_eye_outer_corner_x, \
        y.left_eyebrow_inner_end_x, y.left_eyebrow_outer_end_x, \
        y.right_eyebrow_inner_end_x, y.right_eyebrow_outer_end_x, \
        y.nose_tip_x, \
        y.mouth_left_corner_x, y.mouth_right_corner_x, \
        y.mouth_center_top_lip_x, y.mouth_center_bottom_lip_x ])
    x2 = np.nanmax([ \
        y.left_eye_center_x, y.right_eye_center_x, \
        y.left_eye_inner_corner_x, y.left_eye_outer_corner_x, \
        y.right_eye_inner_corner_x, y.right_eye_outer_corner_x, \
        y.left_eyebrow_inner_end_x, y.left_eyebrow_outer_end_x, \
        y.right_eyebrow_inner_end_x, y.right_eyebrow_outer_end_x, \
        y.nose_tip_x, \
        y.mouth_left_corner_x, y.mouth_right_corner_x, \
        y.mouth_center_top_lip_x, y.mouth_center_bottom_lip_x ])
    y1 = np.nanmin([ \
        y.left_eye_center_y, y.right_eye_center_y, \
        y.left_eye_inner_corner_y, y.left_eye_outer_corner_y, \
        y.right_eye_inner_corner_y, y.right_eye_outer_corner_y, \
        y.left_eyebrow_inner_end_y, y.left_eyebrow_outer_end_y, \
        y.right_eyebrow_inner_end_y, y.right_eyebrow_outer_end_y, \
        y.nose_tip_y, \
        y.mouth_left_corner_y, y.mouth_right_corner_y, \
        y.mouth_center_top_lip_y, y.mouth_center_bottom_lip_y ])
    y2 = np.nanmax([ \
        y.left_eye_center_y, y.right_eye_center_y, \
        y.left_eye_inner_corner_y, y.left_eye_outer_corner_y, \
        y.right_eye_inner_corner_y, y.right_eye_outer_corner_y, \
        y.left_eyebrow_inner_end_y, y.left_eyebrow_outer_end_y, \
        y.right_eyebrow_inner_end_y, y.right_eyebrow_outer_end_y, \
        y.nose_tip_y, \
        y.mouth_left_corner_y, y.mouth_right_corner_y, \
        y.mouth_center_top_lip_y, y.mouth_center_bottom_lip_y ])
    x1 = int(math.floor(x1/shrink_by - window_radius - pos_tolerance -1))
    x2 = int(math.ceil( x2/shrink_by + window_radius + pos_tolerance +1))
    y1 = int(math.floor(y1/shrink_by - window_radius - pos_tolerance -1))
    y2 = int(math.ceil( y2/shrink_by + window_radius + pos_tolerance +1))
    if x1<0:
        x1 = 0
    if y1<0:
        y1 = 0
    if x2>shape[0]:
        x2 = shape[0]
    if y2>shape[1]:
        y2 = shape[1]
    return x1, x2, y1, y2

def createWindowTrainingSet():
    global windowTrainX, windowTrainXcoords, windowTrainY
    for i in xrange(0,faceTrainX.shape[0]):
        image = faceTrainX[i]
        image = image.reshape(w/shrink_by, w/shrink_by)
        
        # Get label coordinates for image
        y = faceTrainY.irow(i)
        
        # Crop image
        x1 = 0
        y1 = 0
        x2 = image.shape[0]
        y2 = image.shape[1]
        #x1, x2, y1, y2 = getImageCroppedCoords(y, image.shape)
        
        # Sliding window
        img = image
        for j in xrange(x1, x2-window_size):
            for k in xrange(y1, y2-window_size):
                window = img[j:j+window_size, k:k+window_size]
                if sum(sum(window!=0))==0:
                    continue
                xpos = j+window_radius
                ypos = k+window_radius
                addWindowToTraining(window, y, xpos, ypos)
                windowTrainXcoords.append([i,xpos*shrink_by,ypos*shrink_by])
        
    # Convert training set to array
    windowTrainX = np.asarray(windowTrainX, dtype=np.float)
    windowTrainY = np.asarray(windowTrainY, dtype=np.bool)
    windowTrainXcoords = np.asarray(windowTrainXcoords, dtype=np.int)

def createWindowTestingSet():
    global windowTestX, windowTestXcoords
    for i in xrange(0,faceTestX.shape[0]):
        image = faceTestX[i]
        image = image.reshape(w/shrink_by, w/shrink_by)
        
        # Sliding window
        img = image
        for j in xrange(0, img.shape[0]-window_size):
            for k in xrange(0, img.shape[1]-window_size):
                window = img[j:j+window_size, k:k+window_size]
                if sum(sum(window!=0))==0:
                    continue
                xpos = j+window_radius
                ypos = k+window_radius
                addWindowToTesting(window)
                windowTestXcoords.append([i,xpos*shrink_by,ypos*shrink_by])
    
    # Convert testing set to array
    windowTestX = np.asarray(windowTestX, dtype=np.float)
    windowTestXcoords = np.asarray(windowTestXcoords, dtype=np.int)

# -----------------------------------------------------------------------------

def run(params):
    start = time.time()
    setParams(params)  
    # -------------------------------------------------------------------------
    print "--- Sliding Windows ---"
  
    print "Training set..."
    loadTrain()
    createWindowTrainingSet()
    print windowTrainX.shape[0], "training observations created"
    saveTrain()
    clearTrain()
    
    print "Test set..."    
    loadTest()
    createWindowTestingSet()
    saveTest()
    clearTest()
    
    # ------------------------------- Clean Up --------------------------------
    clearParams()
    
    m, s = divmod((time.time() - start), 60)
    print "Time taken to run:", int(m), "minutes", round(s,3), "seconds"
    
    # Sound when completed
    if voice_enabled:
        os.system('say "'+`windowTrainX.shape[0]`+' training observations created"')
    gc.collect() # is this doing anything?

if __name__ == '__main__':
    run({'w':96,'window_size':5,'shrink_by':3,'pos_tolerance':2.5,'voice_enabled':False})
