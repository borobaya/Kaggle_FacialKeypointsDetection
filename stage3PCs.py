# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:22:37 2015

@author: Muhammed Onu Miah

Experimenting on detecting facial features

This file extracts the principal components of the sliding windows

"""

# Import packages
import numpy as np
# OpenCV
import cv2
# Other
import os
import gc; gc.enable()
import time

# --------------------------------- Parameters --------------------------------

pc_num = None
pc_variance = None
voice_enabled = None

def setParams(params):
    global pc_num, pc_variance, voice_enabled
    if params['pc_num']:
        pc_num = params['pc_num']
    pc_variance = params['pc_variance']
    voice_enabled = params['voice_enabled']

def clearParams():
    global pc_num, pc_variance, voice_enabled
    pc_num = None
    pc_variance = None
    voice_enabled = None

# ---------------------------------- Data I/O ---------------------------------

windowTrainX = None
windowTestX = None

windowTrainXPCs = None
windowTestXPCs = None

def loadTrain():
    global windowTrainX, windowTrainXPCs
    windowTrainX = np.load('Cache/WindowTrainX.npy')
    windowTrainXPCs = []

def loadTest():
    global windowTestX, windowTestXPCs
    windowTestX = np.load('Cache/WindowTestX.npy')
    windowTestXPCs = []

def saveTrain():
    np.save('Cache/PCTrainX.npy', windowTrainXPCs)

def saveTest():
    np.save('Cache/PCTestX.npy', windowTestXPCs)

def clearTrain():
    global windowTrainX, windowTrainXPCs
    windowTrainX = None
    windowTrainXPCs = None

def clearTest():
    global windowTestX, windowTestXPCs
    windowTestX = None
    windowTestXPCs = None

# ---------------------------------- Methods ----------------------------------

PCAmean = None
PCAvectors = None
def runPCA(imgSet):
    global PCAmean,PCAvectors
    if pc_num>0:
        PCAmean,PCAvectors = cv2.PCACompute(imgSet, maxComponents=pc_num)
    else:
        PCAmean,PCAvectors = cv2.PCAComputeVar(imgSet, retainedVariance=pc_variance)

def getPCs(image):
    image = image.reshape(image.size, 1)
    projection = cv2.PCAProject(image.transpose(), PCAmean, PCAvectors)
    return projection

def clearPCs():
    global PCAmean,PCAvectors    
    PCAmean = None
    PCAvectors = None

def createWindowTrainXPCs(imgSet):
    global windowTrainXPCs
    runPCA(imgSet)
    
    for i in xrange(0,imgSet.shape[0]):
        image = imgSet[i,:]
        image2 = getPCs(image)
        windowTrainXPCs.append(image2.flatten())
    
    # Convert training PCs set to array
    windowTrainXPCs = np.asarray(windowTrainXPCs, dtype=np.float)

def createWindowTestXPCs(imgSet):
    global windowTestXPCs
    for i in xrange(0,imgSet.shape[0]):
        image = imgSet[i,:]
        image2 = getPCs(image)
        windowTestXPCs.append(image2.flatten())
    
    # Convert testing PCs set to array
    windowTestXPCs = np.asarray(windowTestXPCs, dtype=np.float)

# -----------------------------------------------------------------------------

def run(params):
    start = time.time()
    setParams(params)
    # -------------------------------------------------------------------------
    print "--- Principal Components ---"
    
    print "Training set..."
    loadTrain()
    createWindowTrainXPCs(windowTrainX)
    print windowTrainXPCs.shape[1], "principal components computed"
    saveTrain()
    clearTrain()
    
    print "Test set..."
    loadTest()
    createWindowTestXPCs(windowTestX)
    saveTest()
    clearTest()
    
    # ------------------------------- Clean Up --------------------------------
    clearPCs()
    clearParams()
    
    m, s = divmod((time.time() - start), 60)
    print "Time taken to run:", m, "minutes", round(s,3), "seconds"
    
    # Sound when completed
    if voice_enabled:
        os.system('say "'+`windowTrainXPCs.shape[1]`+' principal components computed"')
    gc.collect() # is this doing anything?

if __name__ == '__main__':
    run({'pc_num':7,'pc_variance':0.9,'voice_enabled':False})
