# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:33:00 2015

@author: Muhammed Onu Miah

Experimenting on detecting facial features

This file preprocesses the face images

"""

# Import packages
import numpy as np
import pandas as pd
# OpenCV
import cv2
# Other
import os
import gc; gc.enable()
import time

# --------------------------------- Parameters --------------------------------

partition = None
partition_sizes = None
partition_start_index = None
train_size = None
test_size = None
w = None
shrink_by = None
voice_enabled = None

def setParams(params):
    global partition, partition_sizes, partition_start_index, train_size,\
        test_size, w, shrink_by, voice_enabled
    
    partition = params['partition']
    partition_sizes = params['partition_sizes']
    temp = np.cumsum(partition_sizes)-partition_sizes
    partition_start_index = temp[partition]
    
    if params['train_size']:
        train_size = params['train_size']
    else:
        train_size = partition_sizes[partition]
    if params['test_size']:
        test_size = params['test_size']
    w = params['w']
    shrink_by = params['shrink_by']
    voice_enabled = params['voice_enabled']

def clearParams():
    global partition, partition_sizes, partition_start_index, train_size,\
        test_size, w, shrink_by, voice_enabled
    partition = None
    partition_sizes = None
    partition_start_index = None
    train_size = None
    test_size = None
    w = None
    shrink_by = None
    voice_enabled = None

# ---------------------------------- Data I/O ---------------------------------

imgTrain = None
imgTest = None
labelsTrain = None
faceTrainX = None
faceTrainY = None
faceTestX = None

def loadTrain(path):
    global imgTrain, labelsTrain, train_size, faceTrainX, faceTrainY
    # Load data using Pandas (returns DataFrame)
    train = pd.read_csv(path, nrows=train_size, skiprows=range(1,partition_start_index+1))
    labelsTrain = train.drop(labels='Image', axis=1)
    
    # Update params in case incorrect
    train_size = train.shape[0]
    
    # Extract image data
    imgTrain = np.zeros((train_size, w*w), dtype=np.uint8)
    temp = train.get('Image')
    for i in xrange(0,train_size):
        imgTrain[i,:] = np.fromstring(temp[i], dtype=np.uint8, sep=" ")
    
    faceTrainX = []
    faceTrainY = []

def loadTest(path):
    global imgTest, test_size, faceTestX
    # Load data using Pandas (returns DataFrame)
    test  = pd.read_csv(path, nrows=test_size)
    
    # Update params in case incorrect
    test_size = test.shape[0]
    
    # Extract image data
    imgTest = np.zeros((test_size, w*w), dtype=np.uint8)
    temp = test.get('Image')
    for i in xrange(0,test_size):
        imgTest[i,:] = np.fromstring(temp[i], dtype=np.uint8, sep=" ")
    
    faceTestX = []

def addToTrain(image):
    global faceTrainX
    faceTrainX.append(image.flatten())

def addToTest(image):
    global faceTestX
    faceTestX.append(image.flatten())

def saveTrain():
    np.save('Cache/FaceTrainX.npy', faceTrainX)
    faceTrainY.to_csv('Cache/FaceTrainY.csv')

def saveTest():
    np.save('Cache/FaceTestX.npy', faceTestX)

def clearTrain():
    global imgTrain, labelsTrain, faceTrainX, faceTrainY
    imgTrain = None
    labelsTrain = None
    faceTrainX = None
    faceTrainY = None

def clearTest():
    global imgTest, faceTestX
    imgTest = None
    faceTestX = None

# ---------------------------------- Methods ----------------------------------

def enhance(image):
    # Options:
    #  - Anisotropic diffusion
    #  - Bilateral filter
    #  - Median filter
    #  - Homomorphic Filtering (normalize brightness, increase contrast)
    # http://www.123seminarsonly.com/Seminar-Reports/029/42184313-10-1-1-100-81.pdf
    image = cv2.adaptiveBilateralFilter(image,ksize=(21,21),sigmaSpace=19)
    return image

def histStretch(image):
    # This produces better results in poor lighting
    # But is nonlinear (distorts image) &
    # probably means latter code is not needed
    #image = cv2.equalizeHist(image)
    # Modified histogram stretching
    # Put 5th and 95th percentile as min and max
    percentiles = np.percentile(image, [5,95])
    image[image<percentiles[0]] = percentiles[0]
    image[image>percentiles[1]] = percentiles[1]
    image = cv2.normalize(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image

def edgeDet(image):
    image = image.reshape(w,w)
    otsu_thres_val = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    # http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html
    edges = cv2.Canny(image, otsu_thres_val*0.5, otsu_thres_val*1)
    return edges

def preprocess(image):
    # Image enhancement
    image = enhance(image)
    image = histStretch(image)
    
#    # Edge detection
#    edges = edgeDet(image)
#    # Gaussian Blur   
#    edges = cv2.GaussianBlur(edges, ksize=(3,3), sigmaX=0)
#    edges = histStretch(edges)
    
    # Resize images
    image = image.reshape(w,w)
#    edges = edges.reshape(w,w)
    image = cv2.resize(image, (image.shape[0]/shrink_by,image.shape[0]/shrink_by))
#    edges = cv2.resize(edges, (edges.shape[0]/shrink_by,edges.shape[0]/shrink_by))
    
#    return image, edges
    return image, image

def createFaceTrainingSet(imgSet):
    global faceTrainX, faceTrainY
    for i in xrange(0,imgSet.shape[0]):
        image = imgSet[i]
        image, edges = preprocess(image)
        
        addToTrain(image)
        
    # Convert training set to array
    faceTrainX = np.asarray(faceTrainX, dtype=np.float)
    faceTrainX /= 255.0
    
    # Save labels    
    faceTrainY = labelsTrain[0:imgSet.shape[0]]

def createFaceTestingSet(imgSet):
    global faceTestX
    for i in xrange(0,imgSet.shape[0]):
        image = imgSet[i]
        image, edges = preprocess(image)
        
        addToTest(image)
        
    # Convert testing set to array
    faceTestX = np.asarray(faceTestX, dtype=np.float)
    faceTestX /= 255.0

# -----------------------------------------------------------------------------

def run(params):
    start = time.time()
    setParams(params)
    # -------------------------------------------------------------------------
    print "--- Preprocessing ---"
    
    print "Training set..."
    loadTrain("Data/training.csv")
    createFaceTrainingSet(imgTrain)
    saveTrain()
    clearTrain()
    
    print "Test set..."
    loadTest("Data/test.csv")
    createFaceTestingSet(imgTest)
    saveTest()
    clearTest()
    
    # ------------------------------- Clean Up --------------------------------
    
    m, s = divmod((time.time() - start), 60)
    print "Time taken to run:", int(m), "minutes", round(s,3), "seconds"
    
    # Sound when completed
    if voice_enabled:
        os.system('say "Faces pre processed"')
    clearParams()
    gc.collect() # is this doing anything?

if __name__ == '__main__':
    run({'partition':2,'partition_sizes':[10,10,10,10,10,10,10],
         'train_size':None,'test_size':10,'w':96,'window_size':5,'shrink_by':3,
         'voice_enabled':False})
