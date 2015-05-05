# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:22:37 2015

@author: Muhammed Onu Miah

Experimenting on detecting facial features

This file displays sliding windows of the face images

"""

# Import packages
import numpy as np
import pandas as pd
# OpenCV
import cv2
# Other
import gc; gc.enable()
import random

# Clear anything from previous runs of the script
cv2.destroyAllWindows()

# Paths
path_root = "/Users/mdmiah/Desktop/Kaggle/Facial Keypoints Detection/"
path_train = path_root+"Data/training.csv"
path_test = path_root+"Data/test.csv"

print "The path is "+path_train

# Load data using Pandas (returns DataFrame)
train = pd.read_csv(path_train, nrows=100)
test  = pd.read_csv(path_test,  nrows=2)
labelsTrain = train.drop(labels='Image', axis=1)
print "Data has been loaded from files using Pandas"

# Parameters
trainSize = train.shape[0]
testSize = test.shape[0]
w = 96; # dimensions of the image

# Extract image data
imgTrain = np.zeros((trainSize, w*w), dtype=np.uint8)
temp = train.get('Image')
for i in xrange(0,trainSize):
    imgTrain[i,:] = np.fromstring(temp[i], dtype=np.uint8, sep=" ")

imgTest = np.zeros((testSize, w*w), dtype=np.uint8)
temp = test.get('Image')
for i in xrange(0,testSize):
    imgTest[i,:] = np.fromstring(temp[i], dtype=np.uint8, sep=" ")

# NOTE
# In labelsTrain,
# some of the target keypoint positions are misssing
# (encoded as missing entries in the csv,
# i.e., with nothing between two commas).


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

PCAmean = None
PCAvectors = None
def findPCANoise(imgSet):
    global PCAmean,PCAvectors
    #PCAmean,PCAvectors = cv2.PCAComputeVar(imgTrain, 0.999)
    PCAmean,PCAvectors = cv2.PCACompute(imgTrain, maxComponents=100)

def removePCANoise(image):
    image = image.reshape(image.size, 1)
    projection = cv2.PCAProject(image.transpose(), PCAmean, PCAvectors)
    image2 = cv2.PCABackProject(projection, PCAmean, PCAvectors)
    image2 = image2.astype(np.uint8)
    return image2

def preprocess(image):
    # Image enhancement
    #image = removePCANoise(image)
    image = enhance(image)
    image = histStretch(image)
    
    # Edge detection
    edges = edgeDet(image)
    # Gaussian Blur   
    edges = cv2.GaussianBlur(edges, ksize=(3,3), sigmaX=0)
    edges = histStretch(edges)
    
    # Resize images
    image = image.reshape(w,w)
    edges = edges.reshape(w,w)
    image = cv2.resize(image, (image.shape[0]/3,image.shape[0]/3))
    edges = cv2.resize(edges, (edges.shape[0]/3,edges.shape[0]/3))
    
    return image, edges

def processSlidingWindow(window):
    # Show image
    window = cv2.resize(window, (0,0), fx=40, fy=40)
    cv2.imshow("image", window)
    cv2.waitKey(1)

# ----------------------------------        ----------------------------------

#findPCANoise(imgTrain)

# Load an image and show it
print "Displaying a few images..."
for i in random.sample(range(0,trainSize), 1):
    image = imgTrain[i]
    image, edges = preprocess(image)
    
    # Sliding window
    img = image
    windowSize = 5
    for j in xrange(0, img.shape[0]-windowSize):
        for k in xrange(0, img.shape[1]-windowSize):
            window = img[j:j+windowSize, k:k+windowSize]
            if sum(sum(window!=0))==0:
                continue
            processSlidingWindow(window)

cv2.destroyAllWindows()

