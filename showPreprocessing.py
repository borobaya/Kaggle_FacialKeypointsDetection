# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:22:37 2015

@author: Muhammed Onu Miah

Experimenting on detecting facial features

This file shows preprocessed face images

"""

# Parameters
trainSize = 1000
testSize = 1
w = 96 # dimensions of the image
m = trainSize # Number of images to display
t = 10 # Number of seconds to show images for

# Import packages
import numpy as np
import pandas as pd
# OpenCV
import cv2
# Other
import gc; gc.enable()
import random
import time

start = time.time()

# Clear anything from previous runs of the script
cv2.destroyAllWindows()

# Paths
path_root = "/Users/mdmiah/Desktop/Kaggle/Facial Keypoints Detection/"
path_train = path_root+"Data/training.csv"
path_test = path_root+"Data/test.csv"

print "The path is "+path_train

# Load data using Pandas (returns DataFrame)
train = pd.read_csv(path_train, nrows=trainSize)
test  = pd.read_csv(path_test,  nrows=testSize)
labelsTrain = train.drop(labels='Image', axis=1)
print "Data has been loaded from files using Pandas"

# Update parameters in case it's different
trainSize = train.shape[0]
testSize = test.shape[0]
if m>trainSize:
    m = trainSize

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

def showImage(image):
    image = cv2.resize(image, (0,0), fx=12, fy=12)
    cv2.imshow("image", image)
    cv2.waitKey(1)

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
    #PCAmean,PCAvectors = cv2.PCAComputeVar(imgSet, 0.999)
    PCAmean,PCAvectors = cv2.PCACompute(imgSet, maxComponents=100)

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
    
#    # Edge detection
#    edges = edgeDet(image)
#    # Gaussian Blur
#    edges = cv2.GaussianBlur(edges, ksize=(3,3), sigmaX=0)
#    edges = histStretch(edges)
    
    # Resize images
    image = image.reshape(w,w)
#    edges = edges.reshape(w,w)
    image = cv2.resize(image, (image.shape[0]/3,image.shape[0]/3))
#    edges = cv2.resize(edges, (edges.shape[0]/3,edges.shape[0]/3))
    
#    return image, edges
    return image, image

# ----------------------------------        ----------------------------------

#findPCANoise(imgTrain)

# Load an image and show it
print "Displaying a few images..."
for i in random.sample(range(0,trainSize), m):
    image = imgTrain[i]
    image, edges = preprocess(image)
    
    # Show resulting image
    showImage(image)
    
    if time.time()-start>t:
        break;

cv2.destroyAllWindows()
