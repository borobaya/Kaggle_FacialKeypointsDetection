# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:22:37 2015

@author: Muhammed Onu Miah

Experimenting on detecting facial features

This file shows preprocessed face images

"""

# Parameters
trainSize = 1000 # Number of images to load
testSize = 1
w = 96; # dimensions of the image
m = trainSize # Number of images to show
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

def processImage(image):
    d = 3
    image = image.reshape(w,w)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # --- Eyes ---
    # left_eye_center
    x = labelsTrain.left_eye_center_x[i].astype(np.uint32)
    y = labelsTrain.left_eye_center_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,255,0))
    # right_eye_center
    x = labelsTrain.right_eye_center_x[i].astype(np.uint32)
    y = labelsTrain.right_eye_center_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,0,255))
    # left_eye_inner_corner
    x = labelsTrain.left_eye_inner_corner_x[i].astype(np.uint32)
    y = labelsTrain.left_eye_inner_corner_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,255,0))
    # left_eye_outer_corner
    x = labelsTrain.left_eye_outer_corner_x[i].astype(np.uint32)
    y = labelsTrain.left_eye_outer_corner_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,255,0))
    # right_eye_inner_corner
    x = labelsTrain.right_eye_inner_corner_x[i].astype(np.uint32)
    y = labelsTrain.right_eye_inner_corner_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,0,255))
    # right_eye_outer_corner
    x = labelsTrain.right_eye_outer_corner_x[i].astype(np.uint32)
    y = labelsTrain.right_eye_outer_corner_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,0,255))
    
    # --- Eyebrows ---
    # left_eyebrow_inner_end
    x = labelsTrain.left_eyebrow_inner_end_x[i].astype(np.uint32)
    y = labelsTrain.left_eyebrow_inner_end_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,255,0))
    # left_eyebrow_outer_end
    x = labelsTrain.left_eyebrow_outer_end_x[i].astype(np.uint32)
    y = labelsTrain.left_eyebrow_outer_end_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,255,0))
    # right_eyebrow_inner_end
    x = labelsTrain.right_eyebrow_inner_end_x[i].astype(np.uint32)
    y = labelsTrain.right_eyebrow_inner_end_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,0,255))
    # right_eyebrow_outer_end
    x = labelsTrain.right_eyebrow_outer_end_x[i].astype(np.uint32)
    y = labelsTrain.right_eyebrow_outer_end_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,0,255))
    
    # nose_tip
    x = labelsTrain.nose_tip_x[i].astype(np.uint32)
    y = labelsTrain.nose_tip_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,255,0))
    
    # --- Mouth ---
    # mouth_left_corner
    x = labelsTrain.mouth_left_corner_x[i].astype(np.uint32)
    y = labelsTrain.mouth_left_corner_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,255,0))
    # mouth_right_corner
    x = labelsTrain.mouth_right_corner_x[i].astype(np.uint32)
    y = labelsTrain.mouth_right_corner_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,255,0))
    # mouth_center_top_lip
    x = labelsTrain.mouth_center_top_lip_x[i].astype(np.uint32)
    y = labelsTrain.mouth_center_top_lip_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,255,0))
    # mouth_center_bottom_lip
    x = labelsTrain.mouth_center_bottom_lip_x[i].astype(np.uint32)
    y = labelsTrain.mouth_center_bottom_lip_y[i].astype(np.uint32)
    cv2.rectangle(image, (x-d,y-d), (x+d,y+d), (0,255,0))
    
    image = cv2.resize(image, (0,0), fx=4, fy=4)
    return image

def showImage(image):
    image = processImage(image)
    cv2.imshow("image", image)
    cv2.waitKey(100)

# ----------------------------------        ----------------------------------

# Load an image and show it
print "Displaying a few images..."
for i in random.sample(range(0,trainSize), m):
    showImage(imgTrain[i])
    if time.time()-start>t:
        break;

cv2.destroyAllWindows()
