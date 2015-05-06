# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:22:37 2015

@author: Muhammed Onu Miah

Experimenting on detecting facial features

This file trains on the principal components of sliding windows

"""

# Import packages
import numpy as np
import pandas as pd
# Other
import os
import gc; gc.enable()
import time

# --------------------------------- Parameters --------------------------------

train_ratio = None
train_max = None
voice_enabled = None

def setParams(params):
    global train_ratio, train_max, voice_enabled
    train_ratio = params['train_ratio']
    train_max = params['train_max']
    voice_enabled = params['voice_enabled']

def clearParams():
    global train_ratio, train_max, voice_enabled
    train_ratio = None
    train_max = None
    voice_enabled = None

# ---------------------------------- Data I/O ---------------------------------

windowTrainY = None
windowTrainXPCs = None
windowTestXPCs = None

metrics = None
windowTrainZ = None
windowTestZ = None

def loadTrain():
    global windowTrainXPCs, windowTrainY, metrics, windowTrainZ
    windowTrainXPCs = np.load('Cache/PCTrainX.npy')
    windowTrainY = np.load('Cache/WindowTrainY.npy')
    
    metrics = pd.DataFrame()
    windowTrainZ = np.zeros(windowTrainY.shape, dtype=np.bool)

def loadTest():
    global windowTestXPCs, windowTestZ
    windowTestXPCs = np.load('Cache/PCTestX.npy')
    
    windowTestZ = np.zeros([windowTestXPCs.shape[0],15], dtype=np.bool)

def clearTrain():
    global windowTrainY, windowTrainXPCs
    windowTrainY = None
    windowTrainXPCs = None

def clearTest():
    global windowTestXPCs
    windowTestXPCs = None

def saveTrainResults():
    metrics.to_csv('Cache/Metrics.csv')
    np.save('Cache/WindowTrainZ.npy', windowTrainZ)

def saveTestResults():
    np.save('Cache/WindowTestZ.npy', windowTestZ)

def clearTrainResults():
    global windowTrainZ
    windowTrainZ = None

def clearTestResults():
    global windowTestZ
    windowTestZ = None

# ---------------------------------- Methods ----------------------------------


Xtrain = None
Ytrain = None
Xtest = None
Ytest = None
def setupObservations():
    global Xtrain, Ytrain, Xtest, Ytest
    
    count = windowTrainXPCs.shape[0]
    n = int(count * train_ratio)
    if n>train_max:
        n = train_max
    print "Training on", round(100.0*n/count, 1),"% (", n, ") of the observations (", count, ")"
    
    Xtrain = windowTrainXPCs[:n,:]
    Ytrain = windowTrainY[:n,:]
    Xtest =  windowTrainXPCs[n:,:]
    Ytest =  windowTrainY[n:,:]

def clearObservations():
    global Xtrain, Ytrain, Xtest, Ytest
    Xtrain = None
    Ytrain = None
    Xtest = None
    Ytest = None

def test(X, y):
    Z = clf.predict(X)
    
    ct = pd.crosstab(y, Z, rownames=['actual'], colnames=['preds'])
    
    if ct.size!=4:
        print "WARNING: Everything predicted as", Z[0], ":("
        return Z, 0, 0, 0
    
    # Precision and recall
    tp = np.double(ct[True][True])
    tn = np.double(ct[False][False])
    fp = np.double(ct[True][False])
    fn = np.double(ct[False][True])
    precision = round(100* tp / (tp+fp) ,1)
    recall    = round(100* tp / (tp+fn) ,1)
    accuracy  = round(100* (tp+tn)/ct.sum().sum() ,1)
    
    return Z, precision, recall, accuracy

def learn(i):
    global clf, metrics
    y_train = Ytrain[:,i]
    y_test = Ytest[:,i]
    if sum(y_train==y_train[0]) == len(y_train):
        print "WARNING: All training values are the same :("
        metrics = metrics.append({}, ignore_index=True)
        return []
    print "Fitting model on variable", i, "..."
    clf.fit(Xtrain, y_train)
    Ztrain, train_precision, train_recall, train_accuracy = test(Xtrain, y_train)
    Ztest, test_precision, test_recall, test_accuracy = test(Xtest, y_test)
    # Add to results DataFrame
    m = {\
        'train_precision': train_precision, \
        'train_recall': train_recall, \
        'train_accuracy': train_accuracy, \
        'test_precision': test_precision, \
        'test_recall': test_recall, \
        'test_accuracy': test_accuracy}
    metrics = metrics.append(m, ignore_index=True)
    
    return np.concatenate([Ztrain, Ztest])

# -----------------------------------------------------------------------------

clf = None
def run(params, classifier):
    global windowTrainZ, windowTestZ, clf
    start = time.time()
    setParams(params)
    clf = classifier
    # -------------------------------------------------------------------------
    print "--- Machine Learning ---"
    
    # Load
    loadTrain()
    setupObservations()
    clearTrain()
    
    loadTest()
    
    # Process
    for i in xrange(15):
        temp = learn(i)
        if len(temp)>0:
            windowTrainZ[:,i] = temp
        
        temp = clf.predict(windowTestXPCs)
        if len(temp)>0:
            windowTestZ[:,i] = temp
    
    clearObservations()
    clearTest()
    
    # Results
    saveTrainResults()
    clearTrainResults()
    saveTestResults()
    clearTestResults()
    
    # ------------------------------- Clean Up --------------------------------
    clf = None
    
    m, s = divmod((time.time() - start), 60)
    print "Time taken to run:", m, "minutes", round(s,3), "seconds"
    
    # Sound when completed
    if voice_enabled:
        os.system('say "Model trained"')
    if metrics.sum().sum()==0:
        print "USELESS RESULTS"
    clearParams()
    gc.collect() # is this doing anything?

if __name__ == '__main__':
    run(params = {'train_ratio': 0.666,'train_max': 1000,'voice_enabled':False})
