# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:59:58 2015

@author: Muhammed Onu Miah

Experimenting on detecting facial features

This file runs the entire pipeline

"""

import stage1preprocess
import stage2windows
import stage3PCs
import stage4learn
import stage5coords
import stage6submission
# Machine Learning algorithm
from sklearn import neighbors
# Other
import os
import json
import shutil
import time

params = {
          'partition': 0,
          'partition_sizes': [10, 10, 10, 10, 10],
          'train_size': None, # overrides value in partition_sizes
          'test_size': 1,
          'w': 96, # dimensions of the image
          'window_size': 15, # should be an odd number; 15
          'shrink_by': 1,
          'pos_tolerance': 11.5,
          'pc_num': 7, # Overrides pc_variance; 7
          'pc_variance': 0.9,
          'train_ratio': 0.6666, # Fraction of observations to train on
          'train_max': 2000000, # Maximum number of observations to train on 
          'voice_enabled': False
         }

def run(params):
    # Save params
    pfile = open('Cache/params.txt', 'wb')
    pfile.write(json.dumps(params, indent=4))
    pfile.close()
    
    # Run pipeline
    stage1preprocess.run(params)
    stage2windows.run(params)
    stage3PCs.run(params)
    stage4learn.run(params, neighbors.KNeighborsClassifier(3))
    stage5coords.run(params)
    stage6submission.run(params)
    
    # Backup key files
    dirName = 'Backup/'+`time.time()`+'/'
    os.makedirs(dirName)
    shutil.copyfile('Cache/params.txt', dirName+'params.txt')
    shutil.copyfile('Cache/avgErr.txt', dirName+'avgErr.txt')
    shutil.copyfile('Cache/submission.csv', dirName+'submission.csv')
    shutil.copyfile('Cache/Metrics.csv', dirName+'Metrics.csv')
    
    # Finish
    os.system('say "Stages completed"')

def runAll(params):
    for i in xrange(len(params["partition_sizes"])):
        print "######### Partition", i, "#########"
        params["partition"] = i
        run(params)
    os.system('say "Done"')

if __name__ == '__main__':
    runAll(params)
