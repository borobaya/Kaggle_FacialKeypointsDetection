# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:44:10 2015

@author: Muhammed Onu Miah

Experimenting on detecting facial features

This file computes the average of coordinates in multiple submission files

"""

# Import packages
import numpy as np
import pandas as pd
# Other
import os
import gc; gc.enable()
import time

# --------------------------------- Parameters --------------------------------

voice_enabled = None

def setParams(params):
    global voice_enabled
    voice_enabled = params['voice_enabled']

def clearParams():
    global voice_enabled
    voice_enabled = None

# ---------------------------------- Data I/O ---------------------------------

submissions = None
avg = None
def load(folders):
    global submissions
    submissions = []
    for folder in folders:
        submissions.append(pd.DataFrame.from_csv(folder+'/submission.csv'))

def save(folder):
    avg.to_csv(folder+'/avgSubmission.csv')

def clear():
    global submissions, avg
    submissions = None
    avg = None

# ---------------------------------- Methods ----------------------------------

def computeAvg():
    global avg
    avg = pd.concat(submissions, axis=1)
    avg = avg.mean(axis=1)

# -----------------------------------------------------------------------------

def run(params, folders=["Cache","Cache","Cache","Cache"]):
    start = time.time()
    setParams(params)
    # -------------------------------------------------------------------------
    print "--- Average of Submission Files ---"
    
    load(folders)
    computeAvg()
    save(folders[len(folders)-1])
    clear()
    
    # ------------------------------- Clean Up --------------------------------
    
    m, s = divmod((time.time() - start), 60)
    print "Time taken to run:", int(m), "minutes", round(s,3), "seconds"
    
    # Sound when completed
    if voice_enabled:
        os.system('say "Average of partition submission files computed"')
    clearParams()
    gc.collect() # is this doing anything?

if __name__ == '__main__':
    run({'voice_enabled':False})
