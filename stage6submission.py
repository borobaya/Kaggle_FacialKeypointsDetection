# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:32:14 2015

@author: Muhammed Onu Miah

Experimenting on detecting facial features

This file creates a file ready for submission

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

faceTestZ = None
lookupTable = None
def load():
    global faceTestZ, lookupTable
    faceTestZ = np.load('Cache/FaceTestZ.npy')
    faceTestZ = pd.DataFrame(faceTestZ,
        columns=['left_eye_center_x', 'left_eye_center_y',
        'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x',
        'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
        'left_eye_outer_corner_y', 'right_eye_inner_corner_x',
        'right_eye_inner_corner_y', 'right_eye_outer_corner_x',
        'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
        'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x',
        'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
        'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
        'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y',
        'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x',
        'mouth_right_corner_y', 'mouth_center_top_lip_x',
        'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x',
        'mouth_center_bottom_lip_y'])
    lookupTable = pd.DataFrame.from_csv('Data/IdLookupTable.csv')

def save():
    submission = lookupTable.drop(['ImageId', 'FeatureName'], axis=1)
    submission.to_csv('Cache/submission.csv')

def clear():
    global faceTestZ, lookupTable
    faceTestZ = None
    lookupTable = None

# ---------------------------------- Methods ----------------------------------

def setCoords():
    jmax = faceTestZ.shape[0]
    for i in xrange(lookupTable.shape[0]):
        row = lookupTable.irow(i)
        j = row.ImageId-1
        feature = row.FeatureName
        if j<jmax:
            val = faceTestZ.irow(j)[feature]
            lookupTable.set_value(i+1, 'Location', val)

# -----------------------------------------------------------------------------

def run(params):
    start = time.time()
    setParams(params)
    # -------------------------------------------------------------------------
    print "--- Submission File ---"
    
    load()
    setCoords()
    save()
    clear()
    
    # ------------------------------- Clean Up --------------------------------
    
    m, s = divmod((time.time() - start), 60)
    print "Time taken to run:", int(m), "minutes", round(s,3), "seconds"
    
    # Sound when completed
    if voice_enabled:
        os.system('say "Submission file created"')
    clearParams()
    gc.collect() # is this doing anything?

if __name__ == '__main__':
    run({'voice_enabled':False})
