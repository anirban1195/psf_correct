#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 07:59:33 2025

@author: dutta26
"""

import numpy as np
from astropy.io import fits
import helper_phosim, helper, os
import subprocess,sys
import matplotlib.pyplot as plt

filt = 'i' #str(sys.argv[1])
idNo = 1020
noImages = 6


blurred_list = []
psf_list     = []
psf_large_list = []
weight_list  = []
sharp_list   = []

# =============================================================================
# #Augment with fake data
# blurred_list.append(np.load('/scratch/bell/dutta26/psf_datasets/blurred.npy')[:,:,:,0])   
# psf_list.append(np.load('/scratch/bell/dutta26/psf_datasets/kernel.npy')[:,:,:,0])                   # Shape: (20000, 20, 20, 1)
# sharp_list.append(np.load('/scratch/bell/dutta26/psf_datasets/truth.npy')[:,:,:,0])        # Shape: (20000, 100, 100, 1)
# weight_list.append(np.load('/scratch/bell/dutta26/psf_datasets/wt.npy')[:,:,:,0])      
# =============================================================================


for j in range(int(idNo), int(idNo)+noImages):
    
    #For normal images
    outLoc = '/scratch/bell/dutta26/wiyn_sim/'+filt+'/'+str(j)
    
    blurred_list.append(np.load(outLoc+"/blurred.npy"))
    psf_list.append(np.load(outLoc+"/psf.npy"))
    psf_large_list.append(np.load(outLoc+"/psf_large.npy"))
    weight_list.append(np.load(outLoc+"/weight.npy"))
    sharp_list.append(np.load(outLoc+"/sharp.npy"))

    
# Concatenate arrays from the lists
blurred = np.concatenate(blurred_list, axis=0)
psf     = np.concatenate(psf_list, axis=0)
psf_large = np.concatenate(psf_large_list, axis=0)
weight  = np.concatenate(weight_list, axis=0)
sharp   = np.concatenate(sharp_list, axis=0)

# Shuffle using indices
n = blurred.shape[0]
indices = np.arange(n)
np.random.shuffle(indices)

blurred = blurred[indices]
psf     = psf[indices]
weight  = weight[indices]
sharp   = sharp[indices]
psf_large = psf_large[indices]

# Split 85% train, 15% test
split_idx = int(n * 0.85)

blurred_train, blurred_test = blurred[:split_idx], blurred[split_idx:]
psf_train,     psf_test     = psf[:split_idx],     psf[split_idx:]
weight_train,  weight_test  = weight[:split_idx],  weight[split_idx:]
sharp_train,   sharp_test   = sharp[:split_idx],   sharp[split_idx:]
psf_large_train, psf_large_test = psf_large[:split_idx],     psf_large[split_idx:]



# Expand dims to add the channel axis
blurred_train = np.expand_dims(blurred_train, axis=-1)  # shape: (n,100,100,1)
weight_train  = np.expand_dims(weight_train, axis=-1)
sharp_train   = np.expand_dims(sharp_train, axis=-1)
psf_train     = np.expand_dims(psf_train, axis=-1)      # shape: (n,20,20,1)

blurred_test = np.expand_dims(blurred_test, axis=-1)
weight_test  = np.expand_dims(weight_test, axis=-1)
sharp_test   = np.expand_dims(sharp_test, axis=-1)
psf_test     = np.expand_dims(psf_test, axis=-1)



# Save training set
np.savez_compressed("/scratch/bell/dutta26/wiyn_sim/train_data.npz",
    blurred=blurred_train,
    psf=psf_train,
    weight=weight_train,
    sharp=sharp_train,
    psf_large= psf_large_train
)

# Save test set
np.savez_compressed("/scratch/bell/dutta26/wiyn_sim/test_data.npz",
    blurred=blurred_test,
    psf=psf_test,
    weight=weight_test,
    sharp=sharp_test,
    psf_large = psf_large_test
)



    
    