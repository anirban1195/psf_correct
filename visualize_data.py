#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 09:27:13 2025

@author: dutta26
"""

import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, Reshape
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


#GET DATA and Model from here 
#https://www.dropbox.com/scl/fo/fwhe3rgwm33w15uhodldq/ANkZoWEjrYuiEDEPIQ_6fIA?rlkey=ylr9fivh3zueu7ebyvbek4ez8&st=xn2o4bgw&dl=0

def center_crop(img, crop=96):
    start = (img.shape[1] - crop) // 2
    return img[:, start:start+crop, start:start+crop, :]

# Load data
data = np.load("/scratch/bell/dutta26/wiyn_sim/augment.npz")
blurred = center_crop(data["blurred"])     # (N,96,96,1)
kernel  = data["psf"]                      # (N,20,20,1)
weights = center_crop(data["weight"])      # (N,96,96,1)
sharp   = center_crop(data["sharp"])       # (N,96,96,1)

# Select an index
idx = 677  # change as needed

# Prepare inputs
blur_img   = blurred[idx:idx+1]
kernel_img = kernel[idx:idx+1]
weight_map = weights[idx:idx+1]
true_img   = sharp[idx, :, :, 0]


# Plot
plt.figure(figsize=(12, 3.5))
plt.subplot(1, 4, 1)
plt.title("Blurred")
plt.imshow(blur_img[0, :, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Kernel")
plt.imshow(kernel_img[0, :, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Ground Truth")
plt.imshow(true_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Weight")
plt.imshow(weight_map[0, :, :, 0], cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()