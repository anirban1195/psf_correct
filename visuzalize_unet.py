#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 07:04:25 2025

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
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D,
    Concatenate, Flatten, Dense, Reshape, Lambda,
    LeakyReLU, Activation, BatchNormalization, Add,
    Cropping2D, Dropout, Conv1D, Permute
)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import register_keras_serializable
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply
from keras.layers import (
    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense,
    Reshape, Multiply, Add, Concatenate, Conv2D, Activation, Lambda
)
from keras import backend as K
from keras.models import load_model


@register_keras_serializable()
def tile_and_crop_kernel(x):
    # x: (batch,20,20,1) -> tile 5×5 -> (batch,100,100,1) -> crop center 96×96
    # tile 5x5 to cover at least 96 pixels
    t = tf.tile(x, [1, 5, 5, 1])           # (batch,100,100,1)
    # center-crop to 96x96
    return t[:, 2:2+96, 2:2+96, :]

@tf.keras.utils.register_keras_serializable()
def center_crop(img, crop=96):
    start = (img.shape[1] - crop) // 2
    return img[:, start:start+crop, start:start+crop, :]

@tf.keras.utils.register_keras_serializable()
def compute_ellipticity(image, weight):
    eps = 0.1
    img = tf.squeeze(image, axis=-1)
    wt = tf.squeeze(weight, axis=-1)
    I = img * wt

    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]

    y = tf.cast(tf.range(H), tf.float32) - 47.5
    x = tf.cast(tf.range(W), tf.float32) - 47.5
    X, Y = tf.meshgrid(x, y)
    X = tf.reshape(X, (1, H, W))
    Y = tf.reshape(Y, (1, H, W))
    X = tf.tile(X, [B, 1, 1])
    Y = tf.tile(Y, [B, 1, 1])

    t1 = tf.reduce_sum(X * Y * I, axis=[1, 2])
    t2 = tf.reduce_sum(I, axis=[1, 2]) + eps
    t3 = tf.reduce_sum(X * I, axis=[1, 2])
    t4 = tf.reduce_sum(Y * I, axis=[1, 2])
    t5 = tf.reduce_sum(X * X * I, axis=[1, 2])
    t6 = tf.reduce_sum(Y * Y * I, axis=[1, 2])

    Qxy = (t1 / t2) - (t3 * t4) / (t2 * t2)
    Qxx = (t5 / t2) - (t3 * t3) / (t2 * t2)
    Qyy = (t6 / t2) - (t4 * t4) / (t2 * t2)

    denom = Qxx + Qyy + eps
    e1 = (Qxx - Qyy) / denom
    e2 = 2.0 * Qxy / denom
    return e1, e2

@tf.keras.utils.register_keras_serializable()
def ellipticity_loss(y_true, y_pred, weight, clip_threshold=0.9):
    e1_true, e2_true = compute_ellipticity(y_true, weight)
    e1_pred, e2_pred = compute_ellipticity(y_pred, weight)
    diff_e1 = tf.clip_by_value(e1_true - e1_pred, -clip_threshold, clip_threshold)
    diff_e2 = tf.clip_by_value(e2_true - e2_pred, -clip_threshold, clip_threshold)
    return tf.reduce_mean(diff_e1**2 + diff_e2**2)

@tf.keras.utils.register_keras_serializable()
class WeightedLossModel(Model):
    def train_step(self, data):
        (x, y_true) = data
        blurred_img, kernel_img, weight_map = x

        with tf.GradientTape() as tape:
            y_pred = self([blurred_img, kernel_img, weight_map], training=True)

            weighted_mse = tf.reduce_mean(tf.square(y_pred - y_true)* (weight_map) )
            ellip_loss = ellipticity_loss(y_true, y_pred, weight_map)

            total_loss = 100*weighted_mse + 0.005* ellip_loss  # weight the ellipticity term

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_true, y_pred)
        return {"loss": total_loss, "weighted_mse": weighted_mse, "ellipticity_loss": ellip_loss}

    def test_step(self, data):
        (x, y_true) = data
        blurred_img, kernel_img, weight_map = x
        y_pred = self([blurred_img, kernel_img, weight_map], training=False)

        weighted_mse = tf.reduce_mean(tf.square(y_pred - y_true)* (weight_map) )
        ellip_loss = ellipticity_loss(y_true, y_pred, weight_map)
        total_loss = 100*weighted_mse + 0.005 * ellip_loss

        self.compiled_metrics.update_state(y_true, y_pred)
        return {"loss": total_loss, "weighted_mse": weighted_mse, "ellipticity_loss": ellip_loss}

# Load model with custom layers/functions
model = load_model(
    '/scratch/bell/dutta26/psf_datasets/unet_psf_model_bottleNeckFusion1.h5',
    custom_objects={
        'tile_and_crop_kernel': tile_and_crop_kernel,
        'WeightedLossModel': WeightedLossModel,
        'ellipticity_loss': ellipticity_loss,
        'compute_ellipticity': compute_ellipticity
    },
    compile=False
)

# Load data
data = np.load("/scratch/bell/dutta26/wiyn_sim/test_data.npz")
blurred = center_crop(data["blurred"])     # (N,96,96,1)
kernel  = data["psf"]                      # (N,10,10,1)
weights = center_crop(data["weight"])      # (N,96,96,1)
sharp   = center_crop(data["sharp"])       # (N,96,96,1)

# Select an index
idx = 677  # change as needed

# Prepare inputs
blur_img   = blurred[idx:idx+1]
kernel_img = kernel[idx:idx+1]
weight_map = weights[idx:idx+1]
true_img   = sharp[idx, :, :, 0]

# Predict
pred_img = model.predict([blur_img, kernel_img, weight_map])[0, :, :, 0]

# Plot
plt.figure(figsize=(12, 3.5))
plt.subplot(1, 5, 1)
plt.title("Blurred")
plt.imshow(blur_img[0, :, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.title("Kernel")
plt.imshow(kernel_img[0, :, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.title("Ground Truth")
plt.imshow(true_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.title("Predicted")
plt.imshow(pred_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.title("Weight")
plt.imshow(weight_map[0, :, :, 0], cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()