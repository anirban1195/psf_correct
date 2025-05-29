#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 11:52:58 2025

@author: dutta26
"""



import numpy as np
import tensorflow as tf
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D,
    Concatenate, Flatten, Dense, Reshape, Lambda,
    LeakyReLU, Activation, BatchNormalization, Add,
    Cropping2D, Dropout, Permute, Conv1D, RepeatVector
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

@register_keras_serializable()
def compute_percentile(tensor, q, axis=None, keepdims=False):
    """
    Approximate percentile calculation in TensorFlow using tf.sort.
    `q` is in range [0, 100].
    """
    # Flatten along the desired axes
    if axis is None:
        tensor = tf.reshape(tensor, [-1])
    else:
        # Move axis to the end
        rank = len(tensor.shape)
        perm = [i for i in range(rank) if i not in axis] + list(axis)
        tensor = tf.transpose(tensor, perm)
        flat_shape = tf.concat([tf.shape(tensor)[:rank - len(axis)], [-1]], axis=0)
        tensor = tf.reshape(tensor, flat_shape)

    # Sort and get the value at the desired percentile
    sorted_tensor = tf.sort(tensor, axis=-1)
    n = tf.shape(sorted_tensor)[-1]
    k = tf.cast(tf.round(q / 100.0 * tf.cast(n - 1, tf.float32)), tf.int32)
    result = tf.gather(sorted_tensor, k, axis=-1)

    if keepdims:
        for ax in sorted(axis):
            result = tf.expand_dims(result, axis=ax)

    return result

@register_keras_serializable()
def normalize_all_by_blurred(blurred_img, sharp_img, reblurred_img, eps=1e-6):
    # Compute percentiles
    vmin = compute_percentile(blurred_img, 1.0, axis=[1, 2, 3], keepdims=True)
    vmax = compute_percentile(blurred_img, 99.0, axis=[1, 2, 3], keepdims=True)
    scale = vmax - vmin + eps

    # Normalize all images
    blurred_norm   = tf.clip_by_value((blurred_img   - vmin) / scale, 0.0, 2.0)
    sharp_norm     = tf.clip_by_value((sharp_img     - vmin) / scale, 0.0, 2.0)
    reblurred_norm = tf.clip_by_value((reblurred_img - vmin) / scale, 0.0, 2.0)

    return blurred_norm, sharp_norm, reblurred_norm


# --- Data Loading ---
data = np.load("/scratch/bell/dutta26/wiyn_sim/train_data.npz")
blurred = data["blurred"]      # (N,100,100,1)
kernel  = data["psf"]        # (N,20,20,1)
weights = data["weight"]       # (N,100,100,1)
sharp   = data["sharp"]        # (N,100,100,1)

# Crop to 96x96 to ensure U-Net pooling compatibility
def center_crop(img, crop=96):
    start = (img.shape[1] - crop) // 2
    return img[:, start:start+crop, start:start+crop, :]

blurred = center_crop(blurred)
weights = center_crop(weights)
sharp   = center_crop(sharp)

# --- Train/Validation Split ---
(train_blur, val_blur,
 train_ker, val_ker,
 train_sharp, val_sharp,
 train_wt, val_wt) = train_test_split(
    blurred, kernel, sharp, weights,
    test_size=0.2, random_state=42
)

# Convert to TensorFlow tensors for normalization
train_blur = tf.convert_to_tensor(train_blur, dtype=tf.float32)
train_sharp = tf.convert_to_tensor(train_sharp, dtype=tf.float32)
val_blur = tf.convert_to_tensor(val_blur, dtype=tf.float32)
val_sharp = tf.convert_to_tensor(val_sharp, dtype=tf.float32)

# Normalize both blurred and sharp using blurred percentiles
train_blur, train_sharp, _ = normalize_all_by_blurred(train_blur, train_sharp, train_blur)
val_blur, val_sharp, _ = normalize_all_by_blurred(val_blur, val_sharp, val_blur)

# Convert kernel and weights too
train_ker = tf.convert_to_tensor(train_ker, dtype=tf.float32)
val_ker = tf.convert_to_tensor(val_ker, dtype=tf.float32)
train_wt = tf.convert_to_tensor(train_wt, dtype=tf.float32)
val_wt = tf.convert_to_tensor(val_wt, dtype=tf.float32)
     
     
     
# --- Utility Layers ---
@register_keras_serializable()
def compute_ellipticity(image, weight):
    """
    image, weight: (batch, H, W, 1)
    Use provided weight map as-is (no masking).
    """
    eps = 0.1
    img = tf.squeeze(image, axis=-1)       # (batch, H, W)
    wt  = tf.squeeze(weight, axis=-1)      # (batch, H, W)
    I   = img * wt

    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]

    y = tf.cast(tf.range(H), tf.float32) - 47.5
    x = tf.cast(tf.range(W), tf.float32) - 47.5
    X, Y = tf.meshgrid(x, y)               # (H, W)
    X = tf.reshape(X, (1, H, W))
    Y = tf.reshape(Y, (1, H, W))
    X = tf.tile(X, [B, 1, 1])
    Y = tf.tile(Y, [B, 1, 1])
    
    # Compute the required terms
    t1 = tf.reduce_sum(X * Y * I, axis=[1, 2])
    t2 = tf.reduce_sum(I, axis=[1, 2]) + eps
    t3 = tf.reduce_sum(X * I, axis=[1, 2])
    t4 = tf.reduce_sum(Y * I, axis=[1, 2])
    t5 = tf.reduce_sum(X * X * I, axis=[1, 2])
    t6 = tf.reduce_sum(Y * Y * I, axis=[1, 2])

   
    # Compute second moments
    Qxy = (t1 / t2) - (t3 * t4) / (t2 * t2)
    Qxx = (t5 / t2) - (t3 * t3) / (t2 * t2)
    Qyy = (t6 / t2) - (t4 * t4) / (t2 * t2)

    # Compute ellipticity
    denom = Qxx + Qyy + eps
    e1 = (Qxx - Qyy) / denom
    e2 = 2.0 * Qxy / denom

  
    return e1, e2

@register_keras_serializable()
def blur_with_kernel(image, kernel):
    """
    Convolve predicted sharp image with PSF kernel for each sample in the batch.
    image:  (B, H, W, 1)
    kernel: (B, kH, kW, 1)
    Returns:
    blurred image of shape (B, H, W, 1)
    """
    def single_convolve(inputs):
        img, ker = inputs
        # Expand kernel to 3D and normalize
        ker = ker / (tf.reduce_sum(ker) + 1e-6)
        ker = tf.expand_dims(ker, -1)  # (kH, kW, 1, 1)
        # Expand image
        img = tf.expand_dims(img, 0)   # (1, H, W, 1)
        blurred = tf.nn.conv2d(img, ker, strides=[1, 1, 1, 1], padding='SAME')
        return tf.squeeze(blurred, axis=0)

    blurred_batch = tf.map_fn(single_convolve, (image, kernel), dtype=tf.float32)
    return blurred_batch


@register_keras_serializable()
def ellipticity_loss(y_true, y_pred, weight, clip_threshold=0.9):
    e1_true, e2_true = compute_ellipticity(y_true, weight)
    e1_pred, e2_pred = compute_ellipticity(y_pred, weight)
    diff_e1 = tf.clip_by_value(e1_true - e1_pred, -clip_threshold, clip_threshold)
    diff_e2 = tf.clip_by_value(e2_true - e2_pred, -clip_threshold, clip_threshold)
    return tf.reduce_mean(diff_e1**2 + diff_e2**2)



@register_keras_serializable()
# Tile PSF spatially and crop to match image dimensions (96×96) for early fusion
def tile_and_crop_kernel(x):
    # x: (batch,20,20,1) -> tile 5×5 -> (batch,100,100,1) -> crop center 96×96
    # tile 5x5 to cover at least 96 pixels
    t = tf.tile(x, [1, 5, 5, 1])           # (batch,100,100,1)
    # center-crop to 96x96
    return t[:, 2:2+96, 2:2+96, :]


# --- U-Net Blocks ---


@register_keras_serializable()
def conv_block(x, filters, name=None):
    y = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    y = LeakyReLU(alpha=0.1)(y)
    
    y = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(y)
    y = LeakyReLU(alpha=0.1)(y)
    return y



# --- Build Model ---
# Inputs
i_blur = Input(shape=(96,96,1), name='blurred_input')
i_ker  = Input(shape=(20,20,1), name='kernel_input')
i_wt   = Input(shape=(96,96,1), name='weights_input')

# Image encoder branch
c1 = conv_block(i_blur, 64)
p1 = MaxPooling2D(pool_size=2)(c1)       # 48x48

c2 = conv_block(p1, 128)
p2 = MaxPooling2D(pool_size=2)(c2)      # 24x24

c3 = conv_block(p2, 256)
p3 = MaxPooling2D(pool_size=2)(c3)      # 12x12

c4 = conv_block(p3, 512)
p4 = MaxPooling2D(pool_size=2)(c4)      # 6x6

# PSF encoder branch (compact)
k = Conv2D(32, kernel_size=3, padding='same', activation='relu')(i_ker)
k = MaxPooling2D(pool_size=2)(k)
k = Conv2D(64, kernel_size=3, padding='same', activation='relu')(k)
k = MaxPooling2D(pool_size=2)(k)
k = Conv2D(128, kernel_size=3, padding='same', activation='relu')(k)
k = GlobalAveragePooling2D()(k)
k = Dense(256, activation='relu')(k)
k = Dense(1024, activation='relu')(k)
# Repeat 6×6 times and reshape
k = RepeatVector(6 * 6)(k)                     # (None, 36, 1024)
k = Reshape((6, 6, 1024))(k)



# Bridge with PSF late fusion
b = conv_block(p4, 1024)
b = Concatenate()([b, k])
b = Conv2D(1024, 3, padding='same', activation='relu')(b)

# Decoder
u4 = Conv2DTranspose(512, 2, strides=2, padding='same')(b)  # 12x12
u4 = Concatenate()([u4, c4])
c5 = conv_block(u4, 512)

u3 = Conv2DTranspose(256, 2, strides=2, padding='same')(c5) # 24x24
u3 = Concatenate()([u3, c3])
c6 = conv_block(u3, 256)

u2 = Conv2DTranspose(128, 2, strides=2, padding='same')(c6) # 48x48
u2 = Concatenate()([u2, c2])
c7 = conv_block(u2, 128)

u1 = Conv2DTranspose(64, 2, strides=2, padding='same')(c7)  # 96x96
u1 = Concatenate()([u1, c1])
c8 = conv_block(u1, 64)

# Output
output = Conv2D(1, 1, padding='same', activation='linear', name='output')(c8)

# --- Custom Model for Weighted Loss ---
@register_keras_serializable()
class WeightedLossModel(Model):
    def train_step(self, data):
        (x, y_true) = data
        blurred_img, kernel_img, weight_map = x

        with tf.GradientTape() as tape:
            y_pred = self([blurred_img, kernel_img, weight_map], training=True)

            weighted_mse = tf.reduce_mean(tf.square(y_pred - y_true)* (weight_map) )
            ellip_loss = ellipticity_loss(y_true, y_pred, weight_map)
            #reblurred = blur_with_kernel(y_pred, kernel_img)
            _, _, reblurred = normalize_all_by_blurred(blurred_img, y_pred, blur_with_kernel(y_pred, kernel_img))
            reblur_loss = tf.reduce_mean(tf.square(reblurred - blurred_img)* (weight_map))

            total_loss = 100*weighted_mse + 10*reblur_loss  

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_true, y_pred)
        return {"loss": total_loss, "weighted_mse": weighted_mse, "ellipticity_loss": ellip_loss, "reblur_loss":reblur_loss}

    def test_step(self, data):
        (x, y_true) = data
        blurred_img, kernel_img, weight_map = x
        y_pred = self([blurred_img, kernel_img, weight_map], training=False)

        weighted_mse = tf.reduce_mean(tf.square(y_pred - y_true)* (weight_map) )
        ellip_loss = ellipticity_loss(y_true, y_pred, weight_map)
        #reblurred = blur_with_kernel(y_pred, kernel_img)
        _, _, reblurred = normalize_all_by_blurred(blurred_img, y_pred, blur_with_kernel(y_pred, kernel_img))
        reblur_loss = tf.reduce_mean(tf.square(reblurred - blurred_img)* (weight_map))
        
        total_loss = 100*weighted_mse +  10*reblur_loss

        self.compiled_metrics.update_state(y_true, y_pred)
        return {"loss": total_loss, "weighted_mse": weighted_mse, "ellipticity_loss": ellip_loss, "reblur_loss":reblur_loss}

model = WeightedLossModel(inputs=[i_blur, i_ker, i_wt], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5))

# --- Training ---

from keras.callbacks import LearningRateScheduler

def lr_schedule(epoch, lr):
    decay_rate = 0.8
    return lr * decay_rate

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

history = model.fit(
    x=[train_blur, train_ker, train_wt],
    y=train_sharp,
    validation_data=([val_blur, val_ker, val_wt], val_sharp),
    epochs=20,
    batch_size=256,
    verbose = 1,
    callbacks=[lr_scheduler]
)

# Save model
model.save('/scratch/bell/dutta26/psf_datasets/unet_psf_model_bottleNeckFusion1_consistency.keras')


import matplotlib.pyplot as plt

# Extract loss values
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training Loss', marker='o')
plt.plot(val_loss, label='Validation Loss', marker='s')
plt.title('Training and Validation Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('/scratch/bell/dutta26/psf_datasets/unet_psf_model_bottleNeckFusion_loss_plot_consistency.png')
plt.close() 
