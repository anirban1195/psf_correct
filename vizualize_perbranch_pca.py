#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:00:41 2025

@author: dutta26
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:32:47 2025

@author: dutta26
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from keras.models import load_model
import os
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

# --- Load your model ---
model = load_model("/scratch/bell/dutta26/psf_datasets/unet_psf_model_bottleNeckFusion1_consistency.keras")

# Output directory
base_dir = "pca_per_branch"
os.makedirs(base_dir, exist_ok=True)

# Define branches: update these lists with your actual layer names
image_encoder_layers = [
    'conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5'
]

psf_encoder_layers = [
    'conv2d_6', 'conv2d_7', 'conv2d_8', 'dense', 'dense_1'
]

decoder_layers = [
    'conv2d_9', 'conv2d_10', 'conv2d_11', 'conv2d_12', 'conv2d_13',
    'conv2d_14', 'conv2d_15', 'conv2d_16'
]

branch_map = {
    "image_encoder": image_encoder_layers,
    "psf_encoder": psf_encoder_layers,
    "decoder": decoder_layers
}

# Utility: Apply PCA to weights and plot
def pca_and_plot(weights, layer_name, branch):
    try:
        if weights.ndim == 4:  # Conv2D: (H, W, in_ch, out_ch)
            reshaped = weights.reshape(-1, weights.shape[-1])
        elif weights.ndim == 2:  # Dense: (in, out)
            reshaped = weights
        else:
            return

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(reshaped)

        # Plot
        plt.figure(figsize=(6, 5))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        plt.title(f"PCA - {layer_name} ({branch})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)

        # Save plot
        branch_dir = os.path.join(base_dir, branch)
        os.makedirs(branch_dir, exist_ok=True)
        plt.savefig(os.path.join(branch_dir, f"{layer_name}_pca.png"))
        plt.close()

        print(f"[{branch}] Saved PCA plot for {layer_name}")

    except Exception as e:
        print(f"[{branch}] Skipped {layer_name} due to error: {e}")

# Loop through all layers and group by branch
for branch, layer_names in branch_map.items():
    for layer_name in layer_names:
        layer = model.get_layer(layer_name)
        weights = layer.get_weights()
        if weights:
            w = weights[0]  # First element is usually kernel weights
            if w.size > 10:  # Skip tiny weights
                pca_and_plot(w, layer_name, branch)