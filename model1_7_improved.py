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
def compute_ellipticity_batched_tf(images, counter_target=100, convergence_threshold=1e-3):
    """
    Compute e1, e2 ellipticities from a batch of images using iterative moment matching with adaptive Gaussian weighting.
    Args:
        images: Tensor of shape [B, H, W], batch of images.
        counter_target: Maximum number of iterations (default: 100)
        convergence_threshold: Threshold for convergence check (default: 1e-6)
    Returns:
        e1: Tensor of shape [B], ellipticity component 1
        e2: Tensor of shape [B], ellipticity component 2
    """
    B, H, W = tf.unstack(tf.shape(images))
    y = tf.linspace(0.0, tf.cast(H - 1, tf.float32), H) - tf.cast(H, tf.float32) / 2.0 + 0.5
    x = tf.linspace(0.0, tf.cast(W - 1, tf.float32), W) - tf.cast(W, tf.float32) / 2.0 + 0.5
    Y, X = tf.meshgrid(y, x, indexing='ij')  # [H, W]
    X = tf.expand_dims(X, 0)  # [1, H, W]
    Y = tf.expand_dims(Y, 0)  # [1, H, W]

    # Initial guesses
    alphax = tf.ones([B]) * 3.0
    alphay = tf.ones([B]) * 3.0
    alphaxy = tf.zeros([B])
    mux = tf.zeros([B])
    muy = tf.zeros([B])
    back = tf.zeros_like(images[:, 0, 0])
    prev_sigxx = tf.ones([B]) * 9999.0
    prev_sigyy = tf.ones([B]) * 9999.0
    curr_sigxx = tf.zeros([B])
    curr_sigyy = tf.zeros([B])

    def cond(i, alphax, alphay, alphaxy, mux, muy, prev_sigxx, prev_sigyy, curr_sigxx, curr_sigyy, back, e1, e2):
        # Always run at least one iteration
        if_first_iter = tf.equal(i, 0)
        # Check if we've reached max iterations
        max_iter_reached = tf.less(i, counter_target)
        # Check convergence by comparing current and previous sigxx/sigyy values
        sigxx_converged = tf.less(tf.reduce_max(tf.abs(curr_sigxx - prev_sigxx)), convergence_threshold)
        sigyy_converged = tf.less(tf.reduce_max(tf.abs(curr_sigyy - prev_sigyy)), convergence_threshold)
        converged = tf.logical_and(sigxx_converged, sigyy_converged)
        # Don't check convergence on first iteration (prev values are initialization values)
        converged = tf.logical_and(converged, tf.logical_not(if_first_iter))
        # Continue if: (not reached max iterations) AND (first iteration OR not converged)
        return tf.logical_and(max_iter_reached, tf.logical_or(if_first_iter, tf.logical_not(converged)))

    def body(i, alphax, alphay, alphaxy, mux, muy, prev_sigxx, prev_sigyy, curr_sigxx, curr_sigyy, back, e1, e2):
        Xc = X - tf.reshape(mux, [-1, 1, 1])
        Yc = Y - tf.reshape(muy, [-1, 1, 1])
        # Prevent numerical instability in arb_const calculation
        alpha_ratio = alphaxy / (alphax * alphay + 1e-10)  # Prevent division by zero
        alpha_ratio = tf.clip_by_value(alpha_ratio, -0.99, 0.99)
        arb_const = 2.0 * (1.0 - alpha_ratio**2)
        arb_const = tf.maximum(arb_const, 1e-10)  # Prevent division by zero
        A = 1.0 / (2.0 * tf.constant(np.pi) * alphax * alphay * tf.sqrt(1.0 - alpha_ratio**2))
        exp_term = (
            (Xc**2) / (tf.reshape(arb_const * alphax**2, [-1, 1, 1])) +
            (Yc**2) / (tf.reshape(arb_const * alphay**2, [-1, 1, 1])) -
            2 * tf.reshape(alphaxy, [-1, 1, 1]) * Xc * Yc /
            tf.reshape(arb_const * alphax**2 * alphay**2, [-1, 1, 1])
        )
        k = tf.reshape(A, [-1, 1, 1]) * tf.exp(-tf.clip_by_value(exp_term, 0.0, 50.0))
        img1 = images - tf.reshape(back, [-1, 1, 1])
        t1 = tf.reduce_sum(X * Y * img1 * k, axis=[1, 2])
        t2 = tf.reduce_sum(img1 * k, axis=[1, 2])
        t3 = tf.reduce_sum(X * img1 * k, axis=[1, 2])
        t4 = tf.reduce_sum(Y * img1 * k, axis=[1, 2])
        t5 = tf.reduce_sum(X * X * img1 * k, axis=[1, 2])
        t6 = tf.reduce_sum(Y * Y * img1 * k, axis=[1, 2])

        # Adaptive background estimation
        t7 = tf.reduce_sum(k * k, axis=[1, 2])          # Denominator
        flux_calc = t2 / (t7 + 1e-10)                   # Flux under Gaussian PSF
        total = tf.reduce_sum(images, axis=[1, 2])      # Total image flux
        image_area = tf.cast(H * W, tf.float32)         # Constant
        new_back = (total - flux_calc) / image_area     # Estimated background
        # Prevent division by zero in moment calculations
        t2_safe = tf.maximum(t2, 1e-10)
        new_mux = t3 / t2_safe
        new_muy = t4 / t2_safe
        sigxx = t5 / t2_safe - (t3 / t2_safe)**2
        sigyy = t6 / t2_safe - (t4 / t2_safe)**2
        sigxy = t1 / t2_safe - (t3 * t4) / (t2_safe * t2_safe)
        #print (sigxx)
        # Update alpha values with bounds checking
        new_alphax = tf.sqrt(tf.clip_by_value(sigxx * 2.0, 0.81, 100.0))
        new_alphay = tf.sqrt(tf.clip_by_value(sigyy * 2.0, 0.81, 100.0))
        new_alphaxy = 2.0 * sigxy
        #print (new_alphax)
        # Compute ellipticities with numerical stability
        denominator = sigxx + sigyy + 1e-10  # Prevent division by zero
        new_e1 = (sigxx - sigyy) / denominator
        new_e2 = 2.0 * sigxy / denominator
        # Clip ellipticities to reasonable bounds
        new_e1 = tf.clip_by_value(new_e1, -0.99, 0.99)
        new_e2 = tf.clip_by_value(new_e2, -0.99, 0.99)

        # Return as list to match loop_vars structure
        # Move current sigxx/sigyy to prev for next iteration, update current with new values
        return [i + 1, new_alphax, new_alphay, new_alphaxy, new_mux, new_muy,
                curr_sigxx, curr_sigyy, sigxx, sigyy, new_back, new_e1, new_e2]

    # Initialize loop variables
    i = tf.constant(0)
    e1 = tf.zeros([B])
    e2 = tf.zeros([B])

    # Run the iterative loop with proper convergence checking
    # Use list structure for loop_vars to match body function return
    loop_result = tf.while_loop(
        cond,
        body,
        loop_vars=[i, alphax, alphay, alphaxy, mux, muy, prev_sigxx, prev_sigyy, curr_sigxx, curr_sigyy, back, e1, e2],
        maximum_iterations=counter_target,
        parallel_iterations=1  # Ensure sequential execution for convergence
    )

    # Extract e1 and e2 from the result
    e1 = loop_result[11]
    e2 = loop_result[12]
    
    e1 = tf.where(tf.math.is_nan(e1), 0.0, e1)
    e2 = tf.where(tf.math.is_nan(e2), 0.0, e2)


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)


    def train_step(self, data):
        (x, y_true) = data
        blurred_img, kernel_img, weight_map = x

        with tf.GradientTape() as tape:
            y_pred = self([blurred_img, kernel_img, weight_map], training=True)

            weighted_mse = tf.reduce_mean(tf.square(y_pred - y_true)* (weight_map) )
           # Remove channels for ellipticity calculation
            sharp_true = tf.squeeze(y_true, axis=-1)  # shape [B, H, W]
            sharp_pred = tf.squeeze(y_pred, axis=-1)
            
            # Compute ellipticities
            e1_true, e2_true = compute_ellipticity_batched_tf(sharp_true)
            e1_pred, e2_pred = compute_ellipticity_batched_tf(sharp_pred)
            
            # Ellipticity loss (L2)
            ellip_loss = tf.reduce_mean((e1_true - e1_pred)**2 + (e2_true - e2_pred)**2)
            ellip_weight = tf.constant(0.0, dtype=tf.float32) + tf.constant(0.1, dtype=tf.float32) * tf.cast(tf.minimum(self.current_epoch, 9), tf.float32)            #reblurred = blur_with_kernel(y_pred, kernel_img)
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
        # Remove channels for ellipticity calculation
        sharp_true = tf.squeeze(y_true, axis=-1)  # shape [B, H, W]
        sharp_pred = tf.squeeze(y_pred, axis=-1)
        
        # Compute ellipticities
        e1_true, e2_true = compute_ellipticity_batched_tf(sharp_true)
        e1_pred, e2_pred = compute_ellipticity_batched_tf(sharp_pred)
        
        # Ellipticity loss (L2)
        ellip_loss = tf.reduce_mean((e1_true - e1_pred)**2 + (e2_true - e2_pred)**2)
        ellip_weight = tf.constant(0.0, dtype=tf.float32) + tf.constant(0.1, dtype=tf.float32) * tf.cast(tf.minimum(self.current_epoch, 9), tf.float32)        #reblurred = blur_with_kernel(y_pred, kernel_img)
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

class EpochTracker(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch.assign(epoch)


history = model.fit(
    x=[train_blur, train_ker, train_wt],
    y=train_sharp,
    validation_data=([val_blur, val_ker, val_wt], val_sharp),
    epochs=10,
    batch_size=256,
    verbose = 1,
    callbacks= [lr_scheduler, EpochTracker()]
)

# Save model
model.save('/scratch/bell/dutta26/psf_datasets/unet_psf_model_bottleNeckFusion_consistency3.keras')


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
plt.savefig('/scratch/bell/dutta26/psf_datasets/unet_psf_model_bottleNeckFusion_loss_plot_consistency3.png')
plt.close() 
