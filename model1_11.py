#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 10:40:18 2025

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

# --- Data Loading ---
data = np.load("/scratch/bell/dutta26/wiyn_sim/train_data.npz")
blurred = data["blurred"]      # (N,100,100,3,1)
kernel  = data["psf"]        # (N,48,48,1)
weights = data["weight"]       # (N,100,100,1)
sharp   = data["sharp"]        # (N,100,100,1)

print("Original data shapes:")
print(f"blurred: {blurred.shape}")
print(f"kernel: {kernel.shape}")
print(f"weights: {weights.shape}")
print(f"sharp: {sharp.shape}")

# Crop to 96x96 to ensure U-Net pooling compatibility
def center_crop(img, crop=96):
    start = (img.shape[1] - crop) // 2
    return img[:, start:start+crop, start:start+crop, :]

# Fix the dimensional issue by properly handling the 5D blurred tensor
def center_crop_5d(img, crop=96):
    """Crop 5D tensor (N, H, W, C, 1) to (N, crop, crop, C)"""
    start = (img.shape[1] - crop) // 2
    cropped = img[:, start:start+crop, start:start+crop, :, 0]  # Remove last dimension
    return cropped

# Apply cropping with proper dimension handling
blurred = center_crop_5d(blurred, crop=96)  # Now (N, 96, 96, 3)
weights = center_crop(weights, crop=96)     # (N, 96, 96, 1)
sharp   = center_crop(sharp, crop=96)       # (N, 96, 96, 1)
# kernel remains (N, 48, 48, 1)


# =============================================================================
# #Check dimension 1
# print("\nAfter cropping:")
# print(f"blurred: {blurred.shape}")
# print(f"kernel: {kernel.shape}")
# print(f"weights: {weights.shape}")
# print(f"sharp: {sharp.shape}")
# =============================================================================

# =============================================================================
# # Verify the channels are correct. Check dim 2
# print("\nChannel verification:")
# print(f"blurred channel 0 range: [{blurred[:,:,:,0].min():.4f}, {blurred[:,:,:,0].max():.4f}]")
# print(f"blurred channel 1 range: [{blurred[:,:,:,1].min():.4f}, {blurred[:,:,:,1].max():.4f}]")
# print(f"blurred channel 2 range: [{blurred[:,:,:,2].min():.4f}, {blurred[:,:,:,2].max():.4f}]")
# =============================================================================

# --- Train/Validation Split ---
(train_blur, val_blur,
 train_ker, val_ker,
 train_sharp, val_sharp,
 train_wt, val_wt) = train_test_split(
    blurred, kernel, sharp, weights,
    test_size=0.2, random_state=42
)

# =============================================================================
# #Chek dim 3
# print("\nTraining data shapes:")
# print(f"train_blur: {train_blur.shape}, val_blur: {val_blur.shape}")
# print(f"train_ker: {train_ker.shape}, val_ker: {val_ker.shape}")
# print(f"train_wt: {train_wt.shape}, val_wt: {val_wt.shape}")
# print(f"train_sharp: {train_sharp.shape}, val_sharp: {val_sharp.shape}")
# =============================================================================

# --- Utility Functions (keeping your existing ones) ---
@register_keras_serializable()
def compute_ellipticity_batched_tf(images, counter_target=40, convergence_threshold=1e-2):
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
    """
    Robust version with extensive NaN prevention
    """
    # [Keep your existing implementation]
    B, H, W = tf.unstack(tf.shape(images))
    y = tf.linspace(0.0, tf.cast(H - 1, tf.float32), H) - tf.cast(H, tf.float32) / 2.0 + 0.5
    x = tf.linspace(0.0, tf.cast(W - 1, tf.float32), W) - tf.cast(W, tf.float32) / 2.0 + 0.5
    Y, X = tf.meshgrid(y, x, indexing='ij')
    X = tf.expand_dims(X, 0)
    Y = tf.expand_dims(Y, 0)

    # More conservative initial conditions
    alphax = tf.ones([B]) * 2.0
    alphay = tf.ones([B]) * 2.0
    alphaxy = tf.zeros([B])
    mux = tf.zeros([B])
    muy = tf.zeros([B])
    back = tf.zeros_like(images[:, 0, 0])

    # Initialize with reasonable values
    prev_sigxx = tf.ones([B]) * 4.0
    prev_sigyy = tf.ones([B]) * 4.0
    curr_sigxx = tf.zeros([B])
    curr_sigyy = tf.zeros([B])

    def cond(i, alphax, alphay, alphaxy, mux, muy, prev_sigxx, prev_sigyy, curr_sigxx, curr_sigyy, back, e1, e2):
        if_first_iter = tf.equal(i, 0)
        max_iter_reached = tf.less(i, counter_target)

        sigxx_diff = tf.abs(curr_sigxx - prev_sigxx)
        sigyy_diff = tf.abs(curr_sigyy - prev_sigyy)

        sigxx_valid = tf.logical_not(tf.reduce_any(tf.math.is_nan(curr_sigxx)))
        sigyy_valid = tf.logical_not(tf.reduce_any(tf.math.is_nan(curr_sigyy)))
        values_valid = tf.logical_and(sigxx_valid, sigyy_valid)

        sigxx_converged = tf.less(tf.reduce_max(sigxx_diff), convergence_threshold)
        sigyy_converged = tf.less(tf.reduce_max(sigyy_diff), convergence_threshold)
        converged = tf.logical_and(sigxx_converged, sigyy_converged)
        converged = tf.logical_and(converged, tf.logical_not(if_first_iter))

        should_continue = tf.logical_and(max_iter_reached, values_valid)
        should_continue = tf.logical_and(should_continue, tf.logical_or(if_first_iter, tf.logical_not(converged)))

        return should_continue

    def body(i, alphax, alphay, alphaxy, mux, muy, prev_sigxx, prev_sigyy, curr_sigxx, curr_sigyy, back, e1, e2):
        # [Keep your existing body implementation]
        alphax = tf.clip_by_value(alphax, 0.5, 20.0)
        alphay = tf.clip_by_value(alphay, 0.5, 20.0)
        alphaxy = tf.clip_by_value(alphaxy, -10.0, 10.0)

        Xc = X - tf.reshape(mux, [-1, 1, 1])
        Yc = Y - tf.reshape(muy, [-1, 1, 1])

        alpha_denom = tf.maximum(alphax * alphay, 1e-8)
        alpha_ratio = tf.clip_by_value(alphaxy / alpha_denom, -0.95, 0.95)
        arb_const = tf.maximum(2.0 * (1.0 - alpha_ratio**2), 1e-8)

        sqrt_term = tf.maximum(1.0 - alpha_ratio**2, 1e-8)
        A = 1.0 / (2.0 * tf.constant(np.pi) * alpha_denom * tf.sqrt(sqrt_term))
        A = tf.clip_by_value(A, 1e-10, 1e10)

        alphax_sq = tf.maximum(alphax**2, 1e-8)
        alphay_sq = tf.maximum(alphay**2, 1e-8)

        exp_term1 = (Xc**2) / tf.reshape(arb_const * alphax_sq, [-1, 1, 1])
        exp_term2 = (Yc**2) / tf.reshape(arb_const * alphay_sq, [-1, 1, 1])
        exp_term3 = 2 * tf.reshape(alphaxy, [-1, 1, 1]) * Xc * Yc / tf.reshape(arb_const * alphax_sq * alphay_sq, [-1, 1, 1])

        exp_term = exp_term1 + exp_term2 - exp_term3
        exp_term = tf.clip_by_value(exp_term, 0.0, 30.0)

        k = tf.reshape(A, [-1, 1, 1]) * tf.exp(-exp_term)
        k = tf.where(tf.math.is_nan(k), 0.0, k)
        k = tf.where(tf.math.is_inf(k), 0.0, k)

        img1 = images - tf.reshape(back, [-1, 1, 1])

        t1 = tf.reduce_sum(X * Y * img1 * k, axis=[1, 2])
        t2 = tf.reduce_sum(img1 * k, axis=[1, 2])
        t3 = tf.reduce_sum(X * img1 * k, axis=[1, 2])
        t4 = tf.reduce_sum(Y * img1 * k, axis=[1, 2])
        t5 = tf.reduce_sum(X * X * img1 * k, axis=[1, 2])
        t6 = tf.reduce_sum(Y * Y * img1 * k, axis=[1, 2])
        t7 = tf.reduce_sum(k * k, axis=[1, 2])

        t2_safe = tf.maximum(tf.abs(t2), 1e-6)
        t7_safe = tf.maximum(t7, 1e-6)

        flux_calc = t2 / t7_safe
        total = tf.reduce_sum(images, axis=[1, 2])
        image_area = tf.cast(H * W, tf.float32)
        new_back = tf.clip_by_value((total - flux_calc) / image_area, -1.0, 1.0)

        new_mux = tf.clip_by_value(t3 / t2_safe, -tf.cast(W, tf.float32)/2, tf.cast(W, tf.float32)/2)
        new_muy = tf.clip_by_value(t4 / t2_safe, -tf.cast(H, tf.float32)/2, tf.cast(H, tf.float32)/2)

        sigxx = t5 / t2_safe - (t3 / t2_safe)**2
        sigyy = t6 / t2_safe - (t4 / t2_safe)**2
        sigxy = t1 / t2_safe - (t3 * t4) / (t2_safe * t2_safe)

        sigxx = tf.clip_by_value(sigxx, 0.25, 100.0)
        sigyy = tf.clip_by_value(sigyy, 0.25, 100.0)
        sigxy = tf.clip_by_value(sigxy, -50.0, 50.0)

        new_alphax = tf.sqrt(tf.clip_by_value(sigxx * 2.0, 1.0, 400.0))
        new_alphay = tf.sqrt(tf.clip_by_value(sigyy * 2.0, 1.0, 400.0))
        new_alphaxy = tf.clip_by_value(2.0 * sigxy, -20.0, 20.0)

        denominator = tf.maximum(sigxx + sigyy, 1e-6)
        new_e1 = tf.clip_by_value((sigxx - sigyy) / denominator, -0.95, 0.95)
        new_e2 = tf.clip_by_value(2.0 * sigxy / denominator, -0.95, 0.95)

        new_e1 = tf.where(tf.math.is_nan(new_e1), 0.0, new_e1)
        new_e2 = tf.where(tf.math.is_nan(new_e2), 0.0, new_e2)
        new_alphax = tf.where(tf.math.is_nan(new_alphax), 2.0, new_alphax)
        new_alphay = tf.where(tf.math.is_nan(new_alphay), 2.0, new_alphay)
        new_alphaxy = tf.where(tf.math.is_nan(new_alphaxy), 0.0, new_alphaxy)

        return [i + 1, new_alphax, new_alphay, new_alphaxy, new_mux, new_muy,
                curr_sigxx, curr_sigyy, sigxx, sigyy, new_back, new_e1, new_e2]

    i = tf.constant(0)
    e1 = tf.zeros([B])
    e2 = tf.zeros([B])

    loop_result = tf.while_loop(
        cond,
        body,
        loop_vars=[i, alphax, alphay, alphaxy, mux, muy, prev_sigxx, prev_sigyy, curr_sigxx, curr_sigyy, back, e1, e2],
        maximum_iterations=counter_target,
        parallel_iterations=1
    )

    e1 = loop_result[11]
    e2 = loop_result[12]

    e1 = tf.where(tf.logical_or(tf.math.is_nan(e1), tf.math.is_inf(e1)), 0.0, e1)
    e2 = tf.where(tf.logical_or(tf.math.is_nan(e2), tf.math.is_inf(e2)), 0.0, e2)

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
        ker = ker / (tf.reduce_sum(ker) + 1e-6)
        ker = tf.expand_dims(ker, -1)
        img = tf.expand_dims(img, 0)
        blurred = tf.nn.conv2d(img, ker, strides=[1, 1, 1, 1], padding='SAME')
        return tf.squeeze(blurred, axis=0)

    blurred_batch = tf.map_fn(single_convolve, (image, kernel), dtype=tf.float32)
    return blurred_batch

# --- Model Architecture (keeping your existing architecture) ---
@tf.keras.utils.register_keras_serializable()
def conv_block(x, filters, name=None):
    y = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal', name=name)(x)
    y = LeakyReLU(alpha=0.1)(y)
    return y

@tf.keras.utils.register_keras_serializable()
def psf_feature_extractor(psf_input):
    x = Conv2D(16, 3, padding='same', activation='relu')(psf_input)
    x = MaxPooling2D(2)(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)

    psf_global = Dense(32, activation='relu')(x)

    psf_scale1 = Dense(96*96*4, activation='relu')(psf_global)
    psf_scale1 = Reshape((96, 96, 4))(psf_scale1)

    psf_scale2 = Dense(48*48*4, activation='relu')(psf_global)
    psf_scale2 = Reshape((48, 48, 4))(psf_scale2)

    psf_scale3 = Dense(24*24*4, activation='relu')(psf_global)
    psf_scale3 = Reshape((24, 24, 4))(psf_scale3)

    return psf_scale1, psf_scale2, psf_scale3

@tf.keras.utils.register_keras_serializable()
def fusion_block(features_list, filters, name=None):
    if len(features_list) == 1:
        return features_list[0]

    fused = Concatenate(name=f"{name}_concat" if name else None)(features_list)
    fused = Conv2D(filters, 1, padding='same', activation='relu',
                   name=f"{name}_fusion" if name else None)(fused)
    return fused

# --- Build Model with Correct Input Shapes ---
# Now the input shapes match the actual data dimensions
i_blur = Input(shape=(96, 96, 3), name='blurred_input')  # Correct: 3 channels
i_ker = Input(shape=(48, 48, 1), name='kernel_input')
i_wt = Input(shape=(96, 96, 1), name='weights_input')

# Test the channel separation
print("\nTesting channel separation logic:")
test_input = tf.random.normal((2, 96, 96, 3))  # Batch of 2 samples
blurred_orig_test = test_input[:,:,:,0:1]
tikho_reg1_test = test_input[:,:,:,1:2]
tikho_reg2_test = test_input[:,:,:,2:3]

print(f"Original input shape: {test_input.shape}")
print(f"blurred_orig shape: {blurred_orig_test.shape}")
print(f"tikho_reg1 shape: {tikho_reg1_test.shape}")
print(f"tikho_reg2 shape: {tikho_reg2_test.shape}")

# Channel separation - NOW CORRECT
blurred_orig = Lambda(lambda x: x[:,:,:,0:1], name='original_channel')(i_blur)
tikho_reg1 = Lambda(lambda x: x[:,:,:,1:2], name='tikho_reg1')(i_blur)
tikho_reg2 = Lambda(lambda x: x[:,:,:,2:3], name='tikho_reg2')(i_blur)

# Rest of your model architecture remains the same...
psf_feat_96, psf_feat_48, psf_feat_24 = psf_feature_extractor(i_ker)

# Encoder branches
orig_c1 = conv_block(blurred_orig, 16, name='orig_enc1')
orig_p1 = MaxPooling2D(2)(orig_c1)
orig_c2 = conv_block(orig_p1, 32, name='orig_enc2')
orig_p2 = MaxPooling2D(2)(orig_c2)
orig_c3 = conv_block(orig_p2, 64, name='orig_enc3')

tikho1_c1 = conv_block(tikho_reg1, 16, name='tikho1_enc1')
tikho1_p1 = MaxPooling2D(2)(tikho1_c1)
tikho1_c2 = conv_block(tikho1_p1, 32, name='tikho1_enc2')
tikho1_p2 = MaxPooling2D(2)(tikho1_c2)
tikho1_c3 = conv_block(tikho1_p2, 64, name='tikho1_enc3')

tikho2_c1 = conv_block(tikho_reg2, 16, name='tikho2_enc1')
tikho2_p1 = MaxPooling2D(2)(tikho2_c1)
tikho2_c2 = conv_block(tikho2_p1, 32, name='tikho2_enc2')
tikho2_p2 = MaxPooling2D(2)(tikho2_c2)
tikho2_c3 = conv_block(tikho2_p2, 64, name='tikho2_enc3')

# Fusion
fused_c1 = fusion_block([orig_c1, tikho1_c1, tikho2_c1, psf_feat_96], 24, name='fused_enc1')
fused_c2 = fusion_block([orig_c2, tikho1_c2, tikho2_c2, psf_feat_48], 48, name='fused_enc2')
fused_c3 = fusion_block([orig_c3, tikho1_c3, tikho2_c3, psf_feat_24], 96, name='fused_enc3')

# Bridge
bridge = conv_block(fused_c3, 128, name='bridge')

# Decoder
u2 = Conv2DTranspose(48, 2, strides=2, padding='same')(bridge)
u2 = Concatenate()([u2, fused_c2])
c4 = conv_block(u2, 48, name='dec2')

u1 = Conv2DTranspose(24, 2, strides=2, padding='same')(c4)
u1 = Concatenate()([u1, fused_c1])
c5 = conv_block(u1, 24, name='dec1')

output = Conv2D(1, 1, padding='same', activation='linear', name='output')(c5)

# --- Custom Model Class (updated loss function) ---
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

            weighted_mse = tf.reduce_mean(tf.square(y_pred - y_true) * weight_map)

            sharp_true = tf.squeeze(y_true, axis=-1)
            sharp_pred = tf.squeeze(y_pred, axis=-1)

            e1_true, e2_true = compute_ellipticity_batched_tf(sharp_true)
            e1_pred, e2_pred = compute_ellipticity_batched_tf(sharp_pred)

            ellip_diff = tf.reduce_mean((e1_true - e1_pred)**2 + (e2_true - e2_pred)**2)
            ellip_loss = tf.reduce_mean(tf.clip_by_value(ellip_diff, 0.0, 1.0))
            ellip_weight = tf.constant(0.01, dtype=tf.float32) + tf.constant(0.1, dtype=tf.float32) * tf.cast(tf.minimum(self.current_epoch, 4), tf.float32)

            # FIXED: Now correctly extracting channel 0
            original_blurred = blurred_img[:,:,:,0:1]  # Shape: (B, 96, 96, 1)
            reblurred = blur_with_kernel(y_pred, kernel_img)
            print (np.shape(original_blurred), np.shape(reblurred), np.shape(blurred_img))

            reblur_loss = tf.reduce_mean(tf.square(reblurred - original_blurred) * weight_map)

            total_loss = 100*weighted_mse + 100*reblur_loss + ellip_weight*ellip_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_true, y_pred)

        return {
            "loss": total_loss,
            "weighted_mse": weighted_mse,
            "ellipticity_loss": ellip_loss,
            "reblur_loss": reblur_loss
        }

    def test_step(self, data):
        (x, y_true) = data
        blurred_img, kernel_img, weight_map = x
        y_pred = self([blurred_img, kernel_img, weight_map], training=False)

        weighted_mse = tf.reduce_mean(tf.square(y_pred - y_true) * weight_map)

        sharp_true = tf.squeeze(y_true, axis=-1)
        sharp_pred = tf.squeeze(y_pred, axis=-1)

        e1_true, e2_true = compute_ellipticity_batched_tf(sharp_true)
        e1_pred, e2_pred = compute_ellipticity_batched_tf(sharp_pred)

        ellip_diff = tf.reduce_mean((e1_true - e1_pred)**2 + (e2_true - e2_pred)**2)
        ellip_loss = tf.reduce_mean(tf.clip_by_value(ellip_diff, 0.0, 1.0))
        ellip_weight = tf.constant(0.01, dtype=tf.float32) + tf.constant(0.1, dtype=tf.float32) * tf.cast(tf.minimum(self.current_epoch, 4), tf.float32)

        original_blurred = blurred_img[:,:,:,0:1]
        reblurred = blur_with_kernel(y_pred, kernel_img)
        reblur_loss = tf.reduce_mean(tf.square(reblurred - original_blurred) * weight_map)

        total_loss = 100*weighted_mse + 100*reblur_loss + ellip_weight*ellip_loss

        self.compiled_metrics.update_state(y_true, y_pred)

        return {
            "loss": total_loss,
            "weighted_mse": weighted_mse,
            "ellipticity_loss": ellip_loss,
            "reblur_loss": reblur_loss
        }

# Create and compile model
model = WeightedLossModel(inputs=[i_blur, i_ker, i_wt], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=6e-5))

print("Model parameters:")
print(f"Total parameters: {model.count_params():,}")

# --- Training ---
from keras.callbacks import LearningRateScheduler

def lr_schedule(epoch, lr):
    decay_rate = 0.8
    return lr * decay_rate

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

class EpochTracker(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch.assign(epoch)

# Verify input shapes before training
print("\nFinal verification before training:")
print(f"Model expects input shapes: {[inp.shape for inp in model.inputs]}")
print(f"Data shapes: {[train_blur.shape, train_ker.shape, train_wt.shape]}")
print(f"Target shape: {train_sharp.shape}")

# Training
history = model.fit(
    x=[train_blur, train_ker, train_wt],
    y=train_sharp,
    validation_data=([val_blur, val_ker, val_wt], val_sharp),
    epochs=7,
    batch_size=128,
    verbose=1,
    callbacks=[lr_scheduler, EpochTracker()]
)

# Save model
model.save('/scratch/bell/dutta26/psf_datasets/unet_psf_model_bottleNeckFusion_consistency5.keras')

import matplotlib.pyplot as plt

# Plot training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training Loss', marker='o')
plt.plot(val_loss, label='Validation Loss', marker='s')
plt.title('Training and Validation Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.savefig('/scratch/bell/dutta26/psf_datasets/unet_psf_model_bottleNeckFusion_loss_plot_consistency5.png')
plt.close()