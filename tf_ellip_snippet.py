#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 12:27:43 2025

@author: dutta26
"""
import numpy as np
import tensorflow as tf
import helper
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

    return e1, e2


def center_crop(img, crop=96):
    start = (img.shape[1] - crop) // 2
    return img[:, start:start+crop, start:start+crop, :]


def center_crop(img, crop=96):
    start = (img.shape[1] - crop) // 2
    return img[:, start:start+crop, start:start+crop, :]

# Example usage (assuming data loading works):
data = np.load("/scratch/bell/dutta26/wiyn_sim/test_data_2k.npz")
blurred = center_crop(data["blurred"])     # (N,96,96,1)
# kernel  = data["psf"]                      # (N,10,10,1)
# weights = center_crop(data["weight"])      # (N,96,96,1)
sharp   = center_crop(data["sharp"])       # (N,96,96,1)
a = tf.squeeze(blurred[5:6]-3.637, axis=-1)
e1, e2 = compute_ellipticity_batched_tf(a)
b = ((e1)**2 + (e2)**2)