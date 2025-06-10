#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 20:58:52 2025

@author: dutta26
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch U-Net PSF Correction Model
Converted from TensorFlow/Keras implementation with detailed comments for PyTorch beginners
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
from typing import Tuple, List

# =============================================================================
# DEVICE SETUP
# =============================================================================
# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



def center_crop(img, crop=96):
    """
    Crop images from center to desired size
    Args:
        img: Input image array
        crop: Target crop size (default 96x96)
    Returns:
        Cropped image
    """
    start = (img.shape[1] - crop) // 2
    return img[:, start:start+crop, start:start+crop, :]

def center_crop_5d(img, crop=96):
    """
    Crop 5D tensor (N, H, W, C, 1) to (N, crop, crop, C)
    This handles the extra dimension in the blurred data
    """
    start = (img.shape[1] - crop) // 2
    cropped = img[:, start:start+crop, start:start+crop, :, 0]  # Remove last dimension
    return cropped


# =============================================================================
# PYTORCH DATASET CLASS
# =============================================================================
class PSFDataset(Dataset):
    """
    Custom PyTorch Dataset class for PSF correction data

    In PyTorch, Dataset classes must implement:
    - __init__: Initialize the dataset
    - __len__: Return the size of the dataset
    - __getitem__: Return one sample from the dataset
    """
    def __init__(self, blurred, kernel, weights, sharp):
        """Initialize the dataset with all data tensors"""
        self.blurred = blurred
        self.kernel = kernel
        self.weights = weights
        self.sharp = sharp

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.blurred)

    def __getitem__(self, idx):
        """
        Return one sample at index idx
        Returns a dictionary with all data for one sample
        """
        return {
            'blurred': self.blurred[idx],
            'kernel': self.kernel[idx],
            'weights': self.weights[idx],
            'sharp': self.sharp[idx]
        }



# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_ellipticity_batched(images: torch.Tensor, counter_target: int = 40,
                               convergence_threshold: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute e1, e2 ellipticities from a batch of images using iterative moment matching
    This preserves the original iterative approach from your TensorFlow code

    Args:
        images: Tensor of shape [B, H, W], batch of images
        counter_target: Maximum number of iterations (default: 40)
        convergence_threshold: Threshold for convergence check (default: 1e-2)
    Returns:
        e1: Tensor of shape [B], ellipticity component 1
        e2: Tensor of shape [B], ellipticity component 2
    """
    B, H, W = images.shape
    device = images.device

    # Create coordinate grids - equivalent to TensorFlow's meshgrid
    y = torch.linspace(0.0, H - 1, H, device=device) - H / 2.0 + 0.5
    x = torch.linspace(0.0, W - 1, W, device=device) - W / 2.0 + 0.5
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Expand to batch dimension
    X = X.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    Y = Y.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)

    # Initialize parameters (conservative initial conditions)
    alphax = torch.ones(B, device=device) * 2.0
    alphay = torch.ones(B, device=device) * 2.0
    alphaxy = torch.zeros(B, device=device)
    mux = torch.zeros(B, device=device)
    muy = torch.zeros(B, device=device)
    back = torch.zeros(B, device=device)

    # Initialize convergence tracking
    prev_sigxx = torch.ones(B, device=device) * 4.0
    prev_sigyy = torch.ones(B, device=device) * 4.0

    # Iterative loop - equivalent to TensorFlow's while_loop
    for iteration in range(counter_target):
        # Clamp parameters to reasonable ranges
        alphax = torch.clamp(alphax, 0.5, 20.0)
        alphay = torch.clamp(alphay, 0.5, 20.0)
        alphaxy = torch.clamp(alphaxy, -10.0, 10.0)

        # Centered coordinates
        Xc = X - mux.unsqueeze(-1).unsqueeze(-1)
        Yc = Y - muy.unsqueeze(-1).unsqueeze(-1)

        # Compute Gaussian weight function parameters
        alpha_denom = torch.clamp(alphax * alphay, min=1e-8)
        alpha_ratio = torch.clamp(alphaxy / alpha_denom, -0.95, 0.95)
        arb_const = torch.clamp(2.0 * (1.0 - alpha_ratio**2), min=1e-8)

        sqrt_term = torch.clamp(1.0 - alpha_ratio**2, min=1e-8)
        A = 1.0 / (2.0 * np.pi * alpha_denom * torch.sqrt(sqrt_term))
        A = torch.clamp(A, 1e-10, 1e10)

        # Compute exponential terms for Gaussian
        alphax_sq = torch.clamp(alphax**2, min=1e-8)
        alphay_sq = torch.clamp(alphay**2, min=1e-8)

        exp_term1 = (Xc**2) / (arb_const * alphax_sq).unsqueeze(-1).unsqueeze(-1)
        exp_term2 = (Yc**2) / (arb_const * alphay_sq).unsqueeze(-1).unsqueeze(-1)
        exp_term3 = 2 * alphaxy.unsqueeze(-1).unsqueeze(-1) * Xc * Yc / (arb_const * alphax_sq * alphay_sq).unsqueeze(-1).unsqueeze(-1)

        exp_term = exp_term1 + exp_term2 - exp_term3
        exp_term = torch.clamp(exp_term, 0.0, 30.0)

        # Gaussian weight function
        k = A.unsqueeze(-1).unsqueeze(-1) * torch.exp(-exp_term)

        # Handle NaN and Inf values
        k = torch.where(torch.isnan(k), torch.zeros_like(k), k)
        k = torch.where(torch.isinf(k), torch.zeros_like(k), k)

        # Background-subtracted image
        img1 = images - back.unsqueeze(-1).unsqueeze(-1)

        # Compute weighted moments
        t1 = torch.sum(X * Y * img1 * k, dim=[1, 2])           # <xy>
        t2 = torch.sum(img1 * k, dim=[1, 2])                   # <I>
        t3 = torch.sum(X * img1 * k, dim=[1, 2])               # <x>
        t4 = torch.sum(Y * img1 * k, dim=[1, 2])               # <y>
        t5 = torch.sum(X * X * img1 * k, dim=[1, 2])           # <x²>
        t6 = torch.sum(Y * Y * img1 * k, dim=[1, 2])           # <y²>
        t7 = torch.sum(k * k, dim=[1, 2])                      # <k²>

        # Safe division
        t2_safe = torch.clamp(torch.abs(t2), min=1e-6)
        t7_safe = torch.clamp(t7, min=1e-6)

        # Update background
        flux_calc = t2 / t7_safe
        total = torch.sum(images, dim=[1, 2])
        image_area = H * W
        new_back = torch.clamp((total - flux_calc) / image_area, -1.0, 1.0)

        # Update centroid
        new_mux = torch.clamp(t3 / t2_safe, -W/2, W/2)
        new_muy = torch.clamp(t4 / t2_safe, -H/2, H/2)

        # Compute second moments
        sigxx = t5 / t2_safe - (t3 / t2_safe)**2
        sigyy = t6 / t2_safe - (t4 / t2_safe)**2
        sigxy = t1 / t2_safe - (t3 * t4) / (t2_safe * t2_safe)

        # Clamp second moments
        sigxx = torch.clamp(sigxx, 0.25, 100.0)
        sigyy = torch.clamp(sigyy, 0.25, 100.0)
        sigxy = torch.clamp(sigxy, -50.0, 50.0)

        # Update alpha parameters
        new_alphax = torch.sqrt(torch.clamp(sigxx * 2.0, 1.0, 400.0))
        new_alphay = torch.sqrt(torch.clamp(sigyy * 2.0, 1.0, 400.0))
        new_alphaxy = torch.clamp(2.0 * sigxy, -20.0, 20.0)

        # Compute ellipticity
        denominator = torch.clamp(sigxx + sigyy, min=1e-6)
        e1 = torch.clamp((sigxx - sigyy) / denominator, -0.95, 0.95)
        e2 = torch.clamp(2.0 * sigxy / denominator, -0.95, 0.95)

        # Handle NaN values
        e1 = torch.where(torch.isnan(e1), torch.zeros_like(e1), e1)
        e2 = torch.where(torch.isnan(e2), torch.zeros_like(e2), e2)
        new_alphax = torch.where(torch.isnan(new_alphax), torch.full_like(new_alphax, 2.0), new_alphax)
        new_alphay = torch.where(torch.isnan(new_alphay), torch.full_like(new_alphay, 2.0), new_alphay)
        new_alphaxy = torch.where(torch.isnan(new_alphaxy), torch.zeros_like(new_alphaxy), new_alphaxy)

        # Check convergence
        if iteration > 0:
            sigxx_diff = torch.abs(sigxx - prev_sigxx)
            sigyy_diff = torch.abs(sigyy - prev_sigyy)

            # Check if converged for all samples
            sigxx_converged = torch.max(sigxx_diff) < convergence_threshold
            sigyy_converged = torch.max(sigyy_diff) < convergence_threshold

            if sigxx_converged and sigyy_converged:
                break

        # Update for next iteration
        alphax = new_alphax
        alphay = new_alphay
        alphaxy = new_alphaxy
        mux = new_mux
        muy = new_muy
        back = new_back
        prev_sigxx = sigxx
        prev_sigyy = sigyy

    return e1, e2

def blur_with_kernel(images: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """
    Blurs a batch of images with corresponding PSF kernels using grouped conv2d,
    handling even-sized kernels to ensure 'same' output shape.

    Args:
        images: (B, 1, H, W)
        kernels: (B, 1, kH, kW)

    Returns:
        Tensor of shape (B, 1, H, W)
    """
    B, _, H, W = images.shape
    _, _, kH, kW = kernels.shape

    # Normalize kernels(NOT NEEDED)
    #kernels = kernels / (kernels.sum(dim=(2, 3), keepdim=True) + 1e-6)
    kernels = torch.flip(kernels, dims=[2, 3])  # flip for true convolution


    # Asymmetric padding for 'same' output (manual padding)
    pad_h = (kH - 1) // 2
    pad_h_extra = kH % 2 == 0  # If even, need +1 extra on the right
    pad_w = (kW - 1) // 2
    pad_w_extra = kW % 2 == 0

    pad = (pad_w, pad_w + pad_w_extra, pad_h, pad_h + pad_h_extra)
    images_padded = F.pad(images, pad, mode='reflect')  # shape: (B, 1, H + ?, W + ?)

    # Reshape for grouped convolution
    images_grouped = images_padded.view(1, B, H + pad[2] + pad[3], W + pad[0] + pad[1])
    kernels_grouped = kernels.view(B, 1, kH, kW)

    # Perform grouped convolution
    blurred = F.conv2d(images_grouped, kernels_grouped, groups=B)

    # Convert back to (B, 1, H, W)
    return blurred.permute(1, 0, 2, 3).contiguous()

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ConvBlock(nn.Module):
    """
    Basic convolutional block with two 3x3 convolutions and LeakyReLU activation
    This is equivalent to the conv_block function in your TensorFlow code
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()  # Initialize parent class

        # Define layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.1)  # Negative slope of 0.1

    def forward(self, x):
        """Forward pass through the block"""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class PSFFeatureExtractor(nn.Module):
    """
    Extract multi-scale features from PSF kernels
    Equivalent to psf_feature_extractor in your TensorFlow code
    """
    def __init__(self):
        super().__init__()

        # Initial convolution layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # Reduce spatial dimensions by factor of 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Pool to 1x1 spatial size

        # Fully connected layers
        self.fc = nn.Linear(32, 32)

        # Different scale projections - create features for different U-Net levels
        self.fc_96 = nn.Linear(32, 96*96*4)  # For 96x96 feature maps
        self.fc_48 = nn.Linear(32, 48*48*4)  # For 48x48 feature maps
        self.fc_24 = nn.Linear(32, 24*24*4)  # For 24x24 feature maps

    def forward(self, x):
        """
        Forward pass to extract multi-scale PSF features

        Args:
            x: PSF kernel tensor (B, 1, 48, 48)
        Returns:
            feat_96, feat_48, feat_24: Multi-scale feature tensors
        """
        # Extract global PSF features
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)  # (B, 32, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B, 32)
        x = F.relu(self.fc(x))    # (B, 32)

        # Generate multi-scale features and reshape
        feat_96 = self.fc_96(x).view(-1, 4, 96, 96)  # (B, 4, 96, 96)
        feat_48 = self.fc_48(x).view(-1, 4, 48, 48)  # (B, 4, 48, 48)
        feat_24 = self.fc_24(x).view(-1, 4, 24, 24)  # (B, 4, 24, 24)

        return feat_96, feat_48, feat_24

class AdaptiveFusionBlock(nn.Module):
    """
    Adaptive fusion block that combines multiple deconvolution features with PSF features
    Uses attention mechanism to weight different deconvolution approaches
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: Number of channels in each deconvolution feature
            out_channels: Number of output channels after fusion
        """
        super().__init__()

        # Attention mechanism: learns to weight the 3 deconvolution approaches
        self.attention_conv = nn.Conv2d(in_channels * 3, 3, kernel_size=1)

        # Final fusion: combines weighted features with PSF features
        self.fusion_conv = nn.Conv2d(in_channels + 4, out_channels, kernel_size=1)  # +4 for PSF features

    def forward(self, deconv_features: List[torch.Tensor], psf_features: torch.Tensor):
        """
        Forward pass for adaptive fusion

        Args:
            deconv_features: List of 3 tensors [orig, tikho1, tikho2] from different deconv approaches
            psf_features: PSF features to incorporate
        Returns:
            Fused feature tensor
        """
        # Concatenate all deconvolution features for attention computation
        concat_features = torch.cat(deconv_features, dim=1)  # (B, 3*in_channels, H, W)

        # Compute attention weights (softmax ensures they sum to 1)
        attention = F.softmax(self.attention_conv(concat_features), dim=1)  # (B, 3, H, W)

        # Apply attention weights to each deconvolution feature
        weighted_features = []
        for i, feat in enumerate(deconv_features):
            # Extract attention weight for this feature
            weight = attention[:, i:i+1, :, :]  # (B, 1, H, W)
            # Apply weight
            weighted_feat = feat * weight  # Element-wise multiplication
            weighted_features.append(weighted_feat)

        # Sum all weighted features
        weighted_sum = sum(weighted_features)  # (B, in_channels, H, W)

        # Concatenate with PSF features and apply final fusion
        fused_input = torch.cat([weighted_sum, psf_features], dim=1)  # (B, in_channels+4, H, W)
        output = F.relu(self.fusion_conv(fused_input))  # (B, out_channels, H, W)

        return output

class PSFUNet(nn.Module):
    """
    Main U-Net model for PSF correction
    Processes 3-channel input (original + 2 Tikhonov regularizations) with PSF guidance
    """
    def __init__(self):
        super().__init__()

        # PSF feature extractor
        self.psf_extractor = PSFFeatureExtractor()

        # Encoder branches - separate processing for each input channel
        # Original blurred image branch
        self.orig_enc1 = ConvBlock(1, 16)  # 96x96 -> 96x96, channels: 1->16
        self.orig_enc2 = ConvBlock(16, 32) # 48x48 -> 48x48, channels: 16->32
        self.orig_enc3 = ConvBlock(32, 64) # 24x24 -> 24x24, channels: 32->64

        # Tikhonov regularization 1 branch
        self.tikho1_enc1 = ConvBlock(1, 16)
        self.tikho1_enc2 = ConvBlock(16, 32)
        self.tikho1_enc3 = ConvBlock(32, 64)

        # Tikhonov regularization 2 branch
        self.tikho2_enc1 = ConvBlock(1, 16)
        self.tikho2_enc2 = ConvBlock(16, 32)
        self.tikho2_enc3 = ConvBlock(32, 64)

        # Pooling layer (shared across all branches)
        self.pool = nn.MaxPool2d(2)  # Reduces spatial dimensions by factor of 2

        # Fusion blocks - combine features from different branches with PSF info
        self.fusion1 = AdaptiveFusionBlock(16, 24)  # Fuse 96x96 level features
        self.fusion2 = AdaptiveFusionBlock(32, 48)  # Fuse 48x48 level features
        self.fusion3 = AdaptiveFusionBlock(64, 96)  # Fuse 24x24 level features

        # Bridge - deepest part of U-Net
        self.bridge = ConvBlock(96, 128)

        # Decoder - upsampling path of U-Net
        self.up1 = nn.ConvTranspose2d(128, 48, kernel_size=2, stride=2)  # Upsample 24x24 -> 48x48
        self.dec1 = ConvBlock(48 + 48, 48)  # +48 from skip connection

        self.up2 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)   # Upsample 48x48 -> 96x96
        self.dec2 = ConvBlock(24 + 24, 24)  # +24 from skip connection

        # Output layer - final 1x1 conv to get single channel output
        self.output_conv = nn.Conv2d(24, 1, kernel_size=1)

    def forward(self, blurred, kernel, weights):
        """
        Forward pass through the U-Net

        Args:
            blurred: Blurred input images (B, 3, 96, 96) - 3 channels for different deconv approaches
            kernel: PSF kernels (B, 1, 48, 48)
            weights: Pixel weights (B, 1, 96, 96) - not used in forward pass, only in loss
        Returns:
            Deblurred image (B, 1, 96, 96)
        """
        # Extract the 3 different channels from blurred input
        blurred_orig = blurred[:, 0:1, :, :]  # Original blurred (B, 1, 96, 96)
        tikho_reg1 = blurred[:, 1:2, :, :]    # Tikhonov reg 1 (B, 1, 96, 96)
        tikho_reg2 = blurred[:, 2:3, :, :]    # Tikhonov reg 2 (B, 1, 96, 96)

        # Extract multi-scale PSF features
        psf_96, psf_48, psf_24 = self.psf_extractor(kernel)

        # === ENCODER PATH ===
        # Process original blurred image
        orig_c1 = self.orig_enc1(blurred_orig)  # (B, 16, 96, 96)
        orig_p1 = self.pool(orig_c1)            # (B, 16, 48, 48)
        orig_c2 = self.orig_enc2(orig_p1)       # (B, 32, 48, 48)
        orig_p2 = self.pool(orig_c2)            # (B, 32, 24, 24)
        orig_c3 = self.orig_enc3(orig_p2)       # (B, 64, 24, 24)

        # Process Tikhonov regularization 1
        tikho1_c1 = self.tikho1_enc1(tikho_reg1)  # (B, 16, 96, 96)
        tikho1_p1 = self.pool(tikho1_c1)          # (B, 16, 48, 48)
        tikho1_c2 = self.tikho1_enc2(tikho1_p1)   # (B, 32, 48, 48)
        tikho1_p2 = self.pool(tikho1_c2)          # (B, 32, 24, 24)
        tikho1_c3 = self.tikho1_enc3(tikho1_p2)   # (B, 64, 24, 24)

        # Process Tikhonov regularization 2
        tikho2_c1 = self.tikho2_enc1(tikho_reg2)  # (B, 16, 96, 96)
        tikho2_p1 = self.pool(tikho2_c1)          # (B, 16, 48, 48)
        tikho2_c2 = self.tikho2_enc2(tikho2_p1)   # (B, 32, 48, 48)
        tikho2_p2 = self.pool(tikho2_c2)          # (B, 32, 24, 24)
        tikho2_c3 = self.tikho2_enc3(tikho2_p2)   # (B, 64, 24, 24)

        # === FUSION PATH ===
        # Fuse features from all three branches at each scale with PSF features
        fused_c1 = self.fusion1([orig_c1, tikho1_c1, tikho2_c1], psf_96)  # (B, 24, 96, 96)
        fused_c2 = self.fusion2([orig_c2, tikho1_c2, tikho2_c2], psf_48)  # (B, 48, 48, 48)
        fused_c3 = self.fusion3([orig_c3, tikho1_c3, tikho2_c3], psf_24)  # (B, 96, 24, 24)

        # Bridge
        bridge = self.bridge(fused_c3)

        # Decoder
        up1 = self.up1(bridge)
        up1 = torch.cat([up1, fused_c2], dim=1)
        dec1 = self.dec1(up1)

        up2 = self.up2(dec1)
        up2 = torch.cat([up2, fused_c1], dim=1)
        dec2 = self.dec2(up2)

        # Output
        output = self.output_conv(dec2)

        return output

# =============================================================================
# CUSTOM LOSS FUNCTION
# =============================================================================

class PSFLoss(nn.Module):
    """
    Custom loss function for PSF correction that combines multiple loss components:
    1. Weighted MSE loss - measures pixel-wise reconstruction accuracy with importance weighting
    2. Reblur loss - ensures consistency by re-blurring prediction and comparing to original
    3. Ellipticity loss - preserves astronomical shape measurements (galaxy ellipticity)
    """
    def __init__(self, ellip_weight=0.01):
        """
        Initialize the PSF loss function

        Args:
            ellip_weight: Base weight for ellipticity loss component (default: 0.01)
        """
        super().__init__()
        self.ellip_weight = ellip_weight

    def forward(self, pred, target, blurred, kernel, weights, epoch=0):
        """
        Compute the total loss and its components

        Args:
            pred: Model prediction (B, 1, 96, 96) - deblurred image
            target: Ground truth sharp image (B, 1, 96, 96)
            blurred: Original blurred input (B, 3, 96, 96) - 3 channels for different approaches
            kernel: PSF kernel (B, 1, 48, 48)
            weights: Pixel importance weights (B, 1, 96, 96)
            epoch: Current training epoch (used for adaptive ellipticity weighting)

        Returns:
            Dictionary containing total loss and individual loss components
        """
        # === 1. WEIGHTED MSE LOSS ===
        # Measures how well the prediction matches the target, weighted by pixel importance
        # Higher weights = more important pixels (e.g., galaxy centers)
        weighted_mse = torch.mean((pred - target) ** 2 * weights)

        # === 2. REBLUR LOSS ===
        # Consistency check: blur the predicted sharp image and compare to original blurred image
        # This ensures the deblurred result is physically consistent with the input
        original_blurred = blurred[:, 0:1, :, :]  # Extract first channel (original blurred image)
        reblurred = blur_with_kernel(pred, kernel)  # Re-blur prediction with PSF kernel
        reblur_loss = torch.mean((reblurred - original_blurred) ** 2 * weights)

        # === 3. ELLIPTICITY LOSS ===
        # Preserves astronomical shape measurements - critical for galaxy analysis
        if self.ellip_weight > 0:
            # Remove channel dimension for ellipticity computation
            pred_squeezed = pred.squeeze(1)    # (B, 96, 96)
            target_squeezed = target.squeeze(1)  # (B, 96, 96)

            # Compute ellipticity parameters (e1, e2) for both prediction and target
            e1_pred, e2_pred = compute_ellipticity_batched(pred_squeezed)
            e1_true, e2_true = compute_ellipticity_batched(target_squeezed)

            # L2 loss between predicted and true ellipticity components
            ellip_loss = torch.mean((e1_pred - e1_true) ** 2 + (e2_pred - e2_true) ** 2)

            # Adaptive weighting: increase ellipticity importance over first few epochs
            # This allows the model to first learn basic deblurring before focusing on shape preservation
            ellip_weight = self.ellip_weight + 0.1 * min(epoch, 4)
        else:
            ellip_loss = torch.tensor(0.0, device=pred.device)
            ellip_weight = 0.0

        # === TOTAL LOSS COMBINATION ===
        # Scale factors chosen to balance the different loss components:
        # - 100x for MSE and reblur losses (main reconstruction objectives)
        # - Adaptive weight for ellipticity (shape preservation)
        total_loss = 100 * weighted_mse + 100 * reblur_loss + ellip_weight * ellip_loss

        return {
            'total_loss': total_loss,
            'weighted_mse': weighted_mse,
            'reblur_loss': reblur_loss,
            'ellip_loss': ellip_loss
        }

# =============================================================================
# TRAINING FUNCTION WITH DETAILED MONITORING
# =============================================================================

def train_model(model, train_loader, val_loader, num_epochs=7, lr=6e-5):
    """
    Train the PSF correction model with comprehensive loss monitoring

    Args:
        model: PSFUNet model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs (default: 7)
        lr: Learning rate (default: 6e-5)

    Returns:
        train_losses, val_losses: Lists of average losses per epoch
    """
    # === OPTIMIZER AND SCHEDULER SETUP ===
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = PSFLoss()
    # Learning rate decay: reduce by 20% each epoch for stable convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    # Loss tracking for plotting
    train_losses = []
    val_losses = []

    # Import tqdm for progress bars
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        print("tqdm not available, using basic progress tracking")
        use_tqdm = False

    print(f"Starting training for {num_epochs} epochs...")
    print("=" * 80)

    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch+1}/{num_epochs}")
        print("-" * 50)

        # === TRAINING PHASE ===
        model.train()  # Set model to training mode (enables dropout, batch norm updates)

        # Initialize loss accumulators for detailed monitoring
        train_loss_sum = 0
        train_mse_sum = 0
        train_reblur_sum = 0
        train_ellip_sum = 0
        train_batches = 0

        # Create progress bar for training batches
        if use_tqdm:
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}",
                            leave=False, ncols=150)
        else:
            train_pbar = train_loader

        for batch_idx, batch in enumerate(train_pbar):
            # Move data to GPU/CPU
            blurred = batch['blurred'].to(device)
            kernel = batch['kernel'].to(device)
            weights = batch['weights'].to(device)
            sharp = batch['sharp'].to(device)

            # === FORWARD PASS ===
            optimizer.zero_grad()  # Clear gradients from previous iteration
            pred = model(blurred, kernel, weights)  # Get model prediction
            loss_dict = criterion(pred, sharp, blurred, kernel, weights, epoch)

            # === BACKWARD PASS ===
            loss_dict['total_loss'].backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            # === LOSS TRACKING ===
            train_loss_sum += loss_dict['total_loss'].item()
            train_mse_sum += loss_dict['weighted_mse'].item()
            train_reblur_sum += loss_dict['reblur_loss'].item()
            train_ellip_sum += loss_dict['ellip_loss'].item()
            train_batches += 1

            # Update progress bar with current batch losses
            if use_tqdm:
                train_pbar.set_postfix({
                    'Loss': f"{loss_dict['total_loss'].item():.4f}",
                    'MSE': f"{loss_dict['weighted_mse'].item():.6f}",
                    'Reblur': f"{loss_dict['reblur_loss'].item():.6f}",
                    'Ellip': f"{loss_dict['ellip_loss'].item():.6f}"
                })

        # === VALIDATION PHASE ===
        model.eval()  # Set model to evaluation mode (disables dropout, batch norm updates)

        # Initialize validation loss accumulators
        val_loss_sum = 0
        val_mse_sum = 0
        val_reblur_sum = 0
        val_ellip_sum = 0
        val_batches = 0

        # Create progress bar for validation batches
        if use_tqdm:
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}",
                          leave=False, ncols=150)
        else:
            val_pbar = val_loader

        with torch.no_grad():  # Disable gradient computation for validation (saves memory)
            for batch in val_pbar:
                # Move data to GPU/CPU
                blurred = batch['blurred'].to(device)
                kernel = batch['kernel'].to(device)
                weights = batch['weights'].to(device)
                sharp = batch['sharp'].to(device)

                # Forward pass only (no gradient computation)
                pred = model(blurred, kernel, weights)
                loss_dict = criterion(pred, sharp, blurred, kernel, weights, epoch)

                # Accumulate validation losses
                val_loss_sum += loss_dict['total_loss'].item()
                val_mse_sum += loss_dict['weighted_mse'].item()
                val_reblur_sum += loss_dict['reblur_loss'].item()
                val_ellip_sum += loss_dict['ellip_loss'].item()
                val_batches += 1

                # Update validation progress bar
                if use_tqdm:
                    val_pbar.set_postfix({
                        'Val Loss': f"{loss_dict['total_loss'].item():.4f}",
                        'Val MSE': f"{loss_dict['weighted_mse'].item():.6f}",
                        'Val Reblur': f"{loss_dict['reblur_loss'].item():.6f}",
                        'Val Ellip': f"{loss_dict['ellip_loss'].item():.6f}"
                    })

        # === COMPUTE EPOCH AVERAGES ===
        avg_train_loss = train_loss_sum / train_batches
        avg_train_mse = train_mse_sum / train_batches
        avg_train_reblur = train_reblur_sum / train_batches
        avg_train_ellip = train_ellip_sum / train_batches

        avg_val_loss = val_loss_sum / val_batches
        avg_val_mse = val_mse_sum / val_batches
        avg_val_reblur = val_reblur_sum / val_batches
        avg_val_ellip = val_ellip_sum / val_batches

        # Store for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # === EPOCH SUMMARY ===
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Training   - Total: {avg_train_loss:.6f} | MSE: {avg_train_mse:.6f} | "
              f"Reblur: {avg_train_reblur:.6f} | Ellip: {avg_train_ellip:.6f}")
        print(f"  Validation - Total: {avg_val_loss:.6f} | MSE: {avg_val_mse:.6f} | "
              f"Reblur: {avg_val_reblur:.6f} | Ellip: {avg_val_ellip:.6f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Update learning rate
        scheduler.step()

        # === MEMORY CLEANUP ===
        # Important for long training runs to prevent GPU memory leaks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    return train_losses, val_losses

# =============================================================================
# MAIN TRAINING EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # =============================================================================
    # DATA LOADING AND PREPROCESSING
    # =============================================================================
    # Load the dataset (same as original TensorFlow version)
    data = np.load("/scratch/bell/dutta26/wiyn_sim/train_data.npz")
    blurred = data["blurred"]      # (N,100,100,3,1) - Input blurred images with 3 channels
    kernel = data["psf"]          # (N,48,48,1) - Point Spread Function kernels
    weights = data["weight"]       # (N,100,100,1) - Pixel weights for loss computation
    sharp = data["sharp"]         # (N,100,100,1) - Target sharp images

    print("Original data shapes:")
    print(f"blurred: {blurred.shape}")
    print(f"kernel: {kernel.shape}")
    print(f"weights: {weights.shape}")
    print(f"sharp: {sharp.shape}")
    
    
    # Apply cropping to make data compatible with U-Net architecture
    blurred = center_crop_5d(blurred, crop=96)  # (N, 96, 96, 3)
    weights = center_crop(weights, crop=96)     # (N, 96, 96, 1)
    sharp = center_crop(sharp, crop=96)         # (N, 96, 96, 1)
    # kernel remains (N, 48, 48, 1)

    print("\nAfter cropping:")
    print(f"blurred: {blurred.shape}")
    print(f"kernel: {kernel.shape}")
    print(f"weights: {weights.shape}")
    print(f"sharp: {sharp.shape}")

    # Convert NumPy arrays to PyTorch tensors and change to channel-first format
    # PyTorch uses (N, C, H, W) while TensorFlow uses (N, H, W, C)
    blurred = torch.from_numpy(blurred).permute(0, 3, 1, 2).float()  # (N, 3, 96, 96)
    kernel = torch.from_numpy(kernel).permute(0, 3, 1, 2).float()   # (N, 1, 48, 48)
    weights = torch.from_numpy(weights).permute(0, 3, 1, 2).float() # (N, 1, 96, 96)
    sharp = torch.from_numpy(sharp).permute(0, 3, 1, 2).float()     # (N, 1, 96, 96)

    # Split data into training and validation sets
    train_blur, val_blur, train_ker, val_ker, train_sharp, val_sharp, train_wt, val_wt = train_test_split(
        blurred, kernel, sharp, weights, test_size=0.2, random_state=42
    )

    print("\nTraining data shapes:")
    print(f"train_blur: {train_blur.shape}, val_blur: {val_blur.shape}")
    print(f"train_ker: {train_ker.shape}, val_ker: {val_ker.shape}")
    print(f"train_wt: {train_wt.shape}, val_wt: {val_wt.shape}")
    print(f"train_sharp: {train_sharp.shape}, val_sharp: {val_sharp.shape}")

    # Create dataset instances
    train_dataset = PSFDataset(train_blur, train_ker, train_wt, train_sharp)
    val_dataset = PSFDataset(val_blur, val_ker, val_wt, val_sharp)


    
    print("Setting up data loaders...")
    # Create data loaders with optimized settings
    # num_workers=4 for parallel data loading (adjust based on your CPU cores)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    print("Initializing model...")
    # Create and move model to appropriate device (GPU if available)
    model = PSFUNet().to(device)

    # Display model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")

    print(f"\nStarting training on {device}...")
    # Train the model with detailed monitoring
    train_losses, val_losses = train_model(model, train_loader, val_loader)

    print("\nSaving trained model...")
    # Save model state dictionary (weights only, more efficient)
    model_path = '/scratch/bell/dutta26/psf_datasets/pytorch_unet_psf_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    print("Generating training history plot...")
    # Create and save training loss visualization
    plt.figure(figsize=(12, 8))

    # Main loss plot
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             label='Training Loss', marker='o', linewidth=2, markersize=6)
    plt.plot(range(1, len(val_losses) + 1), val_losses,
             label='Validation Loss', marker='s', linewidth=2, markersize=6)
    plt.title('PSF U-Net Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Log scale plot for better visualization if losses vary greatly
    plt.subplot(2, 1, 2)
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             label='Training Loss', marker='o', linewidth=2, markersize=6)
    plt.plot(range(1, len(val_losses) + 1), val_losses,
             label='Validation Loss', marker='s', linewidth=2, markersize=6)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plot_path = '/scratch/bell/dutta26/psf_datasets/pytorch_unet_psf_loss_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to: {plot_path}")

    # Print final training summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Final Training Loss:   {train_losses[-1]:.6f}")
    print(f"Final Validation Loss: {val_losses[-1]:.6f}")
    print(f"Best Validation Loss:  {min(val_losses):.6f} (Epoch {val_losses.index(min(val_losses)) + 1})")
    print(f"Total Training Time:   {len(train_losses)} epochs")
    print("=" * 80)

    print("Training completed successfully!")


