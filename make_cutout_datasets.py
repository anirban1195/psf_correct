#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:30:08 2025

@author: dutta26
"""

import numpy as np
from astropy.io import fits
import helper_phosim, helper, os
import subprocess,sys
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
import tikho_deconv


def makePSF(arr, blurred_img, n, large = 0 ):
    if(large == 0):
        temp = np.zeros((48,48), dtype = np.float32)
    else:
        temp = np.zeros((60,60), dtype = np.float32)
    tot = 0
    for j in range(n):
        x, y = int(arr[j,0]), int(arr[j,1])
        dist = arr[j,2]
        tot += dist
        if(large == 0):
            temp += dist*blurred_img[y-24:y+24, x-24:x+24]
        else:
            temp += dist*blurred_img[y-30:y+30, x-30:x+30]
    return temp/tot 
        
def normalize (img):
    img[img>1000] = 1000
    min_val = img.min()
    max_val = img.max()
    return (img)/(max_val - min_val)



def normalize_all_by_blurred(blurred_img, sharp_img, tikho_img1, tikho_img2 ,eps=1e-6):
    # Compute percentiles over the entire 2D image
    vmin = np.min(blurred_img)#np.percentile(blurred_img, 0.05)
    vmax = np.max(blurred_img)#np.percentile(blurred_img, 99.95)
    scale = vmax - vmin + eps

    # Normalize and clip
    blurred_norm   = np.clip((blurred_img   - vmin) / scale, 0.0, 5.0)
    sharp_norm     = np.clip((sharp_img     - vmin) / scale, 0.0, 5.0)
    tikho_norm1 = np.clip((tikho_img1     - vmin) / scale, 0.0, 5.0)
    tikho_norm2 = np.clip((tikho_img2     - vmin) / scale, 0.0, 5.0)
    

    return blurred_norm, sharp_norm, tikho_norm1, tikho_norm2

    
def makeWt(mux_calc, muy_calc, alphax, alphay, alphaxy):
    sizex = sizey = 100
    x = np.linspace(0, sizex-1, sizex)
    y = np.linspace(0, sizey-1, sizey)
    x= x -sizex/2.0 + 0.5 
    y= y -sizey/2.0 + 0.5 
    #mux=muy= np.random.normal(29.5, 0.15)
    x, y = np.meshgrid(x, y)
    arbConst = 2*(1- (alphaxy/(alphax*alphay))**2 )  
    A =1/(2*np.pi*alphax*alphay*np.sqrt(1- (alphaxy/(alphax*alphay))**2 ))
    k= (A * np.exp(-((x-mux_calc)**2/(arbConst*alphax**2)+ (y-muy_calc)**2/(arbConst*alphay**2) - 2*alphaxy*(y-muy_calc)*(x-mux_calc)/(arbConst* alphax**2 * alphay**2 ) )))
    return k



filt = 'i' #str(sys.argv[1])
idNo = 1020
noImages = 6


for j in range(int(idNo), int(idNo)+noImages):
    
    #For normal images
    outLoc = '/scratch/bell/dutta26/wiyn_sim/'+filt+'/'+str(j)
    
    #First read the numpy array the image and the sharp image 
    store = np.load(outLoc+"/star_gal.npy")
    
    
    f=fits.open(outLoc+'/output.fits')
    blurred_img = f[0].data
    f.close()
    mean_blur, median_blur, std_blur = sigma_clipped_stats(blurred_img)
    
    f=fits.open(outLoc+'_sharp/output.fits')
    sharp_img = f[0].data
    f.close()
    mean_sharp, median_sharp, std_sharp = sigma_clipped_stats(blurred_img)

    
    f=fits.open(outLoc+'/mask.fits')
    mask_img = f[0].data
    f.close()
    
    
    #Get a list of of the stars
    l=len(np.where(store[:,5] == 1)[0])
    star_arr = np.zeros((l, 3), dtype = np.float32)
    count = 0
    #Now loop over all sources 
    for j in range(len(store)):
        if(store[j,5] == 0):
            continue
        x_b = store[j,0]
        y_b = store[j,1]
        star_arr[count,0:2] = x_b, y_b
        count += 1
        
    
    print (np.shape(star_arr))
    count = 0
    blurred_arr = np.zeros((len(store), 100, 100,3), dtype= np.float32)
    sharp_arr = np.zeros((len(store), 100, 100), dtype= np.float32)
    psf_arr = np.zeros((len(store), 48, 48), dtype= np.float32)
    psf_arr_large = np.zeros((len(store), 60, 60), dtype= np.float32)
    wt_arr = np.zeros((len(store), 100, 100), dtype= np.float32)
    
    #Now loop over all sources 
    for j in range(len(store)):
        
        #If a star i.e index 5= 1 or x and y is not defined i.e 0 then continue
        if(store[j,5] == 1 or store[j,0] == 0 or store[j,1] == 0):
            continue
        
        
        x_b = store[j,0]
        y_b = store[j,1]
        x_s, y_s = x_b, y_b
        #Make 100x100 cutout
        blurred_cut = blurred_img[int(y_b)-50: int(y_b)+50, int(x_b)-50: int(x_b)+50]
        sharp_cut = sharp_img[int(y_s)-50: int(y_s)+50, int(x_s)-50: int(x_s)+50]
        
        if 0.0 in blurred_cut or 0.0 in sharp_cut:
            continue
        
        #Find the PSF at the blurred loc
        temp = np.copy(star_arr)
        temp[:,2] = np.sqrt((temp[:,0]-x_b)**2 + (temp[:,1]-y_b)**2)
        temp = temp[temp[:,2].argsort()]
        psf = makePSF(temp, blurred_img, 10)
        psf_large = makePSF(temp, blurred_img, 10, 1)
        
        
        
        
        #Make the weights
        wt = mask_img[int(y_b)-50: int(y_b)+50, int(x_b)-50: int(x_b)+50]#makeWt(0 , 0 , np.sqrt(2*store[j,2]),np.sqrt(2*store[j,3]), 2*store[j,5])
        
        
        #Put everything in arrays 
        blurred_arr[count,:,:,0] = blurred_cut*2
        sharp_arr[count,:,:] = sharp_cut
        psf_arr[count,:,:] = psf/np.sum(psf)   #Normalize PSF to 1
        psf_arr_large[count,:,:] = psf_large/np.sum(psf_large) ##Normalize PSF to 1
        wt_arr[count,:,:] = wt
        
        
        #Deconvolve using tikho and add to array 
        padded_psf = np.random.normal(0,1e-5, (100,100))
        padded_psf[26:74, 26:74] += psf_arr[count,:,:]
        padded_psf /= np.sum(padded_psf)
        blurred_arr[count,:,:,1] = tikho_deconv.apply_tikhonov_deconv(blurred_arr[count:count+1,:,:,0], padded_psf, 0.1)[0,:,:]
        blurred_arr[count,:,:,2] = tikho_deconv.apply_tikhonov_deconv(blurred_arr[count:count+1,:,:,0], padded_psf, 0.01)[0,:,:]
        
        #Normalize the blurred, sharp, tikho1 and tikho2
        blurred_arr[count,:,:,0],sharp_arr[count,:,:],blurred_arr[count,:,:,1],blurred_arr[count,:,:,2] = normalize_all_by_blurred(blurred_arr[count,:,:,0],sharp_arr[count,:,:],blurred_arr[count,:,:,1],blurred_arr[count,:,:,2])
    
        count += 1
    print (count)
        
        
    
    #Resize the array since we dont need all the sapce 
    blurred_arr = blurred_arr[0:count,:,:]
    sharp_arr = sharp_arr[0:count,:,:]
    psf_arr = psf_arr[0:count,:,:]
    psf_arr_large = psf_arr_large[0:count,:,:]
    wt_arr = wt_arr[0:count,:,:]
    
    #Save array 
    np.save(outLoc+"/blurred.npy", blurred_arr)
    np.save(outLoc+"/sharp.npy", sharp_arr)
    np.save(outLoc+"/weight.npy", wt_arr)
    np.save(outLoc+"/psf.npy", psf_arr)
    np.save(outLoc+"/psf_large.npy", psf_arr_large)
    
        
    
    
    
    
    
    
    
    
    
    
    