#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:48:09 2025

@author: dutta26
"""

import numpy as np
from astropy.io import fits
import helper_phosim, helper, os
import subprocess,sys,os,shutil
import matplotlib.pyplot as plt

phosimLoc = '/home/dutta26/Downloads/phosim_release/'
sex_loc = '/home/dutta26/Downloads/sextractor-master/src/'
filt = 'i'#str(sys.argv[1])
idNo = 1035
noImages = 1

for j in range(int(idNo), int(idNo)+noImages):
    
    #For normal images
    outLoc = '/scratch/bell/dutta26/wiyn_sim/'+filt+'/'+str(j)+"/"
    
    sexCommand = './sex '+ outLoc+'output.fits'
    process = subprocess.Popen(sexCommand.split(), stdout=subprocess.PIPE, cwd=sex_loc)
    output, error = process.communicate()

    
    #Read the sextractor catalog 
    f=open('/home/dutta26/Downloads/sextractor-master/src/test.cat')
    content = f.readlines()
    tot_len = len(content)
    f.close()
    
    #Read the image
    f=fits.open(outLoc+'output.fits')
    img = f[0].data
    f.close()
    
    flux_arr = []
    size_arr= []
    
    x_arr = []
    y_arr =[]
    #Measure the sources 
    for j in range(len(content)):
        temp = content[j].split()
        if temp[0] == "#":
            continue
        flux = float(temp[1])
        if flux < 1:
            continue
        x = int(float(temp[2]))
        y = int(float(temp[3]))
        x_arr.append(x)
        y_arr.append(y)
       
        cutout = img[y-50:y+50, x-50:x+50]
        flux, mux, muy, e1, e2, bkg, size, sigxx, sigyy, sigxy = helper.measure_new(cutout, [], [])
        
        if(size == None or np.isnan(size) or size<0 or size>15 or flux<1):
            continue
        size_arr.append(size)
        flux_arr.append(flux)
        
    
    
    
    
    #Plot them 
    plt.plot(size_arr, flux_arr, 'r.', markersize = 1)
    plt.xlim(0,10)
    plt.xlabel('Size')
    plt.ylabel('Counts')
    print (outLoc)
    

    #Convert x and y to np arrays
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    
    
    #Make a numpy array if there exits a star limit file. Start = 1, glalaxy = 0
    files = os.listdir(outLoc)
    if 'limits.txt' not in files:
        plt.savefig(outLoc+"/size_vs_counts.png")
        plt.close()
        continue
    
    f=open(outLoc+'/limits.txt')
    content_lt = f.readlines()
    f.close()
    
    flux_low , flux_high = float(content_lt[0].split()[0]), float(content_lt[0].split()[1])
    size_low , size_high = float(content_lt[1].split()[0]), float(content_lt[1].split()[1])
    
    plt.plot([size_low, size_high], [flux_high, flux_high], 'k-',markersize= 2)
    plt.plot([size_low, size_high], [flux_low, flux_low], 'k-',markersize= 2)
    plt.plot([size_low, size_low], [flux_low, flux_high], 'k-',markersize= 2)
    plt.plot([size_high, size_high], [flux_low, flux_high], 'k-',markersize= 2)

    
    plt.savefig(outLoc+"/size_vs_counts.png")
    plt.close()
    store = np.zeros((tot_len, 8), dtype = np.float32)
   
    count = 0 
    star_count = 0
    #Measure the sources 
    for j in range(len(content)):
        temp = content[j].split()
        if temp[0] == "#":
            continue
        flux = float(temp[1])
        if flux < 1:
            continue
        x = int(round(float(temp[2])))
        y = int(round(float(temp[3])))
        ra = float(temp[4])
        dec = float(temp[5])
        
        cutout = img[y-50:y+50, x-50:x+50]
        flux, mux, muy, e1, e2, bkg, size, sigxx, sigyy, sigxy = helper.measure_new(cutout, [], [])
        
        if(size == None or np.isnan(size) or size<0 or size>15 or flux<1):
            continue
        
        star_bool = 0
        if size>size_low and size<size_high and flux>flux_low and flux<flux_high:
            #Check if any source is blended or too close to the star ie withint +=50 pixles
            dist = np.sqrt((x_arr-x)**2 + (y_arr-y)**2)
            dist = np.sort(dist)
            #print (dist[0:10])
            if dist[1]>35:
                star_bool = 1
                star_count += 1

            #temp = temp[temp[:,2].argsort()]
        
        store[count, 0:8 ] = x, y , sigxx, sigyy, sigxy , star_bool, ra , dec
        count += 1
    print (count, star_count)
    
    store = store[0: count, :]
    np.save(outLoc+"/star_gal.npy", store)
    
    #Copy the test.fits as mask.fits 
    shutil.copyfile('/scratch/bell/dutta26/wiyn_sim/test.fits', outLoc+'/mask.fits')
    
    #Open mask .fits and set any values >0 = 1. Anything else to 0.1
    f=fits.open(outLoc+'/mask.fits', mode = 'update')
    data = f[0].data
    data[data>0] = 1
    data[data<=0] = 0
    data += 0.02
    f[0].data = data
    f.flush()
    
    

        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    