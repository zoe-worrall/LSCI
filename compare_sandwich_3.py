#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:44:08 2023

@author: zworrall
"""

#%% IMPORTING NECESSARY FILES

from array import *
import numpy as np
import numpy.fft
from PIL import Image
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import norm
import scipy.fftpack
from scipy.fftpack import fft
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#%% CALCULATE SPECKLE CONTRAST

# a method that returns a tuple;
    # first value is the average speckle contrast of the image
    # second value is the matrix passed through a speckle contrast algorithm
def calcSpeckleContrast(photo):
    
    window = photo[:][:]
    w_dev = np.std(window)
    w_meanInt = np.mean(window)
    w_speckle_contrast = w_dev / w_meanInt 
    
    # the selection of SQUARE x SQUARE pixels where speckle will be calculated
    SQUARE = 7
    
    # convert photo into an np array
    pArr = np.asarray(photo)
    
    # the speckle contrast array that will be returned
    contrast_array = [ [0]*(len(pArr[0])-SQUARE) for i in range((len(pArr))-SQUARE) ]
    conMin = np.inf;
    
    # The Loop Does the Following:
    # goes through 7x7 selections of the image
    # computes the speckle contrast for this region
    # using this comparison, create a new matrix with values whose magnitude
        # decreases depending on how "blurry" the selection is
    for r in range(len(pArr) - SQUARE):
        for c in range(len(pArr[0]) - SQUARE):
            x1 = r; x2 = r+SQUARE-1
            y1 = c; y2 = c+SQUARE-1
            selection = pArr[x1:x2, y1:y2]
            
            s_dev = np.std(selection)
            s_meanInt = np.mean(selection)
            
            # to cut out values outside of the speckle caught on camera
            if (s_meanInt <= 0.1 and s_dev <= 0.1):
                contrast_array[r][c] = np.NaN
            else:
                s_speckle_contrast = s_dev / s_meanInt
                
                if s_speckle_contrast < conMin and s_speckle_contrast != 0:
                    conMin = s_speckle_contrast
                    
                contrast_array[r][c] = s_speckle_contrast
    
    conArr = np.asarray(contrast_array)
    conArr[conArr == 0] = conMin
    
    return (w_speckle_contrast, contrast_array)

#%% SET UP IMAGE AND INTENSITY MATRIX FOR FOUR SPEEDS

link120 = "/Users/zworrall/Desktop/coding/august_11/fin/fin_1515_120.bmp"
link60 = "/Users/zworrall/Desktop/coding/august_11/fin/fin_1515_060.bmp"
link10 ="/Users/zworrall/Desktop/coding/august_11/fin/fin_1515_010.bmp"
link0 = "/Users/zworrall/Desktop/coding/august_11/fin/fin_1515_000.bmp"

image = plt.imread(link120, 0)
y1 = 550; y2 = 1650
x1 = 900; x2 = 2000
intensity = np.asarray(image[y1:y2, x1:x2])
(cont, contrast_array120) = calcSpeckleContrast(intensity)
print("120 done")

image = plt.imread(link60, 0)
intensity = np.asarray(image[y1:y2, x1:x2])
(cont, contrast_array60) = calcSpeckleContrast(intensity)
print("60 done")

image = plt.imread(link10, 0)
intensity = np.asarray(image[y1:y2, x1:x2])
(cont, contrast_array10) = calcSpeckleContrast(intensity)
print("10 done")

image = plt.imread(link0, 0)
intensity = np.asarray(image[y1:y2, x1:x2])
(cont, contrast_array0) = calcSpeckleContrast(intensity)
print("still done")

#%%

import matplotlib.colors as colors
minCon = np.nanmin([np.nanmin(contrast_array0), np.nanmin(contrast_array10), np.nanmin(contrast_array60), np.nanmin(contrast_array120)])
maxCon = np.nanmax([np.nanmax(contrast_array0), np.nanmax(contrast_array10), np.nanmax(contrast_array60), np.nanmax(contrast_array120)])


#%%

GAM = 0.65

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[1,0].set_title("120 mL/hr Speckle Contrast Pattern")
im = ax[1,0].imshow(contrast_array120,'jet', norm=colors.PowerNorm(GAM, vmin=minCon**GAM, vmax=maxCon**GAM))

ax[0,1].imshow(contrast_array60,'jet', norm=colors.PowerNorm(GAM, vmin=minCon**GAM, vmax=maxCon**GAM))
ax[0,1].set_title("60 mL/hr Speckle Contrast Pattern")

ax[0,0].imshow(contrast_array10,'jet', norm=colors.PowerNorm(GAM, vmin=minCon**GAM, vmax=maxCon**GAM))
ax[0,0].set_title("10 mL/hr Speckle Contrast Pattern")

ax[1,1].imshow(contrast_array0,'jet', norm=colors.PowerNorm(GAM, vmin=minCon**GAM, vmax=maxCon**GAM))
ax[1,1].set_title("Empty Tube Speckle Contrast Pattern")

# display graphs
# done based on https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax = cbar_ax)
plt.show()

#%%

image = plt.imread("/Users/zworrall/Desktop/coding/august_11/fin/fin_11_tube.bmp", 0)
intensity = np.asarray(image[y1:y2, x1:x2])
plt.imshow(intensity)

#%%

image = plt.imread("/Users/zworrall/Desktop/coding/august_11/fin/fin_11_withdiffuse.bmp", 0)
intensity = np.asarray(image)
plt.imshow(intensity)

import cv2 

start_point = (x1, y1)
end_point = (x2, y2)
color = (0, 0, 255)
thickness = 10
image = cv2.rectangle(image, start_point, end_point, color, thickness)

plt.imshow(image)

plt.imshow(image[y1:y2, x1:x2])

#%% GRAPH INTENSITY (PDF -- PROBABILITY DENSITY FUNCTION)

image = plt.imread(link120, 0)
intensity = np.asarray(image[y1:y2, x1:x2])
plt.imshow(intensity)
fig, ax = plt.subplots(ncols=1, nrows=1)

N_bins = 255
int_hist, int_bin_edges = np.histogram(intensity, bins=N_bins, range=[0, 255])

ax.plot(int_bin_edges[:-1], int_hist, 'b-')
ax.set_title("Intensity Histogram of Laser Speckle", fontsize = 8)
ax.set_xlim(0, 255)



#%%

stat_x1 = x1+0; stat_x2 = x1+200
stat_y1 = y1+100; stat_y2=y1+300

mov_x1 = x1+0; mov_x2 = x1+200
mov_y1 = y1+800; mov_y2 = y1+1000


# give average contrast in two specific sections of the image; one where things
# are moving, one where things aren't, in order to tell if there is a difference
# in moving speeds
image = plt.imread(link120, 0)
static_window120 = np.asarray(image[stat_y1:stat_y2,stat_x1:stat_x2])
stat_con_120 = np.std(static_window120) / np.mean(static_window120)
moving_window120 = np.asarray(image[mov_y1:mov_y2,mov_x1:mov_x2])
move_con_120 = np.std(moving_window120) / np.mean(moving_window120)

image = plt.imread(link60, 0)
static_window60 = np.asarray(image[stat_y1:stat_y2,stat_x1:stat_x2])
stat_con_60 = np.std(static_window60) / np.mean(static_window60)
moving_window60 = np.asarray(image[mov_y1:mov_y2,mov_x1:mov_x2])
move_con_60 = np.std(moving_window60) / np.mean(moving_window60)

image = plt.imread(link10, 0)
static_window10 = np.asarray(image[stat_y1:stat_y2,stat_x1:stat_x2])
stat_con_10 = np.std(static_window10) / np.mean(static_window10)
moving_window10 = np.asarray(image[mov_y1:mov_y2,mov_x1:mov_x2])
move_con_10 = np.std(moving_window10) / np.mean(moving_window10)

image = plt.imread(link0, 0)
static_window0 = np.asarray(image[stat_y1:stat_y2,stat_x1:stat_x2])
stat_con_0 = np.std(static_window0) / np.mean(static_window0)
moving_window0 = np.asarray(image[mov_y1:mov_y2,mov_x1:mov_x2])
move_con_0 = np.std(moving_window0) / np.mean(moving_window0)

print("Empty Tube | Static = ", stat_con_0, " || Moving = ", move_con_0)
print("10    Tube | Static = ", stat_con_10, " || Moving = ", move_con_10)
print("60    Tube | Static = ", stat_con_60, " || Moving = ", move_con_60)
print("134   Tube | Static = ", stat_con_120, " || Moving = ", move_con_120)


#%% GAUSSIAN PIECE

# CALCULATE CROSS-CORRELATION USING THEOREM
image = plt.imread(link120, 0)
photo_array = np.asarray(image[y1:y2, x1:x2])
ft = ifftshift(photo_array)
ft = fft2(ft)
ft = fftshift(ft)
ft_new = ft
matrix = ft_new
conj_m = np.conjugate(ft_new)
mult = conj_m * matrix
cross_corr = fft2(mult)
cross_corr = fftshift(cross_corr)
import matplotlib.patches as patches
im2 = plt.imshow(np.log(np.abs(cross_corr)), 'jet') #norm=colors.LogNorm())
plt.colorbar(im2)
plt.show()
half_corrX = int ( len(cross_corr) / 2 - 1 )
plt.plot(np.log(np.abs(cross_corr[half_corrX])))
plt.show()
plt.title("Cross-Correlation of Small-Aperture Speckle")
plt.xlabel("x-shift amount")
plt.ylabel("y-shift amount")
cross_gauss = np.log(np.abs(cross_corr[half_corrX, :]))
print(cross_gauss)
plt.imshow(np.abs(cross_corr), cmap='jet')
plt.plot(range(len(cross_corr[0])), [half_corrX]*len(cross_corr[0]), linewidth=1)
plt.show()
# https://stackoverflow.com/questions/19206332/gaussian-fit-for-python
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
def gaussian(x,a,x0,sigma, b):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b 

def findFitGaus(oneD_section):
    x = np.arange(len(oneD_section))
    y = np.array(oneD_section)

    # weighted arithmetic mean (corrected - check the section below)
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))


    popt,pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma, mean], maxfev=360000)
    
    return [x, y, popt, pcov]

#%%
# COMPUTE GAUSSIAN FIT FOR BEFORE AND AFTER
about_low = int(half_corrX-20); about_high = int(half_corrX+20)
cross_gauss2 = cross_gauss[about_low:about_high]
org_fit = findFitGaus(cross_gauss2)
# GRAPH GAUSSIAN FIT
fig5, ax5 = plt.subplots(1, 1, figsize=(20, 15))
plt.plot(np.arange(len(cross_gauss2)), cross_gauss2, 'b+:', label='data')
plt.plot(org_fit[0], gaussian(org_fit[0], *(org_fit[2])), 'r-', label='fit')
plt.title("Gaussian Fit")
plt.xlabel('pixels')
plt.ylabel('magnitude')
(x, a, x0, sigma, b) = (org_fit[0], *(org_fit[2]))
a_statement     = "a value: ", round(a, 2)
x0_statement    = "x0 value: ", round(x0, 2)
sigma_statement = "sigma value: ", np.abs(round(sigma, 2))
b_statement     = "b value: ", round(b, 2)
plt.figtext(0.15, 0.65,  a_statement, fontsize=12)
plt.figtext(0.15, 0.7, x0_statement, fontsize=12)
plt.figtext(0.15, 0.75,  sigma_statement, fontsize=12)
plt.figtext(0.15, 0.8, b_statement, fontsize=12)


