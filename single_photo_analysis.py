#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:26:52 2023

Most recent update: Friday, August 4, 2023

Using an input Laser Speckle Image, plots the following:
    * an intensity histogram of pixels (monochrome and bmp file only)
    * the original image versus is 2D-Fourier transform
    * the speckle contrast of the original image
    * a cross correlation of the matrix with itself
    * the gaussian fit of the cross-section: cuts off the sides to prevent
        the slopes outside of the gaussian from influencing the fit
    

@author: zworrall
"""

# to save images directly to your desktop:
    
# numpy_array = np.asarray(name_of_image_array)
# numpy_array = 255 * numpy_array       ## does scaling for conversion to uint
# im = Image.fromarray(npCont.astype(np.uint8))
# im.save('PATH-TO-SAVE-FOLDER/NAME-OF-IMAGE-SAVED.bmp')


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


#%% SET UP IMAGE AND INTENSITY MATRIX

# import only monochrome (black and white) bmp files: files with RGB tuples
#   for individual pixels need additional processing to work in this code
image = plt.imread("/Users/zworrall/Desktop/coding/august_10/880_10_1.bmp", 0)
plt.imshow(image)
plt.set_cmap("gray")

# convert into numpy array for future work
instensity = np.asarray(image)

#%% GRAPH INTENSITY (PDF -- PROBABILITY DENSITY FUNCTION)

fig, ax = plt.subplots(ncols=1, nrows=1)

N_bins = 255
int_hist, int_bin_edges = np.histogram(instensity, bins=N_bins, range=[0, 255])

ax.plot(int_bin_edges[:-1], int_hist, 'b-')
ax.set_title("Intensity Histogram of Laser Speckle", fontsize = 8)
ax.set_xlim(0, 255)


#%% GRAPH FFT OF THE CONTRAST PATTERN

fig2, ax2 = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))

wd = int(len(instensity[0]))
ht = int(len(instensity))
print(wd, " ", ht)

photo_array = [ [0]*wd for i in range(ht)]
for r in range(ht):
    for c in range(wd):
        row = int(r)
        col = int(c)
        px = instensity[row][col]
        photo_array[r][c] = px# (px[0] + px[1] + px[2])/3

# taken from https://stackoverflow.com/questions/32766162/set-two-matplotlib-imshow-plots-to-have-the-same-color-map-scale
#Get the min and max of all your data
_min, _max = np.amin(photo_array), np.amax(photo_array)


# GRAPH FIRST PLOT: ORIGINAL LSCI PATTERN
ax2 = plt.subplot(121)
# Calculate Fourier transform of grating
plt.imshow(photo_array, vmin = _min, vmax = _max)
x = np.arange(-500, 501, 1)
plt.title("LSCI Pattern")
plt.xlabel("x")
plt.ylabel("y")


# GRAPH SECOND PLOT: FOURIER TRANSFORM OF THE ORIGINAL PATTERN
ax2 = plt.subplot(122)

# DO FIRST FOURIER TRANSFORM
plt.set_cmap("gray")
ft = ifftshift(photo_array)
ft = fft2(ft)
ft = fftshift(ft)
ft_new = ft


plt.title("Fourier Transform")
plt.xlabel("k_{x}")
plt.ylabel("k_{y}")
four_min, four_max = np.amin(np.log(np.abs(ft))), np.amax(np.log(np.abs(ft)))
# logNorm plots the log by itself; no change of data necessary
im = plt.imshow(np.log(np.abs(ft)), vmin = four_min, vmax = four_max) #norm=colors.LogNorm())

# display graphs
# done based on https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
fig2.subplots_adjust(right=0.8)
cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
fig2.colorbar(im, cax = cbar_ax)

plt.show()


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


#%%

(cont, contrast_array) = calcSpeckleContrast(photo_array)

#%%

import matplotlib.colors as colors
minCon = np.nanmin(contrast_array)
maxCon = np.nanmax(contrast_array)
im = plt.imshow(contrast_array,'jet', norm=colors.LogNorm(vmin=minCon, vmax=maxCon))
plt.colorbar(im)

#%% CALCULATE CROSS-CORRELATION USING THEOREM

# although scipy does have a 2d cross-correlation function (correlate2d),
#   it takes O(n^2) time compared to this code's O(n * log[n]) time

matrix = ft_new
conj_m = np.conjugate(ft_new)
mult = conj_m * matrix

cross_corr = fft2(mult)
cross_corr = fftshift(cross_corr)

import matplotlib.patches as patches

# plot cross correlation of matrix with itself
im2 = plt.imshow(np.log(np.abs(cross_corr)), 'jet') #norm=colors.LogNorm())
plt.colorbar(im2)
plt.show()

# values along center axis of cross-correlated matrix
half_corrX = int ( len(cross_corr) / 2 - 1 )
plt.plot(np.log(np.abs(cross_corr[half_corrX])))
plt.show()

#%% GRAPH THE CROSS-CORRELATION OF THE IMAGE

plt.title("Cross-Correlation of Small-Aperture Speckle")
plt.xlabel("x-shift amount")
plt.ylabel("y-shift amount")

cross_gauss = np.log(np.abs(cross_corr[half_corrX, :]))
print(cross_gauss)
plt.imshow(np.abs(cross_corr), cmap='jet')
plt.plot(range(len(cross_corr[0])), [half_corrX]*len(cross_corr[0]), linewidth=1)
plt.show()

#%% GAUSSIAN FUNCTIONS

# https://stackoverflow.com/questions/19206332/gaussian-fit-for-python
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


# taken from https://stackoverflow.com/questions/19206332/gaussian-fit-for-python
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


#%% COMPUTE GAUSSIAN FIT FOR BEFORE AND AFTER

# adjust cross_gauss if there's somethign weird happening around the curve

cross_gauss = cross_gauss[1275:1325]

org_fit = findFitGaus(cross_gauss)

#%% GRAPH GAUSSIAN FIT

fig5, ax5 = plt.subplots(1, 1, figsize=(20, 15))

# correlation points graphed
plt.plot(np.arange(len(cross_gauss)), cross_gauss, 'b+:', label='data')

# gaussian fit points graphed
plt.plot(org_fit[0], gaussian(org_fit[0], *(org_fit[2])), 'r-', label='fit')

plt.title("Gaussian Fit")
plt.xlabel('pixels')
plt.ylabel('magnitude')

# save the values of x, a, x0, sigma, and b from the gaussian curves
(x, a, x0, sigma, b) = (org_fit[0], *(org_fit[2]))

# values to print on the graph
a_statement     = "a value: ", round(a, 2)
x0_statement    = "x0 value: ", round(x0, 2)
sigma_statement = "sigma value: ", np.abs(round(sigma, 2))
b_statement     = "b value: ", round(b, 2)

# print the values on the graph
plt.figtext(0.15, 0.65,  a_statement, fontsize=12)
plt.figtext(0.15, 0.7, x0_statement, fontsize=12)
plt.figtext(0.15, 0.75,  sigma_statement, fontsize=12)
plt.figtext(0.15, 0.8, b_statement, fontsize=12)















