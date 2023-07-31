#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:26:52 2023

creates an FFT, phase diagram plots, etc. for an image that the user input

@author: zworrall
"""

#%% IMPORTING NECESSARY FILES

from array import *
import numpy as np
import numpy.fft
from PIL import Image
#from scipy.stats import t
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


#%% DECLARE GLOBAL VARIABLES AND METHODS TO BE USED

SIZE = 2000
NUM_EXPOSURES = 0

image = plt.imread("/Users/zworrall/Downloads/open_ap.bmp", 0)
plt.set_cmap("gray")

intensityBoi = image
intensityBoi = np.asarray(intensityBoi)


#%% GRAPH FFT OF THE CONTRAST PATTERN

fig2, ax2 = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))

wd = int(len(intensityBoi[0])/5)
ht = int(len(intensityBoi)/5)
print(wd, " ", ht)

photo_array = [ [0]*wd for i in range(ht)]
for r in range(ht):
    for c in range(wd):
        row = int(r+2.5 * ht)
        col = int(c+2.2 * wd)
        px = intensityBoi[row][col]
        photo_array[r][c] = px# (px[0] + px[1] + px[2])/3

# taken from https://stackoverflow.com/questions/32766162/set-two-matplotlib-imshow-plots-to-have-the-same-color-map-scale
#Get the min and max of all your data
_min, _max = np.amin(photo_array), np.amax(photo_array)


# GRAPH FIRST PLOT: ORIGINAL LSCI PATTERN

ax2 = plt.subplot(121)
# Calculate Fourier transform of grating
plt.imshow(photo_array, vmin = _min, vmax = _max)
x = np.arange(-500, 501, 1)
plt.title("LSCI Pattern - Small Aperture")
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

# note: can't graph phase, amplitude, intensity since we only have intensity in
    # the photo, not the phase -- not starting with an electric field
    

#%% CALCULATE CROSS-CORRELATION

# calculates cross correlation of the intensity of the speckle ("intensityBoi")
import scipy.signal as si

ift = fftshift(ft_new)
ift = ifft2(ift)
ift = ifftshift(ift)

org_cCorr = si.correlate2d(np.abs(ift), np.abs(ift), boundary='wrap', mode='same')


#%% GRAPH THE CROSS-CORRELATION OF THE IMAGE

plt.title("Cross-Correlation of Small-Aperture Speckle")
plt.xlabel("x-shift amount")
plt.ylabel("y-shift amount")

halfx = int(len(org_cCorr)/2)-1
org_GaussianPoints = org_cCorr[halfx, :]
print(org_GaussianPoints)
plt.imshow(np.abs(org_cCorr), cmap='jet')
plt.plot(range(len(org_cCorr[0])), [halfx]*len(org_cCorr[0]), linewidth=1)
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



#%% COMPUTE GAUSSIAN FIT FOR BEFORE AND AFTER + GRAPH

# shorten so that the gaussian is easier to see
MAXSIZE = int(SIZE/5 * 3.5)
MINSIZE = int(SIZE/5 * 1.5)

# take the center line along the middle of the y axis to see if gaussian
halfx = int(len(org_cCorr)/2) -1
org_GaussianPoints = org_cCorr[halfx, :]
# absOrg_GaussianPoints = np.abs(org_GaussianPoints)

fig5, ax5 = plt.subplots(1, 1, figsize=(20, 15))

org_fit = findFitGaus(org_GaussianPoints)

## GRAPHING GAUSSIAN FIT
# correlation points graphed
plt.plot(np.arange(len(org_GaussianPoints)), org_GaussianPoints, 'b+:', label='data')

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
sigma_statement = "sigma value: ", round(sigma, 2)
b_statement     = "b value: ", round(b, 2)

# print the values on the graph
plt.figtext(0.35, 0.7,  a_statement)
plt.figtext(0.35, 0.75, x0_statement)
plt.figtext(0.35, 0.8,  sigma_statement)
plt.figtext(0.35, 0.85, b_statement)








