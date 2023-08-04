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

image = plt.imread("PATH-TO-IMAGE.bmp", 0)
plt.set_cmap("gray")
intensityBoi = image
intensityBoi = np.asarray(intensityBoi)

#%% GRAPH INTENSITY (PDF -- PROBABILITY DENSITY FUNCTION)

fig, ax = plt.subplots(ncols=1, nrows=1)

N_bins = 255
int_hist, int_bin_edges = np.histogram(intensityBoi, bins=N_bins, range=[0, 255])

ax.plot(int_bin_edges[:-1], int_hist, 'b-')
ax.set_title("Intensity Histogram of Laser Speckle", fontsize = 8)
ax.set_xlim(0, 2)

#%% GRAPH FFT OF THE CONTRAST PATTERN

fig2, ax2 = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))

wd = int(len(intensityBoi[0]))
ht = int(len(intensityBoi))
print(wd, " ", ht)

photo_array = [ [0]*wd for i in range(ht)]
for r in range(ht):
    for c in range(wd):
        row = int(r)
        col = int(c)
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

window = photo_array[:][:]

w_dev = np.std(window)
w_meanInt = np.mean(window)
w_speckle_contrast = w_dev / w_meanInt
print(w_speckle_contrast)

SPECKLE_SQUARE = 7

pArr = np.asarray(photo_array)
contrast_array = [ [0]*(len(photo_array[0])-SPECKLE_SQUARE) for i in range((len(photo_array))-SPECKLE_SQUARE) ]

scale = [ [1]*(len(photo_array[0])) for i in range((len(photo_array))) ]
# pArr = pArr + scale
# to do:
    # define a for loop that goes through 7x7 pieces of the image
    # for each of these, compute the speckle contrast; compare it to the full
    # image's speckle contrast
    # using this comparison, create a new matrix that is composed of values
    # based on how "blurry" the image is; higher contrast than average means
    # less blurry, make it white (more stddev means more light bounced off)
for r in range(len(photo_array) - SPECKLE_SQUARE):
    for c in range(len(photo_array[0]) - SPECKLE_SQUARE):
        x1 = r; x2 = r+SPECKLE_SQUARE-1
        y1 = c; y2 = c+SPECKLE_SQUARE-1
        selection = pArr[x1:x2, y1:y2]
        s_dev = np.std(selection)
        s_meanInt = np.mean(selection)
        # just to avoid any technical difficulties, 
        # since it seems like s_meanInt can sometimes be 0:
        if (s_meanInt == 0):
            contrast_array[r][c] = -2
        else:
            s_speckle_contrast = s_dev / s_meanInt
            contrast_array[r][c] = -s_speckle_contrast


#%% SHOW ORIGINAL IMAGE 

plt.imshow(image)

#%% SHOW THE "CONTRAST IMAGE"

im = plt.imshow(contrast_array, 'gray')
plt.colorbar(im)

# To save the image to desktop, uncomment the following lines
# npCont = np.asarray(contrast_array)
# npCont = 255 * npCont
# im = Image.fromarray(npCont.astype(np.uint8))
# im.save('/Users/zworrall/Desktop/august_3/speck3x/speck3x_speckleContrast.bmp')

#%% CALCULATE CROSS-CORRELATION USING THEOREM

matrix = ft_new
conj_m = np.conjugate(ft_new)
mult = conj_m * matrix

cross_corr = fft2(mult)
cross_corr = fftshift(cross_corr)
center_sec = cross_corr[950:1000, 1275:1350]

import matplotlib.patches as patches

im2 = plt.imshow(np.log(np.abs(center_sec)), 'jet') #norm=colors.LogNorm())
plt.colorbar(im2)

#%%

half_corrX = int ( len(cross_corr) / 2 - 1 )
plt.plot(np.log(np.abs(cross_corr[half_corrX])))

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


#%% COMPUTE GAUSSIAN FIT FOR BEFORE AND AFTER + GRAPH

# shorten so that the gaussian is easier to see
MAXSIZE = int(SIZE/5 * 3.5)
MINSIZE = int(SIZE/5 * 1.5)

fig5, ax5 = plt.subplots(1, 1, figsize=(20, 15))

# adjust cross_gauss if there's somethign weird happening around the curve

cross_gauss = cross_gauss[1100:1500]

org_fit = findFitGaus(cross_gauss)

#__________________________________________________________________#

## GRAPHING GAUSSIAN FIT

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
plt.figtext(0.35, 0.7,  a_statement)
plt.figtext(0.35, 0.75, x0_statement)
plt.figtext(0.35, 0.8,  sigma_statement)
plt.figtext(0.35, 0.85, b_statement)
plt.colorbar()



#%% CALCULATE CROSS-CORRELATION with Scipy (O[n^2] time, don't recommend using)

# calculates cross correlation of the intensity of the speckle ("intensityBoi")
import scipy.signal as si

ift = fftshift(ft_new)
ift = ifft2(ift)
ift = ifftshift(ift)

org_cCorr = si.correlate2d(np.abs(ft_new), np.abs(ft_new), boundary='wrap', mode='same')














