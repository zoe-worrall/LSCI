# testing Gaussian method
#%%
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


#%% DECLARE GLOBAL VARIABLES :D :D

SIZE = 2000
NUM_EXPOSURES = 0
TYPE_RUN = 3

#%% A METHOD THAT GENERATES A SPECKLE PATTERN AND DOES NOT BLUR THE MATRIX

def generateSpeckleContrast():
    
    #% GENERATE THE REAL AND IMAGINARY MATRICES
    my_matrixA = abs(np.random.randn(SIZE, SIZE))
    my_matrixB = abs(np.random.randn(SIZE, SIZE))

    speckle = my_matrixA + 1j*my_matrixB
    
    return speckle

#%% A METHOD THAT GENERATES A SPECKLE PATTERN IN LINES

def generateSpeckleLines():
    
    #% GENERATE THE REAL AND IMAGINARY MATRICES
    my_matrixA = abs(np.random.randn(SIZE, SIZE))
    my_matrixB = abs(np.random.randn(SIZE, SIZE))
    
    maxA = np.max(my_matrixA)
    maxB = np.max(my_matrixB)
    
    for i in range(len(my_matrixA)-1):
        for j in range(len(my_matrixA[0])):
            # flip a coin; if the coin is true: pixel = pixel to left
            # if coin false: pixel stays unchanged
            # basically should create a basic background pattern
            randomNum = random.randrange(0, 100)/100.0
            if (randomNum < 0.75):
                my_matrixA[i+1][j] = my_matrixA[i][j] + random.randrange(0, 50)/1000.0
                my_matrixB[i+1][j] = my_matrixB[i][j] + random.randrange(0, 50)/1000.0
    
    speckle = my_matrixA + 1j*my_matrixB
    
    return speckle


#%% DOWNLOAD ANY NECESSARY IMAGES AND SET THEM UP TO BE USED

if TYPE_RUN == 1: # SEE WHAT HAPPENS WITH MULTIPLE SPECKLE EXPOSURES
    speckleBoi = generateSpeckleContrast()
    for i in range(NUM_EXPOSURES):
        speckleBoi = speckleBoi + generateSpeckleContrast()
    intensityBoi = np.square(np.abs(speckleBoi))
elif TYPE_RUN == 2: # SEE WHAT HAPPENS WITH MULTIPLE SPECKLE (wavey) EXPOSURES
    speckleBoi = generateSpeckleLines()
    for i in range(NUM_EXPOSURES):
        speckleBoi = speckleBoi + generateSpeckleLines()
    intensityBoi = np.square(np.abs(speckleBoi))
elif TYPE_RUN == 3: # do it with images
    # Read and process image
    image = plt.imread("PATH_TO_gridwall.png")
    image = image[:, :, :3].mean(axis=2)  # Convert to grayscale
    
    plt.set_cmap("gray")
  
    intensityBoi = image



#%% GRAPH FFT OF THE CONTRAST PATTERN

fig2, ax2 = plt.subplots(ncols=2, nrows=2)

# taken from https://stackoverflow.com/questions/32766162/set-two-matplotlib-imshow-plots-to-have-the-same-color-map-scale
combined_data = np.append(np.abs(intensityBoi), np.abs(ift))
#Get the min and max of all your data
_min, _max = np.amin(combined_data), np.amax(combined_data)


# GRAPH FIRST PLOT: ORIGINAL LSCI PATTERN

ax2 = plt.subplot(221)
# Calculate Fourier transform of grating
plt.imshow(intensityBoi, vmin = _min, vmax = _max)
x = np.arange(-500, 501, 1)
plt.title("Original LSCI Pattern")
plt.xlabel("x")
plt.ylabel("y")


# GRAPH SECOND PLOT: FOURIER TRANSFORM OF THE ORIGINAL PATTERN
ax2 = plt.subplot(222)

plt.set_cmap("jet")
ft = ifftshift(intensityBoi)
ft = fft2(ft)
ft = fftshift(ft)
ft_new = ft
plt.title("Fourier Transform W/O Aperture")
plt.xlabel("k_{x}")
plt.ylabel("k_{y}")
four_min, four_max = np.amin(np.log(np.abs(ft))), np.amax(np.log(np.abs(ft)))
# logNorm plots the log by itself; no change of data necessary
plt.imshow(np.log(np.abs(ft)), vmin = four_min, vmax = four_max) #norm=colors.LogNorm())

# GRAPH FOURTH PLOT: FOURIER TRANSFORM WITH APERTURE
halfx = len(ft_new)/2
halfy = len(ft_new[0])/2
WAVELENGTH = min(halfx, halfy) * 0.1667/2      # 50.0 worked for 300 x 300; do 1/6 for now # (nm)
RADIUS = 0.5  * WAVELENGTH
RADIUS = 3 * RADIUS # added to make it less blurry
for i in range(len(ft_new)):
    for j in range(len(ft_new[0])):
        dist = np.sqrt( np.square(halfx-i) + np.square(halfy-j))
        if (dist >= RADIUS):
            ft_new[i][j] = 0
ax2 = plt.subplot(224)
plt.title("Filtered FFT")
plt.xlabel("k_{x}")
plt.ylabel("k_{y}")
plt.imshow(np.log(np.abs(ft_new)), vmin = four_min, vmax = four_max)

# GRAPH THIRD PLOT: INVERT FOURIER TRANSFORM TO FIND BLURRED ORIGINAL IMAGE
# Calculate Fourier transform of grating
ift = fftshift(ft_new)
ift = ifft2(ift)
ift = ifftshift(ift)
ax2 = plt.subplot(223)
plt.title("LPF LSCI Pattern")
plt.xlabel("x")
plt.ylabel("y")
im = plt.imshow(np.abs(ift), vmin = _min, vmax = _max)

# DISPLAY
# done based on https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
fig2.subplots_adjust(right=0.8)
cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
fig2.colorbar(im, cax = cbar_ax)
plt.show()


#%% SIGNAL GRAPHING




fig, ax = plt.subplots(ncols=3, nrows=2)



N_bins = 100;
amp_hist,    amp_bin_edges = np.histogram(np.abs(speckleBoi), bins=N_bins)
int_hist,    int_bin_edges = np.histogram(np.abs(speckleBoi)**2, bins=N_bins)
phase_hist,  phase_bin_edges = np.histogram(np.angle(speckleBoi), bins=N_bins)

ax[0,0].plot(amp_bin_edges[:-1], amp_hist)
ax[0,0].set_title("Amplitude Histogram of Laser Speckle", fontsize = 8)
ax[0,1].plot(int_bin_edges[:-1], int_hist)
ax[0,1].set_title("Intensity Histogram of Laser Speckle", fontsize = 8)
ax[0,2].plot(phase_bin_edges[:-1], phase_hist)
ax[0,2].set_title("Phase Histogram of Laser Speckle", fontsize = 8)



n_amp_hist,    n_amp_bin_edges = np.histogram(np.abs(ift), bins=N_bins)
n_int_hist,    n_int_bin_edges = np.histogram(np.abs(ift)**2, bins=N_bins)
n_phase_hist,  n_phase_bin_edges = np.histogram(np.angle(ift), bins=N_bins)

ax[1,0].plot(n_amp_bin_edges[:-1], n_amp_hist)
ax[1,0].set_title("Amplitude Histogram of LPF Laser Speckle", fontsize = 8)
ax[1,1].plot(n_int_bin_edges[:-1], n_int_hist)
ax[1,1].set_title("Intensity Histogram of LPF Laser Speckle", fontsize = 8)
ax[1,2].plot(n_phase_bin_edges[:-1], n_phase_hist)
ax[1,2].set_title("Phase Histogram of LPF Laser Speckle", fontsize = 8)





#%% CALCULATE CROSS-CORRELATION

# calculates cross correlation for both the original (speckleBoi) and the lpf
# version (ift)
import scipy.signal as si

org_cCorr = si.correlate2d(np.abs(speckleBoi), np.abs(speckleBoi), boundary='wrap', mode='same')
new_cCorr = si.correlate2d(np.abs(ift), np.abs(ift), boundary='wrap', mode='same')

#%% GRAPH THE CROSS-CORRELATION


fig4, (ax4_orig, ax4_corr) = plt.subplots(1, 2, figsize=(6, 15))

ax4_orig.imshow(org_cCorr, cmap='jet')
ax4_orig.set_title("Original Cross-Correlation of LSCI")
ax4_orig.set_xlabel("x-shift amount")
ax4_orig.set_ylabel("y-shift amount")

ax4_corr.imshow(new_cCorr, cmap='jet')
ax4_corr.set_title("LSCI's LPF Cross Correlation")
ax4_corr.set_xlabel("x-shift amount")
ax4_corr.set_ylabel("y-shift amount")


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


#%% COMPUTE FIT FOR BEFORE AND AFTER + GRAPH

# shorten so that the gaussian is easier to see
MAXSIZE = int(SIZE/5 * 3.5)
MINSIZE = int(SIZE/5 * 1.5)

# take the center line along the middle of the y axis to see if gaussian
halfx = int(len(new_cCorr)/2) -1

lpf_GaussianPoints = new_cCorr[:, halfx]
org_GaussianPoints = org_cCorr[:, halfx]

   
fig5, ax5 = plt.subplots(1, 2, figsize=(20, 15))

org_fit = findFitGaus(org_GaussianPoints)
lpf_fit = findFitGaus(lpf_GaussianPoints)

## begin graphing

### graph the original data set's Gaussian fit

ax5 = plt.subplot(121)

plt.plot(np.arange(len(org_GaussianPoints)), org_GaussianPoints, 'b+:', label='data')
plt.plot(org_fit[0], gaussian(org_fit[0], *(org_fit[2])), 'r-', label='fit')
plt.title("Gaussian Fit Pre-Filter")

(x, a, x0, sigma, b) = (org_fit[0], *(org_fit[2]))

a_statement = "a value: ", round(a, 2)
x0_statement = "x0 value: ", round(x0, 2)
sigma_statement = "sigma value: ", round(sigma, 2)
b_statement = "b value: ", round(b, 2)

plt.figtext(0.35, 0.7, a_statement)
plt.figtext(0.35, 0.75, x0_statement)
plt.figtext(0.35, 0.8, sigma_statement)
plt.figtext(0.35, 0.85, b_statement)

### graph the low-pass filter's Gaussian fit

ax5 = plt.subplot(122)

plt.plot(np.arange(len(lpf_GaussianPoints)), lpf_GaussianPoints, 'b+:', label='data')
plt.plot(lpf_fit[0], lpf_fit[1], 'b+:', label='data')
plt.plot(lpf_fit[0], gaussian(lpf_fit[0], *(lpf_fit[2])), 'r-', label='fit')
plt.title("Gaussian Fit Low Pass Filter")

(x, a,x0,sigma, b) = (lpf_fit[0], *(lpf_fit[2]))
a_statement = "a value: ", round(a, 2)
x0_statement = "x0 value: ", round(x0, 2)
sigma_statement = "sigma value: ", round(sigma, 2)
b_statement = "b value: ", round(b, 2)

plt.figtext(0.75, 0.7, a_statement)
plt.figtext(0.75, 0.75, x0_statement)
plt.figtext(0.75, 0.8, sigma_statement)
plt.figtext(0.75, 0.85, b_statement)

plt.show()