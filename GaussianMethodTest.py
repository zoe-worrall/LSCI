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


#%% DECLARE GLOBAL VARIABLES

SIZE = 400
NUM_EXPOSURES = 0
TYPE_RUN = 2

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
    grid_im = "/Users/zoeworrall/Desktop/Mudd/2023_2/research/gridwall.png"
    
    # Read and process image
    image = plt.imread(grid_im)
    image = image[:, :, :3].mean(axis=2)  # Convert to grayscale
    
    plt.set_cmap("gray")
  
    intensityBoi = image


#%% GRAPH FFT OF THE CONTRAST PATTERN

fig2, ax2 = plt.subplots(ncols=2, nrows=1)


ax2 = plt.subplot(121)
# Calculate Fourier transform of grating

plt.imshow(intensityBoi)

x = np.arange(-500, 501, 1)

plt.title("Original LSCI Pattern")
plt.xlabel("x")
plt.ylabel("y")
plt.set_cmap("jet")
plt.colorbar()


# FOURIER TRANSFORM

ax2 = plt.subplot(122)

ft = ifftshift(intensityBoi)
ft = fft2(ft)
ft = fftshift(ft)

ft_new = ft


plt.title("Fourier Transform W/O Aperture")
plt.xlabel("k_{x}")
plt.ylabel("k_{y}")


import matplotlib.colors as colors

# logNorm plots the log by itself; no change of data necessary
plt.imshow(np.abs(ft_new)**2, norm=colors.LogNorm())

plt.show()


#%% FOURIER TRANSFORM WITH APERTURE

fig3, ax3 = plt.subplots(ncols=2, nrows=1)

halfx = len(ft_new)/2
halfy = len(ft_new[0])/2

WAVELENGTH = min(halfx, halfy) * 0.1667/2      # 50.0 worked for 300 x 300; do 1/6 for now # (nm)
RADIUS = 0.5  * WAVELENGTH


for i in range(len(ft_new)):
    for j in range(len(ft_new[0])):
        if (np.sqrt(np.square(halfx-i) + np.square(halfy-j)) > RADIUS):
            ft_new[i][j] = 0

ax3 = plt.subplot(122)
plt.title("Filtered FFT")
plt.xlabel("k_{x}")
plt.ylabel("k_{y}")
plt.imshow(np.log(np.abs(ft_new)))


# INVERT FOURIER TRANSFORM TO FIND BLURRED ORIGINAL IMAGE

# Calculate Fourier transform of grating

ift = fftshift(ft_new)
ift = ifft2(ift)
ift = ifftshift(ift)

ax3 = plt.subplot(121)
plt.title("LPF LSCI Pattern")
plt.xlabel("x")
plt.ylabel("y")
plt.imshow(np.abs(ift))


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

plt.show()


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


#%% COMPUTE USING FOURIER TRANSFORMS


# read about setting up for the speckle image
    # adjusting aperture
    # take survey of LSCI I've looked up; Confluence page of their setups


#  f*g=F[F^_(nu)G(nu)] ** F^_ means complex conjugate
fig4, (ax5) = plt.subplots(1, 1, figsize=(6, 15))

def padArr(arr, val):
    # convert into numpy arrays
    arrBoi = np.asarray(arr)
    fluffBoi = [[val] * len(arrBoi)] * len(arrBoi[0])
    fluff = np.asarray(fluffBoi)
    
    # create matrix with padding to allow for best cross-correlation
    fluffMat = np.concatenate((fluff, fluff, fluff), axis=1)
    newArr = np.concatenate((fluff, arrBoi, fluff), axis=1)
    newArr = np.concatenate((fluffMat, newArr, fluffMat), axis=0)
    
    return newArr


def cross_correlate_same(arr):
    
    padded = padArr(arr, 0)
    newArr = np.array()
    
    initIndex_x = len(padded)/3 - 1
    initIndex_y = len(padded[0])/3 - 1
    
    ft_x = len(padded)/3
    st_x = len(padded)/3 * 2
    
    ft_y = len(padded[0])/3
    st_y = len(padded[0])/3 * 2
    
    # work through every row and column in padded that 
        # corresponds to the inital values we had
    for r in range(len(padded)/3):
        for c in range(len(padded[0])/3):
            newArr[r][c] = sum(( arr[initIndex_x][initIndex_y] * arr[i+r][j+c] for j in range(ft_y, st_y) for i in ange(ft_x, st_x) ) )

    return newArr


spk = np.abs(speckleBoi)
print(spk)
ax5.imshow(cross_correlate_same(spk), cmap='jet')




#%%
    
    

# https://stackoverflow.com/questions/31543775/how-to-perform-cubic-spline-interpolation-in-python
def f(x):
    x_points = np.asarray(range(len(org_GaussianPoints)))
    y_points = np.conjugate(org_GaussianPoints)
    
    tck = scipy.interpolate.splrep(x_points, y_points)
    return scipy.interpolate.splev(x, tck)

F_nu = scipy.integrate.quad(f, -np.inf, np.inf)

def g(x):
    x_points = np.asarray(range(len(org_GaussianPoints)))
    y_points = org_GaussianPoints
    
    tck = scipy.interpolate.splrep(x_points, y_points)
    return scipy.interpolate.splev(x, tck)

G = scipy.integrate.quad(f, -np.inf, np.inf)

fStarG = F_nu[0] * G[0]

cross_corr = fft(fStarG)

plt.plot(cross_corr)


#%%

# for future reference: https://ieeexplore.ieee.org/abstract/document/8886444
    # S=2.44λ(1+M)f/#
    #   λ is the illumination wavelength
    #   M is the imaging system magnification
    #   f/# is the camera lens aperture



#%%

## autocorrelation notes
# f(t) * f(t)

# convolution
# (f*g)(t) = integral(f(tau) * g(t-tau) * d_tau) for neg_inf to infinity

# speckle pattern is a random pattern
# correlation is 

# convolution vs cross-correlation; cross-correlation doesn't flip g(tau)
    # conv -> g(t-tau) :: cross-corr -> g(t + tau)
    # for cross-corr, if complex numbers u take the abs(?)

# flip the g(tau): why? idk man

# if we have two functions that are very correlated with each other, 
    # expect wide cross-correlated function
    
# if two functions that are very uncorrelated, we expect very narrow 
    # cross correlated function
    # for randomly generated LCSI, we expect small
    # for bigger speckles, we expect higher correlation

# this is how we measure speckle size:
    # take autocorrelation (correlation of pattern with itself)
    # np and scipy have cross correlation, check what happens at edges
    
    
# cross correlation of speckle pattern should be sharp at middle
# if we blur it, cross correlation should get bigger
    # cut cross-correlation; should look gaussian; full width half max is xyz
    # take pattern, compute cross correlation, and figure out how large the pixel is
    # you want at least Nyquist; want speckle size to be at least two pixels
    
# cross correlation theorem; cross calc by taking fourier transforms and multiplying
# together, and then taking fourier transform again

#%%
# Low pass filter speckle

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

#%% DEFINE A FUNCTION THAT BLURS A 2X2 MATRIX TO A 5X5

# A function that generates/blurs four corners of a matrix and expands it from 1x1 to 5x5
    # takes the four corners of the square and makes a matrix that blends them together
    # a _ _ _ b
    # _ _ _ _ _
    # _ _ _ _ _
    # _ _ _ _ _
    # c _ _ _ d
    #    ^^ fills in the "_" tiles
def makeMat(a, b, c, d):
    mat = np.random.randn(5, 5)
    for i in range(5):
        mat[0, i] = a * (4-i)/4 + b * i/4
        mat[i, 0] = a * (4-i)/4 + c * i/4
        mat[4, i] = c * (4-i)/4 + d * i/4
        mat[i, 4] = b * (4-i)/4 + d * i/4
    for i in range(3):
        mat[1, i+1] = mat[1, 0] * (4-(i+1))/4 + mat[1, 4] * (i+1)/4
        mat[2, i+1] = mat[2, 0] * (4-(i+1))/4 + mat[2, 4] * (i+1)/4
        mat[3, i+1] = mat[3, 0] * (4-(i+1))/4 + mat[3, 4] * (i+1)/4
    mat[2, 2] = (mat[1, 2] + mat[2, 1] + mat[3, 2] + mat[2, 3]) / 4
    return mat


#%% A METHOD THAT GENERATES A SPECKLE PATTERN AND RETURNS THE SPECKLE PATTERN MATRIX
def generateSpeckle():
    
    #% GENERATE THE REAL AND IMAGINARY MATRICES
    my_matrixA = abs(np.random.randn(SIZE, SIZE))
    my_matrixB = abs(np.random.randn(SIZE, SIZE))
    
    # matrices to help combine all the values in the matrices together
    part_mat = []
    col_mat = []

    # make them the right size beforehand
    n_matA = [[0] * (SIZE*5-5) for i in range(SIZE*5-5)]
    n_matB = [[0] * (SIZE*5-5) for i in range(SIZE*5-5)]

    # n_matA is real, n_matB is imaginary
    meanInt = 0

    #% BLUR THE PIXELS TOGETHER
    # adjust the pixels within the matrix
    for r in range(SIZE-1):
        for c in range(SIZE-1):
            part_matA = makeMat(my_matrixA[r, c], my_matrixA[r, c+1], my_matrixA[r+1, c], my_matrixA[r+1, c+1])
            part_matB = makeMat(my_matrixB[r, c], my_matrixB[r, c+1], my_matrixB[r+1, c], my_matrixB[r+1, c+1])

            for i in range(5):
                for j in range(5):
                    rNew = int(r*5+i); cNew = int(c*5+j)
                    n_matA[rNew][cNew] = part_matA[i, j]
                    n_matB[rNew][cNew] = part_matB[i, j]
                    meanInt = meanInt + (n_matA[rNew][cNew]**2 + n_matB[rNew][cNew]**2)

    #% MAKE THE SPECKLE PATTERN
    speckle = my_matrixA + 1j*my_matrixB
    
    return speckle


















