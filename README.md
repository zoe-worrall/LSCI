# LSCI
Laser Speckle Contrast Imaging Practice and Models


## Background

This code was written in Summer, 2023 in conjunction with Harvey Mudd College's Engineering Professor Joshua Brake within the HMC Biophotonics Lab, with the purpose of evaluating the laser-speckle contrast imaging capabilities of the openIVIS project. It currently contains three Python files, which were used with photos taken by a U1800-A camera.

## Python Files

If you have any additional questions about this code, reach out to zworrall@g.hmc.edu with further questions.


### GaussianMethodTest.py

A collection of functions and tests originally written to work with both simulated and real LSCI systems. The program was written in Spyder, and is not meant to be run sequentially; you must run them in blocks in order to avoid errors.

The main purpose of this program was testing, but features that it includes is:
- generation of random speckle patterns
- confirmation what an aperture in a 4F system does to an image
- convert images into matrices (“intensityBoi”)

For real images, in particular, GaussianMethodTest.py is able to generate:
- Amplitude Histogram of the laser speckle
- Intensity Histogram of the laser speckle
- Phase Histogram of the laser speckle
- Gaussian distribution
- FWHM of the covariance of the system
	- Full Width, Half-Maximum. For Speckle contrast, that is how we measure how big, or rather "blurry", a part of the image is. It is important for determining that the camera is focused correctly during the trial. For further information, I recommend reading [this article](https://www.nature.com/articles/s41598-023-45303-z), specifically in the Methods > Simulation section, for a brief explanation of what this means.

#### Important Functions within GaussianMethodTest.py

scipy.correlate2d(matrix1, matrix2, boundary, mode): Cross correlates an image with itself using scipy. Note that this is not the fastest way to cross correlate. For a mathematical understanding of what cross correlation does, read [this MathWorld page](https://mathworld.wolfram.com/Cross-CorrelationTheorem.html).

gaussian(x, a, x0, sigma, b): Generates a list of gaussian points that will be used by the function findFitGaus.

findFitGaus(oneD_section): Using some set of points, finds the values x, y, popt, and popv that will be used to generate a Gaussian curve. In the context of the program, this is used to fit the center line of the cross correlation (which should be noted to appear Gaussian) to a Gaussian curve, which can then be used to find the FWHM (Full-Width Half Maximum) value.


### single_photo_analysis.py

Similar to GaussianMethodTest.py, this program was written using the Spyder GUI, and resultantly works best when run in blocks rather than run as a whole.

There are __ important functions that this is meant to do:
1. Convert an image into an array of intensities
2. Graph the intensity histogram of the array (used to confirm a proper exposure time has been found for the camera. The histogram should appear as a negative exponential)
3. Graph the Fourier Transform of the speckle pattern.
4. Calculate the speckle contrast of the image using the function "calcSpeckleContrast". In this calculates the speckle contrast of a 7x7 area, but the dimensions of this region is at the beginning of the program using the variable SQUARE.
5. Calculate the cross-correlation of the system before and after an aperture using the theorem found in [this MathWorld page](https://mathworld.wolfram.com/Cross-CorrelationTheorem.html).
6. Graph the cross correlation of the image
7. Match the central line of these cross-correlations with a gaussian curve, and plot this fit while labeling significant variables a, x0, sigma, and b.


### compare_sandwich_3.py

This program was written with the specific purpose of comparing four liquid speeds. A tube filled with cream and water was attached to a syringe pump, which was then set to move at four speeds, which can be found on [my Confluence page](https://hmcbiophotonics.atlassian.net/wiki/spaces/~63ffd3c09ce2cd2c240e0aff/pages/51511320/2023-08-10), specifically in the Expand tab labeled "Figure of Four Speeds".

The setup was the following:
- A crude lid made of wood was placed on top of the Summer, 2023 design of the openIVIS box (before a proper lid had been constructed). This lid held up both the camera and the laser (which was beam expanded).
- Two diffusers were stacked on top of each other in a "sandwich" around the syringe pump's tube.
- The tube was kept still by masking tape throughout the box.


The program takes advantage of the same speckle contrast loop that is used in single_photo_analysis.py, however it is used as a function and run over all four images.

The majority of the code contains the same evaluations of a speckle pattern as the single_photo_analysis, mainly calculating the speckle contrast, performing cross-correlation, and computing the gaussian fit of the images.
