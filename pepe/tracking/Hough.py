"""
Particle position tracking via Hough transform.
"""
import numpy as np

import cv2
from scipy.ndimage import filters

import numba

from pepe.analysis import gSquared
from pepe.preprocess import sobelEdgeDetection, laplacianEdgeDetection, cannyEdgeDetection

import matplotlib.pyplot as plt

def houghCircle(singleChannelFrame, radius, edgeDetection='laplacian', blurKernel=None, cannyEdgeThreshold=70, accumulatorThreshold=20, radiusTolerance=15, minSeparation=None, debug=False):
    """
    Perform Hough circle detection on the provided image.

    For background on the Hough method (especially in the context
    of granular materials), see [1].

    OpenCV's implementation of the  Hough transform is used, so this
    method can be considered a wrapper for `cv2.HoughCircle()`.

    Tends to be less accurate than the convolution method (see
    `pepe.tracking.convCircle()`) but usually offers shorter computation
    times, especially when the radius is not well known. For cases
    in which the radius is known to a tolerance of ~10 pixels, the convolution
    method is recommended.

    Parameters
    ----------

    singleChannelFrame : np.uint8[H, W]
        The frame to perform circle detection on. Should not include multiple
        color channels.

    radius : float or [float, float]
        The approximate radius to look for, or a range of radii to look between.
        If a single value is provided, a range will be constructed around that
        value with the `radiusTolerance` parameter.

    edgeDetection : ['sobel', 'laplacian', 'canny', None]
        What type of edge detection should be performed on the image before the hough
        algorithm is applied.

    cannyEdgeThreshold : float
        Value between 0 and 100 representing the threshold for edge detection. Smaller
        values means the edges of circles do not have to be as strongly defined, but
        this also means you may get false positive detections.

    accumulatorThreshold : float
        Value between 0 and 100 representing the threshold for the accumulator (how
        the Hough method votes on which objects are circles). Higher values should
        be used when circles are well defined, and lower values for weaker signals.

    radiusTolerance : float
        Value used to construct a radius interval assuming only a single value is
        passed for parameter `radius`. Interval will be [radius - radiusTolerance,
        radius + radiusTolerance].

    minSeparation : float
        The minimum distance allowed between detected circles. If None (the
        default value) this will be calculated as twice the radius minus the
        radius tolerance.

    debug : bool
        Whether or not to draw debug information on a plot.

    Returns
    -------

    centers : np.ndarray[N,2]
        The detected circles' centers.

    radii : np.ndarray[N]
        The detected circles' radii.

    References
    ----------

    [1] Franklin, S. V., & Shattuck, M. D. (Eds.). (2016). Handbook
    of Granular Materials. Chapter 2: Experimental Techniques. p.68-70.
    CRC Press. https://doi.org/10.1201/b19291

    """

    # Check if we are given a range of radii or just a single value
    minRadius = None
    maxRadius = None

    if hasattr(radius, '__iter__'):
        if len(radius) == 2:
           minRadius = radius[0]
           maxRadius = radius[1]
    else:
        # Use the tolerance
        minRadius = radius - radiusTolerance
        maxRadius = radius + radiusTolerance


    if minRadius is None or maxRadius is None:
        raise Exception('Exception parsing expected radius!')

    if blurKernel is not None:
        blurredImage = cv2.blur(singleChannelFrame, (blurKernel, blurKernel))
    else:
        blurredImage = singleChannelFrame

    if edgeDetection == 'laplacian':
        detectImage = laplacianEdgeDetection(blurredImage)
    elif edgeDetection == 'sobel':
        detectImage = sobelEdgeDetection(blurredImage)
    elif edgeDetection == 'canny':
        detectImage = cannyEdgeDetection(blurredImage)
    else:
        detectImage = blurredImage


    # We don't expect too much overlap, so this should be a reasonable
    # separation distance
    if minSeparation is None:
        minSeparation = 2*minRadius - radiusTolerance

    detectedCircles = cv2.HoughCircles(np.array(detectImage, dtype=np.uint8), 
                    cv2.HOUGH_GRADIENT,
                    dp=1, # Accumulator (no idea what it does, but 1 is good)
                    minDist=minSeparation, # Minimum distance between circles
                    param1=cannyEdgeThreshold, # Threshold for canny edge detection
                    param2=accumulatorThreshold, # Accumulator threshold
                    minRadius=minRadius,
                    maxRadius=maxRadius)

    if detectedCircles is None:
        return np.array([]), np.array([])

    centers = np.array(list(zip(detectedCircles[0,:,1], detectedCircles[0,:,0])))
    radii = detectedCircles[0,:,2]


    if debug:
        fig, ax = plt.subplots(1, 3, figsize=(10,4))
        ax[0].imshow(singleChannelFrame)
        ax[0].set_title('Original image')

        ax[1].imshow(detectImage)
        ax[1].set_title('Detection image')

        ax[2].imshow(detectImage)
        ax[2].set_title('Detected circles')
        for i in range(len(centers)):
            c = plt.Circle(centers[i][::-1], radii[i], color='red', fill=False, linewidth=1)
            ax[2].add_artist(c)

        fig.tight_layout()

    return centers, radii




