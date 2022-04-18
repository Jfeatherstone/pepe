"""
Filter application and edge detection.
"""
import numpy as np

from scipy.ndimage import filters

import numba
import cv2

def applyFilter(singleChannelFrame, filter):
    """
    Apply an arbitrary filter (in the form of a matrix) to an image.

    Pretty much just a wrapper for `scipy.ndimage.filter.convolve()`.

    Parameters
    ----------

    singleChannelFrame : np.uint8[H,W]
        The frame to perform circle detection on. Should not include multiple
        color channels; if it does contain multiple channels, the first one will
        be used.

    filter : np.ndarray[N,N]
        Square matrix with an odd kernel size (N is odd) to be applied across the image.

    Returns
    -------

    filtedImage : np.uint8[H,W]
        Filtered image.
    """
    if singleChannelFrame.ndim > 2:
        image = singleChannelFrame.astype(np.float64)[:,:,0]
    else:
        image = singleChannelFrame.astype(np.float64)

    # Make sure the filter looks good
    assert filter.shape[0] == filter.shape[1], 'Invalid filter provided; must be square matrix'
    assert filter.shape[0] % 2 == 1, 'Invalid filter provided; must have odd kernel size'

    # So I wrote this and then found out that scipy has what is most likely a 
    # way better implementation, so we'll just use theirs
    
    return filters.convolve(image, filter)

#    kernelSize = filter.shape[0]
#    # Because we always have an odd kernel size
#    halfKernelSize = int((kernelSize - 1) / 2)
#    # Add some padding to our image so we can apply
#    # the filter to the entire image
#    paddedImage = np.zeros((image.shape[0] + 2*halfKernelSize, image.shape[1] + 2*halfKernelSize))
#    paddedImage[halfKernelSize:-halfKernelSize,halfKernelSize:-halfKernelSize] = image
#    
#    return _applyFilterOpt(paddedImage.astype(np.float64), filter, kernelSize).astype(np.uint8)
#    
#@numba.njit()
#def _applyFilterOpt(paddedImage, filter, kernelSize):
#    """
#    numba optimized method that is wrapped by applyFilter().
#    """
#    filteredImage = np.zeros((paddedImage.shape[0] - kernelSize + 1, paddedImage.shape[1] - kernelSize + 1))
#    for i in range(filteredImage.shape[0]):
#        for j in range(filteredImage.shape[1]):
#            filteredImage[i,j] = np.sum(paddedImage[i:i+kernelSize,j:j+kernelSize] * filter)
#
#    return filteredImage


def sobelEdgeDetection(singleChannelFrame, threshold=.005):
    """
    Perform Sobel edge detection on an image, returning a binarized
    version that should help in tracking circles.

    Parameters
    ----------

    singleChannelFrame : np.uint8[H, W]
        The frame to perform circle detection on. Should not include multiple
        color channels; if it does contain multiple channels, the first one will
        be used.

    threshold : float
        Value between 0 and 1 representing the percentage of the maximum image
        intensity under which the binarized image will have a value of 0.

    Returns
    -------

    binarizedImage : np.uint8[H,W]
        Binarized image (containing only values of 0 and 1).
    """

    if singleChannelFrame.ndim > 2:
        image = singleChannelFrame.astype(np.float64)[:,:,0]
    else:
        image = singleChannelFrame.astype(np.float64)

    # sobel filter in x and y direction
    image = filters.sobel(image, 0)**2 + filters.sobel(image, 1)**2
    image -= np.min(image)

    # binarize image
    image = np.uint8(image > np.max(image)*threshold)

    return image


def laplacianEdgeDetection(singleChannelFrame, threshold=.005):
    """
    Perform edge detection on an image using a laplacian kernel,
    returning a version that should help in tracking circles.

    Note that this is actually what we refer to as the 'gradient
    squared' elsewhere in the code, so this is mostly just a
    wrapper for that method.

    Parameters
    ----------

    singleChannelFrame : np.uint8[H, W]
        The frame to perform circle detection on. Should not include multiple
        color channels; if it does contain multiple channels, the first one will
        be used.

    threshold : float
        Value between 0 and 1 representing the percentage of the maximum image
        intensity under which the binarized image will have a value of 0.

    Returns
    -------

    binarizedImage : np.uint8[H,W]
        Binarized image (containing only values of 0 and 1).
    """
    # This has to go here otherwise we get a circular import warning
    # Working on thinking of a more permanent solution rn.
    from pepe.analysis import gSquared

    if singleChannelFrame.ndim > 2:
        image = singleChannelFrame.astype(np.float64)[:,:,0]
    else:
        image = singleChannelFrame.astype(np.float64)

    # Laplacian filter
    image = gSquared(image)
    image -= np.min(image)

    # binarize image
    image = np.uint8(image > np.max(image)*threshold)

    return image


def cannyEdgeDetection(singleChannelFrame, threshold=.005):
    return cv2.Canny(singleChannelFrame, 2*threshold, threshold)

