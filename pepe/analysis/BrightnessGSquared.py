"""
Basic methods for computing metrics based on the pixel intensity within images.
"""

import numpy as np

import cv2
import numba

from pepe.preprocess.Image import checkImageType

def averageBrightness(frame):
    r"""
    Compute the average brightness (pixel intensity) of a frame.

    Parameters
    ----------

    frame : np.ndarray[H,W] or str
        The frame for which to compute the average brightness. Can also
        be a path to an image.


    Returns
    -------

    averageBrightness : float
        The average brightness of the image.
    """
    return np.mean(checkImageType(frame))


def varianceBrightness(frame):
    r"""
    Compute the standard deviation in brightness (pixel intensity) of a frame.

    Parameters
    ----------

    frame : np.ndarray[H,W] or str
        The frame for which to compute the average brightness. Can also
        be a path to an image.


    Returns
    -------

    varianceBrightness: float
        The standard deviation of the brightness of the image.
    """
    return np.var(checkImageType(frame))


@numba.jit(nopython=True)
def gSquared(properFrame):
    """
    The gradient squared at each pixel of the image, also known as the convolution
    of a Laplacian kernel across the image.

    Optimized via `numba`.

    Edge values are padded with values of 0.

    Parameters
    ----------

    properFrame : np.ndarray[H,W]
        An array representing a single channel of an image.


    Returns
    -------

    gSqr : np.ndarray[H,W]
        The gradient squared at every point.


    References
    ----------

    [1] DanielsLab Matlab implementation, https://github.com/DanielsNonlinearLab/Gsquared

    [2] Abed Zadeh, A., Bares, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N.,
    Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H.
    (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular
    Matter, 21(4), 83. [10.1007/s10035-019-0942-2](https://doi.org/10.1007/s10035-019-0942-2)
    """
    
    # Take the full size of the image, though know that the outermost row and
    # column of pixels will be 0
    gSquared = np.zeros_like(properFrame)

    # Iterate over every pixel
    # Regardless of whether we have a good region, G^2 needs a buffer of 1 pixel on each
    # side, so we have to crop down more
    for j in range(1, np.shape(properFrame)[0]-1):
        for k in range(1, np.shape(properFrame)[1]-1):
            # I've put a little picture of which pixels we are comparing
            # for each calculation (O is the current pixel, X are the
            # ones we are calculating)

            # - - -
            # X O X
            # - - -
            g1 = float(properFrame[j, k-1]) - float(properFrame[j, k+1])

            # - X -
            # - O -
            # - X -
            g2 = float(properFrame[j-1, k]) - float(properFrame[j+1, k])

            # - - X
            # - O -
            # X - -
            g3 = float(properFrame[j-1, k+1]) - float(properFrame[j+1, k-1])

            # X - -
            # - O -
            # - - X
            g4 = float(properFrame[j-1, k-1]) - float(properFrame[j+1, k+1])

            gSquared[j,k] = (g1*g1/4.0 + g2*g2/4.0 + g3*g3/8.0 + g4*g4/8.0)/4.0

    return gSquared


def averageGSquared(frame):
    r"""
    Compute the average local gradient squared, or \(G^2\), of a frame.

    If multichannel image is provided, will convert to grayscale by averaging
    over the channels.

    Parameters
    ----------

    frame : np.ndarray[H,W] or str
        The frame for which to compute the average gradient squared. Can also
        be a path to an image.


    Returns
    -------

    averageGSquared : float
        The average gradient squared of the image.


    References
    ----------

    [1] DanielsLab Matlab implementation, https://github.com/DanielsNonlinearLab/Gsquared

    [2] Abed Zadeh, A., Bares, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N.,
    Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H.
    (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular
    Matter, 21(4), 83. [10.1007/s10035-019-0942-2](https://doi.org/10.1007/s10035-019-0942-2)
    """
    
    # This will load in the image if the method is passed an image
    # file path
    properFrame = checkImageType(frame)

    # Make sure that our image is grayscale
    if properFrame.ndim == 3:
        properFrame = np.mean(properFrame, axis=-1)

    # Use the optimzed gSquared method so it is fast
    return np.mean(gSquared(properFrame))


def varianceGSquared(frame):
    r"""
    Compute the variance in the local gradient squared, or \(G^2\), of a frame.

    If multichannel image is provided, will convert to grayscale by averaging
    over the channels.

    Parameters
    ----------

    frame : np.ndarray[H,W] or str
        The frame for which to compute the variance of the gradient squared. Can also
        be a path to an image.


    Returns
    -------

    varianceGSquared : float
        The variance of the gradient squared of the image.
 

    References
    ----------

    [1] DanielsLab Matlab implementation, https://github.com/DanielsNonlinearLab/Gsquared

    [2] Abed Zadeh, A., Bares, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N.,
    Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H.
    (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular
    Matter, 21(4), 83. [10.1007/s10035-019-0942-2](https://doi.org/10.1007/s10035-019-0942-2)
    """

    # This will load in the image if the method is passed an image
    # file path
    properFrame = checkImageType(frame)

    # Make sure that our image is grayscale
    if properFrame.ndim == 3:
        properFrame = np.mean(properFrame, axis=-1)

    # Use the optimzed gSquared method so it is fast
    return np.var(gSquared(properFrame))
