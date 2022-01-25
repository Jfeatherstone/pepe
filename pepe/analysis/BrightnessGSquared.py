import numpy as np

import cv2
import numba

from pepe.preprocess.Image import checkImageType

def averageBrightness(frame):
    r"""
    Compute the average brightness of a frame.

    Parameters
    ----------

    frame : numpy.array
        The frame to compute for (or path to an image)
    """
    return np.mean(checkImageType(frame))


def varianceBrightness(frame):
    r"""
    Compute the standard deviation in brightness of a frame.

    Parameters
    ----------

    frame : numpy.array
        The frame to compute for (or path to an image)
    """
    return np.var(checkImageType(frame))


@numba.jit(nopython=True)
def gSquared(properFrame):
    
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
    Compute the average local gradient squared, or $G^2$, of a frame.

    For more information, see the [DanielsLab Matlab implementation](https://github.com/DanielsNonlinearLab/Gsquared) or:

    Abed Zadeh, A., Bares, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N., Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H. (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular Matter, 21(4), 83. [10.1007/s10035-019-0942-2](https://doi.org/10.1007/s10035-019-0942-2)

    Parameters
    ----------

    frame : numpy.array
        The frame to compute the average of (or path to an image)
    """
    
    # This will load in the image if the method is passed an image
    # file path
    properFrame = checkImageType(frame)

    # Make sure that our image is grayscale
    if len(np.shape(properFrame)) == 3:
        properFrame = properFrame[:,:,0] 

    # Use the optimzed gSquared method so it is fast
    return np.mean(gSquared(properFrame))


def varianceGSquared(frame):
    r"""
    Compute the variance in the local gradient squared, or $G^2$, of a frame.

    For more information, see:

    Abed Zadeh, A., Bares, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N., Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H. (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular Matter, 21(4), 83. [10.1007/s10035-019-0942-2](https://doi.org/10.1007/s10035-019-0942-2)
 
    """

    # This will load in the image if the method is passed an image
    # file path
    properFrame = checkImageType(frame)

    # Make sure that our image is grayscale
    if len(np.shape(properFrame)) == 3:
        properFrame = properFrame[:,:,0] 

    # Use the optimzed gSquared method so it is fast
    return np.var(gSquared(properFrame))
