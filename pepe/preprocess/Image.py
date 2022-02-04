import numpy as np

from PIL import Image
import cv2
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

def checkImageType(frame):
    """
    Make sure that the image is a proper image, and not a path

    Parameters
    ----------

    frame : str or numpy.ndarray
        Either a path to an image, or an image array

    Returns
    -------

    numpy.ndarray : The image array

    """
    if isinstance(frame, str):
        # I don't want to overwrite the image itself, so create a new var for that
        newFrame = np.array(Image.open(frame), dtype=np.uint8)
    else:
        newFrame = frame

    return newFrame


def lightCorrectionDiff(calibrationImage, verticalMask=None, horizontalMask=None, smoothCorrection=True, debug=False, smoothingKernel=31, channel=0, rectify=False):
    """
    
    """

    calImg = checkImageType(calibrationImage)
    if calImg.ndim == 3:
        calImg = calImg[:,:,channel]

    if verticalMask is None:
        verticalMask = np.ones_like(calImg)

    if horizontalMask is None:
        horizontalMask = np.ones_like(calImg)

    verticallyMasked = calImg * verticalMask
    horizontallyMasked = calImg * horizontalMask

    # This excludes all values of zero, so that we get an actual pixel value we can directly add
    brightnessByRow = np.nanmean(np.where(verticallyMasked != 0, verticallyMasked, np.nan), axis=1)
    brightnessByColumn = np.nanmean(np.where(horizontallyMasked != 0, horizontallyMasked, np.nan), axis=0)

    # Now smooth the two curves
    # Can't say I know much about this filter, but it seems to work pretty well
    if smoothCorrection:
        smoothedBrightnessByColumn = savgol_filter(brightnessByColumn, smoothingKernel, 1)
        smoothedBrightnessByRow = savgol_filter(brightnessByRow, smoothingKernel, 1)
    else:
        smoothedBrightnessByColumn = brightnessByColumn
        smoothedBrightnessByRow = brightnessByRow

    # Now calculate the correction
    horizontalCorrection = np.mean(smoothedBrightnessByColumn) - smoothedBrightnessByColumn
    verticalCorrection = np.mean(smoothedBrightnessByRow) - smoothedBrightnessByRow

    # This object will have the same size as the image, and can just added to
    # any similar image to correct detected light gradients
    totalCorrection = np.add.outer(verticalCorrection, horizontalCorrection)

    if rectify:
        totalCorrection -= np.min(totalCorrection)

    if debug:
        fig, ax = plt.subplots(3, 2, figsize=(10, 10))

        ax[0,0].imshow(verticallyMasked, cmap=plt.get_cmap('terrain'))
        ax[0,1].imshow(horizontallyMasked, cmap=plt.get_cmap('terrain'))

        ax[1,0].plot(brightnessByRow, label='Average')
        ax[1,0].set_ylabel('Image brightness')
        ax[1,0].set_xlabel('Y [pixels]')
        ax[1,0].plot(smoothedBrightnessByRow, label='Smoothed')
        ax[1,0].legend()

        ax[1,1].plot(brightnessByColumn, label='Average')
        ax[1,1].set_ylabel('Image brightness')
        ax[1,1].set_xlabel('X [pixels]')
        ax[1,1].plot(smoothedBrightnessByColumn, label='Smoothed')
        ax[1,1].legend()

        ax[2,0].imshow(totalCorrection)
        ax[2,0].set_title('Correction')
        ax[2,1].imshow(calImg + totalCorrection)
        ax[2,1].set_title('Corrected image')

        fig.tight_layout()
        plt.show()

    return totalCorrection
