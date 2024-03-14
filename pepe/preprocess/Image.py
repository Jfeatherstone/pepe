"""
Image loading and light gradient correction.
"""
import numpy as np

from PIL import Image
import cv2
import rawpy
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

IMAGE_TYPES = ['jpg',
               'jpeg',
               'png',
               'bmp',
               'tiff',
               'nef']

def imageType(file):
    """
    Returns the image type of a file.

    More or less just extracts the extension, but will return
    `None` if the extension is not that of an image, so can be
    used in the following way:

        listOfFiles = ['a.png', 'b.bmp', 'c.txt', ...]
        imageFiles = [f for f in listOfFiles if imageType(f)]

    Parameters
    ----------

    file : str or Path

    Returns
    -------

    imageExt : str or None
        Image extension if file is an image, else None.
    """
    isImage = file.lower().endswith(tuple(IMAGE_TYPES))
    return file.split('.')[-1].lower() if isImage else None

# In reading raw nef files, rawpy will automatically
# do some post processing. These kwargs seem to give
# me the most natural recreation of the original image
# (or at least in comparing to the unedited version in
# darktable)
DEFAULT_NEF_PROCESS_KWARGS = {"no_auto_scale": True,
                              "bright": .1,
                             }

def checkImageType(frame, nef_kwargs=DEFAULT_NEF_PROCESS_KWARGS):
    """
    Make sure that the image is a proper image, and not a path

    Parameters
    ----------

    frame : str or numpy.ndarray
        Either a path to an image, or an image array

    nef_kwargs : dict
        Dictionary of arguments of postprocessing of NEF
        images by rawpy. Only relevant if `frame` is the path
        to an NEF file.

        For full list of possible parameters, see:
        https://letmaik.github.io/rawpy/api/rawpy.Params.html#rawpy.Params

    Returns
    -------

    numpy.ndarray : The image array

    """
    if isinstance(frame, str):
        # I don't want to overwrite the image itself, so create a new var for that
        # Also, we need to open nef images in a special way, since they are a special
        # format (uses rawpy)
        if imageType(frame) == 'nef':
            raw = rawpy.imread(frame)
            newFrame = raw.postprocess(**nef_kwargs)
        # Otherwise, we can just read the image normally using Pillow
        else:
            newFrame = np.array(Image.open(frame), dtype=np.uint8)
    else:
        newFrame = frame

    return newFrame


def lightCorrectionDiff(calibrationImage, verticalMask=None, horizontalMask=None, smoothCorrection=True, debug=False, smoothingKernel=31, channel=0, rectify=False):
    """
    
    """

    calImg = checkImageType(calibrationImage)

    # Slightly different behavior depending on whether we are passed a multichannel
    # image vs a grayscale one. For multichannel, we calculate a correction for each
    # channel separately.
    if calImg.ndim == 3:
        imgSize = calImg.shape[:2]
        numChannels = calImg.shape[-1]
    else:
        imgSize = calImg.shape
        numChannels = 1
        # Add a third dim, so we can treat multi/single channel images
        # exactly the same way
        calImg = calImg[:,:,None]

    if verticalMask is None:
        verticalMask = np.ones(imgSize)

    if horizontalMask is None:
        horizontalMask = np.ones(imgSize)

    fullLightCorrection = np.zeros((*imgSize, numChannels))

    for i in range(numChannels):

        verticallyMasked = calImg[:,:,i] * verticalMask
        horizontallyMasked = calImg[:,:,i] * horizontalMask

        # If there are no non-zero pixels, we just move on to the next channel
        # (and leave this correction as an array of zeros)
        if len(np.where(verticallyMasked != 0)[0]) == 0:
            continue

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
        fullLightCorrection[:,:,i] = np.add.outer(verticalCorrection, horizontalCorrection)

        if rectify:
            fullLightCorrection[:,:,i] -= np.min(fullLightCorrection[:,:,i])

    # If we have a single channel image originally, we want to keep the same shape
    # for our return value -- so that the return can immediately be multipled by the
    # original image -- so we remove the last channel dimension
    if numChannels == 1:
        fullLightCorrection = fullLightCorrection[:,:,0]

    if debug:
        if numChannels > 1:
            fig, ax = plt.subplots(2, 3, figsize=(12, 8))

            channelNames = ['Red', 'Green', 'Blue']
            
            for i in range(3):
                ax[0,i].imshow(calImg[:,:,i])
                ax[0,i].set_title(f'Original {channelNames[i]} Channel')
                ax[1,i].imshow(calImg[:,:,i] + fullLightCorrection[:,:,i])
                ax[1,i].set_title(f'Corrected {channelNames[i]} Channel')

        else:
            fig, ax = plt.subplots(1, 2, figsize=(8,4))

            ax[0].imshow(calImg[:,:,0])
            ax[0].set_title('Original Image')
            ax[1].imshow(calImg[:,:,0] + fullLightCorrection)
            ax[1].set_title('Corrected Image')

        fig.tight_layout()
        plt.show()

    return fullLightCorrection.astype(calibrationImage.dtype)
