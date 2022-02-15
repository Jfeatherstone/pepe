import numpy as np
import os

import progressbar

import matplotlib.pyplot as plt

from sklearn.neighbors import kdtree

from pepe.preprocess import checkImageType
from pepe.analysis import initialForceSolve, forceOptimize, gSquared
from pepe.tracking import houghCircle, convCircle
from pepe.simulate import genSyntheticResponse


def forceSolve(imageDirectory, guessRadius, fSigma, pxPerMeter, brightfield, outputFolder, correctionImage=None, maskImage=None, peBlurKernel=3, imageExtension='bmp', imageStartIndex=None, imageEndIndex=None, showProgressBar=True, circleDetectionMethod='convolution', circleTrackingKwargs={}, circleTrackingChannel=0, photoelasticChannel=1):
    """
    Complete pipeline to solve for forces for all image files in a directory.
    """

    # Find all images in the directory
    imageFiles = os.listdir(imageDirectory)

    # This goes before the sorting/extension filtering so we can get more specific
    # error messages (and we have another one of these below)
    if len(imageFiles) < 1:
        print(f'Error: directory {imageDirectory} contains no files!')
        return None

    imageFiles = np.sort([img for img in imageFiles if img[-len(imageExtension):] == imageExtension])

    # We have to do the end index first, so it doesn't mess up the start one
    if imageEndIndex is not None:
        imageFiles = imageFiles[:min(imageEndIndex, len(imageFiles))]

    if imageStartIndex is not None:
        imageFiles = imageFiles[max(imageStartIndex, 0):]

    # Make sure we still have some proper images
    if len(imageFiles) < 1:
        print(f'Error: directory \'{imageDirectory}\' contains no files with extension \'{imageExtension}\'!')
        return None

    if showProgressBar:
        bar = progressbar.ProgressBar(max_value=len(imageFiles))

    # This will calculation the light correction across the images 
    if correctionImage is not None:
        cImageProper = checkImageType(correctionImage)

        height, width = cImageProper.shape[:2]
        verticalMask = np.array([[int(i < maskXBounds[1] and i > maskXBounds[0]) for i in range(width)] for j in range(height)])
        horizontalMask = np.transpose([[int(i < maskYBounds[1] and i > maskYBounds[0]) for i in range(height)] for j in range(width)])

        lightCorrection = lightCorrectionDiff(cImageProper, verticalMask, horizontalMask)
    else:
        # It probably isn't great hygiene to have this variableflip between a single
        # value and an array, but you can always add a scalar to a numpy array, so
        # this is the easiest way (since we haven't loaded any images yet)
        lightCorrection = 0

    # Load up the mask image, which will be used to remove parts of the images
    # that we don't care about, and also potentially indicate which particles
    # are close to the boundary.
    if maskImage is not None:
        maskArr = checkImageType(maskImage)

        if maskArr.ndim > 2:
            maskArr = maskArr[:,:,0]
    else:
        # Same deal as above: scalar multiplication functions exactly how we want
        # in the case that we don't have a mask, so it's just easier to do this.
        maskArr = 1

    # Which method we will be using to detect circles
    if circleDetectionMethod == 'convolution':
        circFunc = convCircle
    elif circleDetectionMethod == 'hough':
        circFunc = houghCircle
    else:
        print(f'Error: circle detection option \'{circleDetectionMethod}\' not recognized!')
        return None

    # The big loop that iterates over every image
    for i in range(len(imageFiles)):

        image = checkImageType(imageDirectory + imageFiles[i])
        # Convert to floats on the domain [0,1], so we can compare to the output of 
        # genSyntheticResponse()
        peImage = cv2.blur((image[:,:,photoelasticChannel] + lightCorrection).astype(np.float64) / 255, (peBlurKernel,peBlurKernel))

        # -------------
        # Track circles
        # -------------
        centers, radii = circFunc(image[:,:,circleTrackingChannel] * maskArr, guessRadius, **circleTrackingKwargs)

        # We do some indexing using the centers/radii, so it is helpful
        # to have them as an integer type
        centers = centers.astype(np.int64)
        radii = radii.astype(np.int64)

        # We want to keep the order of particles constant, so we make sure
        # that they are (to whatever extent possible) in the same order
        # as the previous frame. This involves finding the closest neighbor
        # from the previous frame.
        centers = preserveOrder(centersArr[-1], centers)
