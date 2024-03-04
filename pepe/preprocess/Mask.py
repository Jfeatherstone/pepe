"""
Image mask creation and manipulation.
"""
import numpy as np

import cv2

import alphashape as ap

import numba

@numba.jit(nopython=True)
def crossMask(size, ylim=[0,0], xlim=[0,0], channels=3):
    """
    Create a mask in the shape of a cross, in which the inside of the cross is
    the desired region.

    A stripe (horizontal or vertical) can be made by only supplying a single set
    of limits, either for x (vertical) or y (horizontal).

    Parameters
    ----------

    size : [int, int] or [int, int, int]
        The size of the mask to be created (height, width[, channels]). Can include number of channels.
        If not included, default is 3; see `channels`.

    ylim : [int, int]
        The y coordinates of the (beginning and end of the) desired horizontal region

    xlim : [int, int]
        The x coordinates of the (beginning and end of the) desired vertical region

    channels : int
        The number of channels to create for the mask, if not included in `size` parameter.

    Returns
    -------

    np.uint8[3] : Mask array (height, width, channels)
    """

    # If channels are included in the size, use that
    if len(size) == 3:
        channels = size[2]

    maskArr = np.zeros((size[0], size[1], channels), dtype=np.uint8)

    for i in range(size[0]):
        for j in range(size[1]):
            if (i >= ylim[0] and i <= ylim[1]):
                maskArr[i,j] = np.ones(channels)
            elif (j >= xlim[0] and j <= xlim[1]):
                maskArr[i,j] = np.ones(channels)

    return maskArr


@numba.jit(nopython=True)
def rectMask(size, corner=[0,0], dimensions=[0,0], channels=3):
    """
    Create a mask in the shape of a (non-rotated) rectangle, in which the inside
    of the rectangle is the desired region.

    Parameters
    ----------

    size : [int, int] or [int, int, int]
        The size of the mask to be created (height, width[, channels]). Can include
        number of channels. If not included, default is 3; see `channels`.

    corner : [int, int]
        The [y, x] coordinates of the top-left corner of the desired region. 

    dimensions : [int, int]
        The [height, width] of the desired region. If either is set to 0, the
        entire rest of the image will be included in that direction.

    channels : int
        The number of channels to create for the mask, if not included in `size` parameter.

    Returns
    -------

    np.uint8[3] : Mask array (height, width, channels)
    """

    # If channels are included in the size, use that
    # Though if the channels variable is explicitly set to None, that overrides this
    if len(size) == 3:
        channels = size[2]

    xlim = [corner[1], corner[1] + dimensions[1]]
    ylim = [corner[0], corner[0] + dimensions[0]]

    maskArr = np.zeros((size[0], size[1], channels), dtype=np.uint8)
    
    for i in range(size[0]):
        for j in range(size[1]):
            if (i >= ylim[0] and i <= ylim[1] and j >= xlim[0] and j <= xlim[1]):
                maskArr[i,j] = np.ones(channels)

    return maskArr

@numba.jit(nopython=True)
def circularMask(size, center, radius, channels=3):
    """
    Create a mask in the shape of a circle, in which the inside of the circle is
    the desired region.

    Parameters
    ----------

    size : [int, int] or [int, int, int]
        The size of the mask to be created (height, width[, channels]). Can include number of channels.
        If not included, default is 3; see `channels`.

    center : [int, int]
        The [y, x] coordinates of the center of the desired region. 

    radius : int
        The radius of the desired region.

    channels : int
        The number of channels to create for the mask, if not included in `size` parameter.

    Returns
    -------

    np.uint8[3] : Mask array (height, width, channels)
    """

    # If channels are included in the size, use that
    if len(size) == 3:
        channels = size[2]

    # numba doesn't support ogrid, so we just do it manually
    #Y, X = np.ogrid[:size[0], :size[1]]
    Y = np.arange(size[0]).reshape((size[0], 1)) # Column vector
    X = np.arange(size[1]).reshape((1, size[1])) # Row vector

    dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)

    maskArr = np.zeros((size[0], size[1], channels), np.uint8)
    for i in range(channels):
        maskArr[:,:,i] = dist_from_center <= radius

    return maskArr


@numba.jit(nopython=True)
def ellipticalMask(size, majorAxisEndpoint1, majorAxisEndpoint2, minorAxisLength, channels=3):
    """
    Create a mask in the shape of a circle, in which the inside of the circle is
    the desired region.

    Parameters
    ----------

    size : [int, int] or [int, int, int]
        The size of the mask to be created (height, width[, channels]). Can include number of channels.
        If not included, default is 3; see `channels`.

    majorAxisEndpoint1 : [int, int]
        The first of two sets of [y, x] coordinates that define the major axis of the desired region. 
        Order does not matter.

    majorAxisEndpoint2 : [int, int]
        The second of two sets of [y, x] coordinates that define the major axis of the desired region. 
        Order does not matter.

    minorAxisLength : int
        The length of the minor axis, or the extent of the shorter dimension of the ellipse.

    channels : int
        The number of channels to create for the mask, if not included in `size` parameter.

    Returns
    -------

    np.uint8[3] : Mask array (height, width, channels)
    """

    minorAxisLength = float(minorAxisLength)

    # If channels are included in the size, use that
    # Though if the channels variable is explicitly set to None, that overrides this
    if len(size) == 3:
        channels = size[2]

    # Center of ellipse
    center = np.array([(majorAxisEndpoint1[0] + majorAxisEndpoint2[0])/2,
              (majorAxisEndpoint1[1] + majorAxisEndpoint2[1])/2], dtype=np.float64)

    # Unit vector along major axis
    eMajor = np.array([majorAxisEndpoint1[0] - majorAxisEndpoint2[0],
                       majorAxisEndpoint1[1] - majorAxisEndpoint2[1]], dtype=np.float64)
    eMajor /= np.sqrt(eMajor[0]**2 + eMajor[1]**2)

    semiMajorAxis = np.sqrt((majorAxisEndpoint1[0] - majorAxisEndpoint2[0])**2 + (majorAxisEndpoint1[1] - majorAxisEndpoint2[1])**2)/2
    semiMinorAxis = minorAxisLength/2

    # Distance from center to either focus part of a triangle with axes
    focalDistance = np.sqrt(semiMajorAxis**2 - semiMinorAxis**2)

    # Foci of the ellipse
    foci = np.zeros((2, 2))
    foci[0,0] = center[0] + focalDistance * eMajor[0]
    foci[0,1] = center[1] + focalDistance * eMajor[1]
    foci[1,0] = center[0] - focalDistance * eMajor[0]
    foci[1,1] = center[1] - focalDistance * eMajor[1]
    #foci = [[center[0] + focalDistance * eMajor, center[1] + focalDistance * eMajor],
    #        [center[0] - focalDistance * eMajor, center[1] - focalDistance * eMajor]]

    # For any point outside of the ellipse, the sum of distance from both foci will
    # be greater than the length of the major axis

    # numba doesn't support ogrid, so we just do it manually
    #Y, X = np.ogrid[:size[0], :size[1]]
    Y = np.arange(size[0]).reshape((size[0], 1)) # Column vector
    X = np.arange(size[1]).reshape((1, size[1])) # Row vector

    fociDistanceSum = np.sqrt((X - foci[0][1])**2 + (Y - foci[0][0])**2) + np.sqrt((X - foci[1][1])**2 + (Y - foci[1][0])**2) 
    
    maskArr = np.zeros((size[0], size[1], channels), dtype=np.uint8)
    for i in range(channels):
        maskArr[:,:,i] = fociDistanceSum <= 2*semiMajorAxis

    return maskArr


@numba.jit(nopython=True)
def mergeMasks(listOfMasks, signs=None, channels=3):
    """
    Merge several masks together into a single new mask, serving as the union of the
    previous set.

    Parameters
    ----------

    listOfMasks : iterable of np.uint8[2] or np.uint8[3]
        A list of the masks to be merged together. If they have different numbers of channels, the
        union will follow the `channels` parameter; otherwise, original shape will be preserved.

    signs : iterable of int
        Array of 1 or -1 that indicates which masks should be added to the final result, and which should
        be subtracted. By default, all are added.

    channels : int
        The number of channels to create for the final mask; used only if not consistent in `listOfMasks` parameter.

    Returns
    -------

    np.uint8[2] or np.uint8[3] : Mask array; can include channels in last dim, or not.
    """

    if signs is None:
        signs = np.ones(len(listOfMasks), dtype=np.int16)

    sumMask = np.zeros((*np.shape(listOfMasks[0])[:2], channels), dtype=np.int16)

    for i in range(len(listOfMasks)):
        for j in range(channels):
            # Assume shape of [H, W, C]
            sumMask[:,:,j] += listOfMasks[i][:,:,0] * signs[i]
                

    #sumMask += np.abs(np.min(sumMask))
    #sumMask = np.ceil(sumMask / np.max(sumMask))
    return (sumMask > 0).astype(np.uint8)


def identifyFeatureBoundary(image,
                            approxFeatureLocation,
                            featureSize,
                            meanThresholdFactor=2,
                            minIslandSize=5,
                            pointDownsampleFactor=5,
                            concaveHullParameter=0,
                            debug=False):
    """
    Detect the boundary of a feature near the specified location.

    Feature detection is done using the same algorithm as in
    `pepe.preprocess.maskFeature()`. Feature should be
    brighter than the background.

    Parameters
    ----------
    image : numpy.ndarray[H,W[,C]]
        Input image to identify feature in. If multi-channel,
        the channels will be averaged (gray-scaled).

    approxFeatureLocation : (float, float)
        Approximate coordinates (y,x) of the location of the feature.

    meanThresholdFactor : float
        Parameter for determining which pixels are a part of the feature.
        Lower value means the detected feature will be larger.

    minIslandSize : int
        Parameter for culling pixels initially detected to be a part of
        the feature, but likely are not actually. eg. If a value of 5,
        is chosen, only pixels surrounded by 24 other pixels detected to
        be a part of the feature will be kept.

        If not an odd number, the value will be rounded up to the next odd
        number.

    pointDownsampleFactor : int >= 1
        Parameter for generating the hull around the points detected to be
        a part of the feature. Higher value means more approximate shape,
        but faster calculations. Value of `1` will give the most accurate
        result.

    concaveHullParamter : float
        Alphashape parameter for generating the hull. A value of `0`
        means the hull will be convex, a value `>= 0` will mean the
        hull is increasingly concave.

        For more information, see [alphashape documentation](https://alphashape.readthedocs.io/en/latest/readme.html)
        related to the 'alpha parameter'.

    debug : bool
        Whether to plot debug information about the feature
        detection once completed or not.


    Returns
    -------
    boundaryPoints : numpy.ndarray[N,2]
        List of N points that comprise the boundary ordered
        such that the boundary can be drawn by drawing all pairs
        of points (i,i+1) (and (-1,0) for the final line).
    """
    # Grayscale if necessary
    if len(np.shape(image)) == 3:
        grayImage = np.mean(image, axis=-1)
    else:
        grayImage = image

    # Crop
    croppedImage = grayImage[approxFeatureLocation[0]-int(featureSize):approxFeatureLocation[0]+int(featureSize),
                                approxFeatureLocation[1]-int(featureSize):approxFeatureLocation[1]+int(featureSize)]

    # Identify regions that are above the mean
    thresholdedImage = np.float64(croppedImage > meanThresholdFactor*np.mean(croppedImage))

    # Blur and then threshold again, meaning that anything who isn't surrounded by
    # `minIslandSize/2` neighbors in each direction gets removed
    kernel = minIslandSize + int(not minIslandSize%2) # Make sure it is odd
    thresholdedImage = np.float64(cv2.blur(thresholdedImage, (kernel,kernel)) == 1)

    # Generate a convex hull
    alphashape = ap.alphashape(np.array(np.where(thresholdedImage)).T[::pointDownsampleFactor], concaveHullParameter)
    
    boundaryPoints = np.array(alphashape.boundary.coords)[:,:2:][:,::-1]

    if debug:
        fig, ax = plt.subplots(1, 2, figsize=(8,4))

        ax[0].imshow(thresholdedImage)
        ax[1].imshow(grayImage)
        for i in range(len(boundaryPoints-1)):
            ax[1].plot(boundaryPoints[i:i+2,0], boundaryPoints[i:i+2,1], c='red', alpha=1)

        ax[0].set_title('Thresholded Image')
        ax[1].set_title('Identified Feature Boundary')
        plt.show()

    # Convert local coordinates back to absolute image coordinates
    return approxFeatureLocation + boundaryPoints - featureSize

def maskFeature(image,
                approxFeatureLocation,
                featureSize,
                meanThresholdFactor=1,
                minIslandSize=5,
                pointDownsampleFactor=5,
                concaveHullParameter=0,
                extraPadding=0,
                allowRecurse=True,
                debug=False):
    """
    Generate a mask to remove a detected feature near the specified location.

    Feature detection is done using the same algorithm as in
    `pepe.preprocess.detectFeatureBoundary()`. Feature should be
    brighter than the background.

    Parameters
    ----------
    image : numpy.ndarray[H,W[,C]]
        Input image to identify feature in. If multi-channel,
        the channels will be averaged (gray-scaled).

    approxFeatureLocation : (float, float)
        Approximate coordinates (y,x) of the location of the feature.

    meanThresholdFactor : float
        Parameter for determining which pixels are a part of the feature.
        Lower value means the detected feature will be larger.

    minIslandSize : int
        Parameter for culling pixels initially detected to be a part of
        the feature, but likely are not actually. eg. If a value of 5,
        is chosen, only pixels surrounded by 24 other pixels detected to
        be a part of the feature will be kept.

        If not an odd number, the value will be rounded up to the next odd
        number.

    pointDownsampleFactor : int >= 1
        Parameter for generating the hull around the points detected to be
        a part of the feature. Higher value means more approximate shape,
        but faster calculations. Value of `1` will give the most accurate
        result.

    concaveHullParamter : float
        Alphashape parameter for generating the hull. A value of `0`
        means the hull will be convex, a value `>= 0` will mean the
        hull is increasingly concave.

        For more information, see [alphashape documentation](https://alphashape.readthedocs.io/en/latest/readme.html)
        related to the 'alpha parameter'.

    extraPadding : int
        Extra padding at the end of the feature identification
        added to the shape.

    allowRecurse : bool
        If the  detected feature is larger than the size,
        and this value is `True`, the function will recursively
        call itself (at most once) with a larger feature size to try
        and detect the entire feature.
        
    debug : bool
        Whether to plot debug information about the feature
        detection once completed or not.


    Returns
    -------
    imageMask : numpy.ndarray[H,W]
        Image mask with values of `1` everywhere
        except for points that are detected to be a
        part of the feature, which will have values
        of `0`.

    """
    # Grayscale if necessary
    if len(np.shape(image)) == 3:
        grayImage = np.mean(image, axis=-1)
    else:
        grayImage = image

    # Crop
    croppedImage = grayImage[max(approxFeatureLocation[0]-int(featureSize), 0):min(approxFeatureLocation[0]+int(featureSize), grayImage.shape[0]),
                                max(approxFeatureLocation[1]-int(featureSize), 0):min(approxFeatureLocation[1]+int(featureSize), grayImage.shape[1])]

    # Identify regions that are above the mean
    thresholdedImage = np.float64(croppedImage > meanThresholdFactor*np.mean(croppedImage))

    # Blur and then threshold again, meaning that anything who isn't surrounded by
    # `minIslandSize/2` neighbors in each direction gets removed
    kernel = minIslandSize + int(not minIslandSize%2) # Make sure it is odd
    thresholdedImage = np.float64(cv2.blur(thresholdedImage, (kernel,kernel)) == 1)

    # Generate a convex hull
    alphashape = ap.alphashape(np.array(np.where(thresholdedImage)).T[::pointDownsampleFactor], concaveHullParameter)

    if alphashape is not None:
        boundaryPoints = np.array(alphashape.boundary.coords)[:,:2:][:,::-1]
    else:
        boundaryPoints = None

    # If any of the boundary points are at the edge of the image, that means the 
    # feature extends beyond the cropped region, and we should try again with a slightly
    # larger region
    edgePadding = 5
    if (True in (boundaryPoints.flatten() < edgePadding) or True in (boundaryPoints.flatten() + edgePadding > 2*featureSize)) and allowRecurse:
        # We could try and recenter the crop window, but that doesn't actually work all that well...
        # Maybe since averaging the boundary points doesn't actually calculate the center of mass
        #newFeatureCenter = approxFeatureLocation + np.int64(np.mean(boundaryPoints, axis=0)) - featureSize
        newFeatureCenter = approxFeatureLocation
        return maskFeature(image, newFeatureCenter, featureSize*1.5,
                           meanThresholdFactor, minIslandSize, pointDownsampleFactor,
                           concaveHullParameter, False, debug)
        
    maskImage = np.zeros_like(croppedImage, dtype=np.uint8)

    for i in range(len(boundaryPoints)-1):
        cv2.line(maskImage, np.int64(boundaryPoints[i]), np.int64(boundaryPoints[i+1]), 1, minIslandSize)        
    cv2.line(maskImage, np.int64(boundaryPoints[-1]), np.int64(boundaryPoints[0]), 1, minIslandSize)
                    
    outsidePoint = (0,0)
    cv2.floodFill(maskImage, None, outsidePoint, 2)

    # Add extra padding
    if extraPadding > 0:
        oddPadding = extraPadding + int(not extraPadding%2) # Make sure it is odd
        maskImage = cv2.blur(maskImage, (oddPadding, oddPadding))

    maskImage = maskImage == 2

    fullImageMask = np.ones(image.shape[:2], dtype=np.uint8)
    fullImageMask[max(approxFeatureLocation[0]-int(featureSize), 0):min(approxFeatureLocation[0]+int(featureSize), grayImage.shape[0]),
                                max(approxFeatureLocation[1]-int(featureSize), 0):min(approxFeatureLocation[1]+int(featureSize), grayImage.shape[1])] = maskImage

    if debug:
        fig, ax = plt.subplots(1, 3, figsize=(12,4))

        ax[0].imshow(thresholdedImage)
        ax[1].imshow(croppedImage)
        for i in range(len(boundaryPoints)-1):
            ax[1].plot(boundaryPoints[i:i+2,0], boundaryPoints[i:i+2,1], c='red', alpha=1)

        ax[2].imshow(fullImageMask)
        
        ax[0].set_title('Thresholded Image')
        ax[1].set_title('Identified Feature Boundary')
        ax[2].set_title('Full Image Mask')
        plt.show()

    return fullImageMask


def polygonMask(boundaryPoints, maskSize, debug=False):
    """
    Create a mask of a polygon defined by the points that
    comprise its boundary.

    Parameters
    ----------
    boundaryPoints : iterable of points
        Iterable of points (y,x) that define the
        boundary of the polygon. Should be ordered
        such that drawing each pair of points (i,i+1)
        creates the desired shape.

    maskSize : [H,W]
        The size of the mask array to create.

    debug : bool
        Whether or not to plot a visualization of
        the mask once it is completed.

    Returns
    -------
    maskImage : numpy.ndarray[H,W]
        Mask array, with values of `1` inside
        the polygon, and `0` outside.

    """
    maskImage = np.ones(maskSize, dtype=np.uint8)

    for i in range(len(boundaryPoints)-1):
        # Have to reverse coordinates
        cv2.line(maskImage, np.int64(boundaryPoints[i][::-1]), np.int64(boundaryPoints[i+1][::-1]), 255, 1)        
    cv2.line(maskImage, np.int64(boundaryPoints[-1][::-1]), np.int64(boundaryPoints[0][::-1]), 255, 1)
                    
    outsidePoint = (0,0)
    cv2.floodFill(maskImage, None, outsidePoint, 255)

    maskImage = (maskImage != 255).astype(np.float64)

    if debug:
        plt.imshow(maskImage)
        plt.show()

    return maskImage


def removePadding(image):
    """
    Remove padding from an image.

    Parameters
    ----------
    image : array-like
        Image to remove excess padding from.

    Returns
    -------
    unpaddedImage : array-like
        Original image with rows and columns
        removed if they contained no image data.
    """
    # Grayscale if necessary
    if len(np.shape(image)) == 3:
        grayImage = np.mean(image, axis=-1)
    else:
        grayImage = image

    rowSum = np.sum(grayImage, axis=0)
    columnSum = np.sum(grayImage, axis=-1)

    # There is probably a much fancier way to do this, but this
    # is fast enough, so... essentially we just count until the sum
    # is not zero anymore (ie there is some image data)
    
    leftEdge = None
    for i in range(len(rowSum)):
        leftEdge = i
        if rowSum[i] > 0:
            break

    rightEdge = None
    for i in range(len(rowSum))[::-1]:
        rightEdge = i
        if rowSum[i] > 0:
            break

    topEdge = None
    for i in range(len(columnSum)):
        topEdge = i
        if columnSum[i] > 0:
            break
            
    bottomEdge = None
    for i in range(len(columnSum))[::-1]:
        bottomEdge = i
        if columnSum[i] > 0:
            break

    return image[topEdge:bottomEdge,leftEdge:rightEdge]

