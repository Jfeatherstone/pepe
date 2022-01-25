import numpy as np

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
