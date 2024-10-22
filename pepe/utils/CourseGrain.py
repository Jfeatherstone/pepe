import numpy as np
from scipy.signal import convolve


def courseGrainField(points, values=None, defaultValue=0, latticeSpacing=None, defaultLatticeSize=100, fixedBounds=None, kernel='gaussian', kernelSize=5, subsample=None, returnSpacing=False, returnCorner=False):
    """
    Course grains a collection of values at arbitrary points,
    into a discrete field.

    If `values=None`, course-grained field is the point density.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Spatial positions of N points in d-dimensional space.

    values : numpy.ndarray[N,[k]] or func(points)->numpy.ndarray[N,[k]] or None
        Field values at each point. Can be k-dimensional vector,
        resulting in k course-grained fields.

        Can also be a (vectorized) function that returns a value given
        a collection of points. eg. neighbor counting function. This
        functionality is provided such that if the function is computationally
        expensive, eg. neighbor counting, the points can be subdivided into
        batches and the course grained fields can be summed at the end. This
        is a way to approximate the course grained field for a huge (>1e6)
        number of points, while still remaining computationally feasible.
        See `subsample`.

        If `None`, returned field will be the point density.

    defaultValue : float or numpy.ndarray[k]
        The default value of the course-grained field;
        probably `0` for most applications.

    latticeSpacing : float or None
        The spacing of lattice points for the course-grained field.

        If `None`, will be chosen such that each axis has
        `defaultLatticeSize` points.

    defaultLatticeSize : int
        The number of lattice points for the course grained field, assuming
        no explicit value for the lattice spacing is given (see `latticeSpacing`).

    fixedBounds : numpy.ndarray[d] or None
        The bounds of the field to define the discretized
        grid over. If None, will be calculated based on the
        extrema of the provided points.

    kernel : str or numpy.ndarray[A,A]
        The kernel to course-grain the field with. 'gaussian'
        option is implemented as default, but a custom matrix
        can be provided. If using default gaussian option,
        kernel size can be set with `kernelSize`.

    kernelSize : int
        The kernel size to use if `kernel='gaussian'`.
        If a custom kernel is provided, this has no effect.

    returnSpacing : bool

    returnCorner : bool
    """
    # TODO: Make sure this works for 1D data
    dim = np.shape(points)[-1] if len(np.shape(points)) > 1 else 1

    if dim == 1:
        points = np.array(points)[:,None]
    
    if not hasattr(fixedBounds, '__iter__'):
        occupiedVolumeBounds = np.array(list(zip(np.min(points, axis=0), np.max(points, axis=0))))
    else:
        occupiedVolumeBounds = np.array(fixedBounds)
    
    # Create a lattice with the selected scale for that cube
    if latticeSpacing is not None:
        spacing = latticeSpacing
        # We also have to correct the occupied volume bounds if we were provided with
        # a fixed set of bounds. Otherwise, we will end up with an extra bin at the
        # end
        if hasattr(fixedBounds, '__iter__'):
            occupiedVolumeBounds[:,1] -= spacing
    else:
        # Choose such that each axis has 100 lattice points (-1 because we will add one later)
        spacing = (occupiedVolumeBounds[:,1] - occupiedVolumeBounds[:,0]) / (defaultLatticeSize-1)

    # In the exceptional case that the data is given as a d dimensional array
    # but the data is actually d-1 dimensional (or d-2, etc.), we will have a value
    # of spacing for that dimension as 0, which will cause a divide by zero error
    # above. If this is the case, we only need a single entry in that dimension.
    if hasattr(spacing, '__iter__'):
        spacing[spacing == 0] = 1
    else:
        spacing = spacing if spacing != 0 else 1
    
    fieldDims = (np.ceil(1 + (occupiedVolumeBounds[:,1] - occupiedVolumeBounds[:,0])/(spacing))).astype(np.int64)

    latticePositions = np.floor((points - occupiedVolumeBounds[:,0])/(spacing)).astype(np.int64)

    # Check if an array of values was passed for each point
    # Otherwise we just have a scalar field (and we'll collapse
    # the last dimension later on).
    if hasattr(values, '__iter__'):
        k = np.shape(values)[-1]
        valArr = values
    else:
        k = 1
        valArr = np.ones((np.shape(points)[0], 1))

    fieldArr = np.zeros((*fieldDims, k))
    # Instead of actually applying a gaussian kernel now, which would be
    # very inefficient since we'd need to sum a potentially very large number
    # of k*d dimensional matrices (more or less), we instead just assign each
    # lattice point, then smooth over it after with the specified kernel.
    # Where this might cause issues:
    # - If the lattice spacing is too large, you will get some weird artifacts
    #   from this process. Though in that case, you'll get a ton of artifacts from
    #   elsewhere too, so just don't use too large a lattice spacing :)
    #print(tuple(latticePositions[0]))
    for i in range(np.shape(points)[0]):
        fieldArr[tuple(latticePositions[i])] += valArr[i]

    # Now smooth over the field
    if kernel == 'gaussian':
        gaussianBlurKernel = np.zeros(np.repeat(kernelSize, np.shape(points)[-1]))
        singleAxis = np.arange(kernelSize)
        kernelGrid = np.meshgrid(*np.repeat([singleAxis], np.shape(points)[-1], axis=0))
        #kernelGrid = np.meshgrid(singleAxis, singleAxis, singleAxis)
        # No 2 prefactor in the gaussian denominator because I want the kernel to
        # decay nearly to 0 at the corners
        kernelArr = np.exp(-np.sum([(kernelGrid[i] - (kernelSize-1)/2.)**2 for i in range(np.shape(points)[-1])], axis=0) / (kernelSize))
        # Now account for however many dimensions k we have
        #kernelArr = np.repeat([kernelArr] if k > 1 else kernelArr, k, axis=0)

    # Otherwise, we expect that kernel should already be passed as a
    # proper square d-dimensional matrix
    else:
        kernelArr = kernel

    # Perform a convolution of the field with our kernel
    # 'same' keeps the same bounds on the field, but might cause
    # some weird effects near the boundaries
    # Divide out the sum of the kernel to normalize
    transConvolution = np.zeros_like(fieldArr.T)

    for i in range(k):
        # Note that convolve(x, y) == convolve(x.T, y.T).T
        # We need this so we can go over our k axis
        transConvolution[i] = convolve(fieldArr.T[i], kernelArr.T, mode='same') / np.sum(kernelArr)

    convolution = transConvolution.T

    # If k == 1, collapse the extra dimension
    if k == 1:
        convolution = convolution[..., 0]
    
    returnResult = [convolution]

    if returnSpacing:
        returnResult += [spacing]

    if returnCorner:
        returnResult += [occupiedVolumeBounds[:,0]]

    return returnResult if len(returnResult) > 1 else convolution

