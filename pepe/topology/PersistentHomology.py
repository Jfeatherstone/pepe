"""
Peak finding in arbitrary dimensional spaces via persistent homology.
"""
import numpy as np

import numba
import matplotlib.pyplot as plt


def _iterNeighbors(p, tShape, neighborInclusion=1, periodic=False):
    """
    Find the indices of all neighbors of an `d`-dimensional point.

    Used primarily in `pepe.topology.multiFindPeaks()`.

    Parameters
    ----------
    p : tuple[d]
        A tuple of indices representing a point in $Z^d$.

    tShape : tuple[d]
        The size of the grid/tensor in each dimension, such that
        points outside of the domain can be removed.

    neighborInclusion : int
        The number of points on each side of the point in each dimension
        to include as neighbors.

    periodic : bool
        Whether points should be wrapped around to the opposite side
        at the boundaries.

    Returns
    -------
    generator[tuple]
    """
    d = len(p)
    # Convert to numpy array to do math (which can't be done on tuples)
    nP = np.array(p, dtype=np.int16)
    # This will create an array that looks like [-2, -1, 0, 1, 2], [-1, 0, 1], etc.
    # depending on the value of neighborInclusion
    neighborDirections = np.arange(-neighborInclusion, neighborInclusion+1, dtype=np.int16)
        
    # Now find the combinations based on the possible values
    # This piece of code is partially taken from the method
    # itertools.product.
    # More or less the equivalent of itertools.product(neighborDirections, repeat=d)
    result = [[]] 
    for pool in [neighborDirections]*d:
        result = [x+[y] for x in result for y in pool]
       
    # Result is just the difference vectors from our point to the neighbors
    # (eg. [-1, 0] or [1, 1]) so we need to add these to our original point.
    neighbors = [nP + np.array(n) for n in result]
    
    # This element will always be the original point because the ordering for arange is
    # consistent such that the difference vector of [0,0,0,...] will always be at the
    # center (even after permuting for arbitrary dimensions).
    del neighbors[len(neighbors)//2]
   
    # Now make sure all of the points are within the domain of the data
    # (or that we allow points to wrap around with `periodic`).
    for n in neighbors:
        if (not True in ((np.array(tShape) - n - 1) < 0) and not True in (n < 0)) or periodic:
            yield tuple(n % np.array(tShape))


def findPeaksMulti(data, neighborInclusion=1, minPeakPrevalence=None, normalizePrevalence=True, periodic=False, allowOptimize=True):
    """
    Identify peaks in multi-dimensional data using persistent topology.

    Peak prevalence is calculated as the age of topological features,
    which is not necessarily the same as the actual height of the peaks.

    Originally adapted from [1] (and partially from [2]), but generalized for data with
    an arbitrary number of dimensions. For background on persistence, see: [3] for
    application of persistence to signal processing, or [4] for general domain knowledge
    (these are just my personal recommendations, probably plenty of other resources out there).

    May struggle to find peaks in noisier data if the fluctuations due to
    noise are large enough to create a spiky-looking profile. This is because
    such fluctuations do represent topological objects, even if they are not ones
    of any physical relevance; this method does not distinguish between the two.
    In this case, smoothing functions are recommended to be used on the data before
    it is passed to this method. A higher `neighborInclusion` can partially address this.

    Method is not able to be optimized using `numba` (mostly due to the use of tuples,
    which `numba` can't type as easily as a list or array. While this method can identify
    peaks in 1D or 2D data, it is recommended to instead use either `pepe.topology.findPeaks1D()`
    or `pepe.topology.findPeaks2D()`, as these *are* optimized.

    The kwarg `allowOptimize` (Default: True) will instead call either
    `pepe.topology.findPeaks1D()` or `pepe.topology.findPeaks2D()` for 1- or 2-dimensional cases
    instead of the generalized method. This usually results in a 10x-100x speedup.

    Parameters
    ----------
    data : numpy.ndarray[d]
        An array of data points within which to identify peaks in d-dimensional space.

    neighborInclusion : int
        Controls the number of neighbors on either side of a point to look
        at when deciding which peak a given point belongs to. 

        eg. In 1D, a value of `1` means that `[i-1, i+1]` are considered neighbors,
        a value of `2` means that `[i-2, i-1, i+1, i+1]` are considered neighbors, etc.

        May help to supress the effects of noise in data, but should not be treated
        as a substitute for smoothing data properly beforehand.

    minPeakPrevalence : float or None
        The minimum prevalence of a peak (maximum - minimum) that will be
        returned to the user. If `normalizePrevalence` is True, this should
        be in the range `[0, 1]`, representing a percentage of the full range
        the data occupies.

    normalizePrevalence : bool
        Whether to normalize the prevalences such that they represent the
        percent of the data range that is spanned by a peak (True) or not (False).
        For example, if normalized, a prevalence of .4 would mean that the peak
        spans 40% of the total range of the data.

        Note that this normalization happens before using the minPeakPrevalence
        kwarg to clip smaller peaks, and thus the minimum value should be
        specified as a percent (eg. .3 for 30%) is the normalizePrevalence
        kwarg is True.

    periodic : bool
        Whether the discrete field should be wrapped around to itself at the
        boundaries.

    allowOptimize : bool
        Whether to allow the function to call the numba-optimized methods
        specific to 1- or 2-dimensional data when possible instead of
        the generalized method.

        See `pepe.topology.findPeaks1D()` or `pepe.topology.findPeaks2D()` for
        more information.

        Can only be used when `neighborInclusion=1`. WIP to remove this
        constraint.

    Returns
    -------
    peakPositions : numpy.ndarray[N,d]
        The indices of peaks in the provided data, sorted from most
        to least prevalent (persistent).

    peakPrevalences : numpy.ndarray[N]
        The prevalence of each peak in the provided data, or the persistence
        of the topological feature. If `normalizePrevalence` is True, this will
        be normalized to the domain `[0, 1]`; otherwise, these values are
        equivalent to the maximum data value minus the minimum data value evaluated
        over all points that identify with a given peak.

    References
    ----------

    [1] Huber, Stefan. Persistent Topology for Peak Detection. 
    <https://www.sthu.org/blog/13-perstopology-peakdetection/index.html>

    [2] Huber, Stefan. Topological peak detection in two-dimensional data.
    <https://www.sthu.org/code/codesnippets/imagepers.html>

    [3] Huber, S. (2021). Persistent Homology in Data Science. In P. Haber,
    T. Lampoltshammer, M. Mayr, & K. Plankensteiner (Eds.), Data Science – Analytics
    and Applications (pp. 81–88). Springer Fachmedien. <https://doi.org/10.1007/978-3-658-32182-6_13>

    [4] Edelsbrunner, H., & Harer, J. (2010). Computational topology: An introduction.
    Chapter 7: Persistence. p. 149-156. American Mathematical Society. ISBN: 978-0-8218-4925-5

    """ 
    d = np.array(data).ndim
    
    # Call the optimized versions of the peak finding if allowed.
    if allowOptimize and neighborInclusion == 1:
        if d == 1:
            peaks, prev = findPeaks1D(data, minPeakPrevalence, normalizePrevalence, periodic)
            # Have to convert to numpy arrays to be consistent with the
            # unoptimized call return
            return (np.array(peaks), np.array(prev))
        elif d == 2:
            peaks, prev = findPeaks2D(data, minPeakPrevalence, normalizePrevalence, periodic)
            return (np.array(peaks), np.array(prev))
        
    # This array contains the indices of the peak that each point
    # belongs to (assuming it does belong to a peak)
    # We start it at -1 such that we can check if a point belongs to
    # a certain peak with peakMembership[i,j,...] >= 0
    peakMembership = np.zeros_like(data, dtype=np.int16) - 1
    peakBirthIndices = []

    # Go through each value in our data, starting from highest and ending
    # with lowest. This is important because we will be merging points into
    # groups as we go, and so starting with the highest values means that
    # points will almost never have to be overwritten.
    # This is a bit complex here because we could have any number of dimensions, but
    # the gist is that we sort a flattened version of the array from highest to lowest,
    # then turn those 1d indices into Nd indices, then pair the indices together.
    # Each element of this array will be a set of indices that can index a single
    # element of data. 
    # eg. data[sortedIndices[0]] will be the largest value (for any number of dimensions)
    sortedIndices = np.dstack(np.unravel_index(np.argsort(data.flatten())[::-1], data.shape))[0]
    # To be able to actually index data with an element, they need to all be tuples
    sortedIndices = [tuple(si) for si in sortedIndices]
   
    # I've avoided using the index i here since we are iterating over sets of indices,
    # not just a single number
    for si in sortedIndices:
        # See if any neighbors have been assigned to a peak
        assignedNeighbors = [n for n in _iterNeighbors(si, data.shape, neighborInclusion, periodic) if peakMembership[n] >= 0]

        # If there aren't any assigned neighbors yet, then we create a new
        # peak
        if len(assignedNeighbors) == 0:
            peakMembership[si] = len(peakBirthIndices)
            peakBirthIndices.append(si)

        # If only a single one has been assigned, or all of the assigned peaks have the
        # same membership, then this point joins that peak
        elif len(assignedNeighbors) == 1 or len(np.unique([(peakMembership[n] for n in assignedNeighbors)])) == 1:
            peakMembership[si] = peakMembership[assignedNeighbors[0]]

        # Otherwise, we have to resolve a conflict between multiple, in which the
        # oldest one gains the new point.
        else:      
            # Find which one is the oldest
            order = np.argsort(data[peakBirthIndices[(peakMembership[n] for n in assignedNeighbors)]])[::-1]

            # New point joins oldest peak
            peakMembership[si] = peakMembership[assignedNeighbors[order][0]]

    # This is the position of the proper crest for each peak
    peakPositions = peakBirthIndices

    # Calculate the prevalence of each peak as the height of the birth points minus the
    # height of the death point
    # We could do this using the arrays we've stored along the way, but it's easier just to take the
    # max/min heights directly from the data
    peakPrevalences = np.array([data[peakPositions[i]] - np.min(data[peakMembership == i]) for i in range(len(peakBirthIndices))])
    # Also note that I have made the decision here to normalize these prevalences by the total range
    # of the data, such that a prevalence of .6 means that the peak spans 60% of the range of the data.
    # This can be altered with the normalizePrevalence kwarg
    if normalizePrevalence:
        dataRange = np.max(data) - np.min(data)
        peakPrevalences /= dataRange

    # Cut off small peaks, if necessary
    if minPeakPrevalence is not None:
        peakPositions = np.array([peakPositions[i] for i in range(len(peakPositions)) if peakPrevalences[i] > minPeakPrevalence])
        peakPrevalences = np.array([peakPrevalences[i] for i in range(len(peakPrevalences)) if peakPrevalences[i] > minPeakPrevalence])

    # Sort the peaks by their prevalence
    #order = np.argsort(peakPrevalences)[::-1]
    #peakPositions = peakPositions[order]
    #peakPrevalences = peakPrevalences[order]

    return (peakPositions, peakPrevalences)


def findPeaks2D(data, minPeakPrevalence=None, normalizePrevalence=True, periodic=False):
    """
    Identify peaks in 2-dimensional data using persistent topology.

    Peak prevalence is calculated as the age of topological features,
    which is not necessarily the same as the actual height of the peaks.

    Originally adapted from [1] (and partially from [2]), but adjusted for data with
    an specifically 2 dimensions. For background on persistence, see: [3] for
    application of persistence to signal processing, or [4] for general domain knowledge
    (these are just my personal recommendations, probably plenty of other resources out there).

    May struggle to find peaks in noisier data if the fluctuations due to
    noise are large enough to create a spiky-looking profile. This is because
    such fluctuations do represent topological objects, even if they are not ones
    of any physical relevance; this method does not distinguish between the two.
    In this case, smoothing functions are recommended to be used on the data before
    it is passed to this method.

    Method is optimized using `numba`.

    Parameters
    ----------
    data : np.ndarray[H,W]
        A discretized 2-dimensional scalar field within which to identify peaks.

    minPeakPrevalence : float or None
        The minimum prevalence of a peak (maximum - minimum) that will be
        returned to the user. If `normalizePrevalence` is True, this should
        be in the range `[0, 1]`, representing a percentage of the full range
        the data occupies.

    normalizePrevalence : bool
        Whether to normalize the prevalences such that they represent the
        percent of the data range that is spanned by a peak (True) or not (False).
        For example, if normalized, a prevalence of .4 would mean that the peak
        spans 40% of the total range of the data.

        Note that this normalization happens before using the minPeakPrevalence
        kwarg to clip smaller peaks, and thus the minimum value should be
        specified as a percent (eg. .3 for 30%) is the normalizePrevalence
        kwarg is True.

    periodic : bool
        Whether the discrete field should be wrapped around to itself at the
        boundaries.

    Returns
    -------

    peakPositions : np.ndarray[N,d]
        The indices of peaks in the provided data, sorted from most
        to least prevalent (persistent).

    peakPrevalences : np.ndarray[N]
        The prevalence of each peak in the provided data, or the persistence
        of the topological feature. If `normalizePrevalence` is True, this will
        be normalized to the domain `[0, 1]`; otherwise, these values are
        equivalent to the maximum data value minus the minimum data value evaluated
        over all points that identify with a given peak.

    References
    ----------

    [1] Huber, Stefan. Persistent Topology for Peak Detection. 
    <https://www.sthu.org/blog/13-perstopology-peakdetection/index.html>

    [2] Huber, Stefan. Topological peak detection in two-dimensional data.
    <https://www.sthu.org/code/codesnippets/imagepers.html>

    [3] Huber, S. (2021). Persistent Homology in Data Science. In P. Haber,
    T. Lampoltshammer, M. Mayr, & K. Plankensteiner (Eds.), Data Science – Analytics
    and Applications (pp. 81–88). Springer Fachmedien. <https://doi.org/10.1007/978-3-658-32182-6_13>

    [4] Edelsbrunner, H., & Harer, J. (2010). Computational topology: An introduction.
    Chapter 7: Persistence. p. 149-156. American Mathematical Society. ISBN: 978-0-8218-4925-5

    """ 
    # Go through each value in our data, starting from highest and ending
    # with lowest. This is important because we will be merging points into
    # groups as we go, and so starting with the highest values means that
    # points will almost never have to be overwritten.
    # This is a bit complex here because we could have any number of dimensions, but
    # the gist is that we sort a flattened version of the array from highest to lowest,
    # then turn those 1d indices into Nd indices, then pair the indices together.
    # Each element of this array will be a set of indices that can index a single
    # element of data. 
    # eg. data[sortedIndices[0]] will be the largest value (for any number of dimensions)
    sortedIndices = np.dstack(np.unravel_index(np.argsort(data.flatten())[::-1], data.shape))[0]
    # To be able to actually index data with an element, they need to all be tuples
    sortedIndices = [tuple(si) for si in sortedIndices]
  
    # Numba gets mad if we pass lists between methods, so we have to turn our list
    # into a numba (typed) list
    typedSortedIndices = numba.typed.List(sortedIndices)

    peakMembership, peakBirthIndices = _findPeaks2DIter(data, typedSortedIndices, periodic)

    # Calculate the prevalence of each peak as the height of the birth points minus the
    # height of the death point
    # We could do this using the arrays we've stored along the way, but it's easier just to take the
    # max/min heights directly from the data
    peakPrevalences = np.array([data[peakBirthIndices[i]] - np.min(data[peakMembership == i]) for i in range(len(peakBirthIndices))])
    # Also note that I have made the decision here to normalize these prevalences by the total range
    # of the data, such that a prevalence of .6 means that the peak spans 60% of the range of the data.
    # This can be altered with the normalizePrevalence kwarg
    if normalizePrevalence:
        dataRange = np.max(data) - np.min(data)
        peakPrevalences /= dataRange

    # Cut off small peaks, if necessary
    if minPeakPrevalence is not None:
        peakBirthIndices = np.array([peakBirthIndices[i] for i in range(len(peakBirthIndices)) if peakPrevalences[i] > minPeakPrevalence])
        peakPrevalences = np.array([peakPrevalences[i] for i in range(len(peakPrevalences)) if peakPrevalences[i] > minPeakPrevalence])

    # Sort the peaks by their prevalence
    #order = np.argsort(peakPrevalences)[::-1]
    #peakPositions = peakPositions[order]
    #peakPrevalences = peakPrevalences[order]

    return (peakBirthIndices, peakPrevalences)


@numba.njit(cache=True)
def _findPeaks2DIter(data, sortedIndices, periodic=False):
    """
    Iterative part of 2D peak finding, optimized using `numba`.

    As usual with these types of methods, there may be some statements
    that generally would be written better/easier another way, but end up
    looking weird because of a particular `numba` requirement.

    Not meant to be used outside of `pepe.topology.findPeaks2D()`.
    """
    # This array contains the indices of the peak that each point
    # belongs to (assuming it does belong to a peak)
    # We start it at -1 such taht we can check if a point belongs to
    # a certain peak with peakMembership[i,j,...] >= 0
    peakMembership = np.zeros_like(data, dtype=np.int16) - 1
    peakBirthIndices = []

    # I've avoided using the index i here since we are iterating over sets of indices,
    # not just a single number
    for si in sortedIndices:
        # See if any neighbors have been assigned to a peak
        # No options for expanding which points are considered neighbors here
        # since the 8 surrounding points should be fine.
        assignedNeighbors = [(si[0]+i, si[1]+j) for i in [0, 1, -1] for j in [0, 1, -1]][1:]

        # Remove points outside the domain or wrap them, depending on periodicity option
        if periodic:
            # Wrap points around
            assignedNeighbors = [(n[0] % data.shape[0], n[1] % data.shape[1]) for n in assignedNeighbors]
        else:
            # Check if in bounds
            assignedNeighbors = [n for n in assignedNeighbors if n[0] >= 0 and n[1] >= 0 and n[0] < data.shape[0] and n[1] < data.shape[1]]

        # Remove points that haven't been assigned yet
        assignedNeighbors = [n for n in assignedNeighbors if peakMembership[n] >= 0]

        # If there aren't any assigned neighbors yet, then we create a new
        # peak
        if len(assignedNeighbors) == 0:
            peakMembership[si] = len(peakBirthIndices)
            peakBirthIndices.append(si)

        # If only a single one has been assigned, or all of the assigned peaks have the
        # same membership, then this point joins that peak
        elif len(assignedNeighbors) == 1 or len(np.unique(np.array([peakMembership[n] for n in assignedNeighbors]))) == 1:
            peakMembership[si] = peakMembership[assignedNeighbors[0]]

        # Otherwise, we have to resolve a conflict between multiple, in which the
        # oldest one gains the new point.
        else:      
            # Find which one is the oldest
            order = np.argsort(np.array([data[peakBirthIndices[peakMembership[n]]] for n in assignedNeighbors]))[::-1]
            # New point joins oldest peak
            peakMembership[si] = peakMembership[assignedNeighbors[order[0]]]

    return peakMembership, peakBirthIndices


#@numba.njit(cache=True)
def findPeaks1D(data, minPeakPrevalence=None, normalizePrevalence=True, periodic=False):
    """
    Find peaks in one dimensional data using persistent homology.

    Peak prevalence is calculated as the age of topological features,
    which is not necessarily the same as the actual height of the peaks.

    Originally adapted from [1], with changes mostly pertaining to readability and 
    stronger emphasis on efficiency. For background on persistence, see: [2] for
    application of persistence to signal processing, or [3] for general domain knowledge.

    May struggle to find peaks in noisier data if the fluctuations due to
    noise are large enough to create a spiky-looking profile. This is because
    such fluctuations do represent topological objects, even if they are not ones
    of any physical relevance; this method does not distinguish between the two.
    In this case, smoothing functions are recommended to be used on the data before
    it is passed to this method.

    Parameters
    ----------

    data : np.ndarray
        An array of data points within which to identify peaks.

    minPeakPrevalence : float or None
        The minimum prevalence of a peak (maximum - minimum) that will be
        returned to the user. If `normalizePrevalence` is True, this should
        be in the range `[0, 1]`, representing a percentage of the full range
        the data occupies.

    normalizePrevalence : bool
        Whether to normalize the prevalences such that they represent the
        percent of the data range that is spanned by a peak (True) or not (False).
        For example, if normalized, a prevalence of .4 would mean that the peak
        spans 40% of the total range of the data.

        Note that this normalization happens before using the minPeakPrevalence
        kwarg to clip smaller peaks, and thus the minimum value should be
        specified as a percent (eg. .3 for 30%) is the normalizePrevalence
        kwarg is True.

    periodic : bool
        Whether the array should be wrapped around to itself at the
        boundary.

    Returns
    -------

    peakPositions : np.ndarray[N]
        The indices of peaks in the provided data, sorted from most
        to least prevalent (persistent). If `returnValues` is True, this
        will instead be the values at these indices, or `data[peakPositions]`.

    peakPrevalences : np.ndarray[N]
        The prevalence of each peak in the provided data, or the persistence
        of the topological feature. If `normalizePrevalence` is True, this will
        be normalized to the domain `[0, 1]`; otherwise, these values are
        equivalent to the maximum data value minus the minimum data value evaluated
        over all points that identify with a given peak.

    References
    ----------

    [1] Huber, Stefan. Persistent Topology for Peak Detection. 
    <https://www.sthu.org/blog/13-perstopology-peakdetection/index.html>

    [2] Huber, S. (2021). Persistent Homology in Data Science. In P. Haber,
    T. Lampoltshammer, M. Mayr, & K. Plankensteiner (Eds.), Data Science – Analytics
    and Applications (pp. 81–88). Springer Fachmedien. <https://doi.org/10.1007/978-3-658-32182-6_13>

    [3] Edelsbrunner, H., & Harer, J. (2010). Computational topology: An introduction.
    Chapter 7: Persistence. p. 149-156. American Mathematical Society. ISBN: 978-0-8218-4925-5

    """

    # Entries will be a list(2) in this list, with the first element as the
    # left index, and the second element as the right index
    peakBoundsIndices = []
    peakBirthIndices = []
    # We don't actually need to record when a peak dies, since we will eventually
    # extract the persistence from the minimum value in each set of points,
    #peakDeathIndices = []

    # This array contains the indices of the peak that each point
    # belongs to (assuming it does belong to a peak)
    # We start it at -1 such taht we can check if a point belongs to
    # a certain peak with peakMembership[i] >= 0
    peakMembership = np.zeros(len(data), dtype=np.int16) - 1

    # Go through each value in our data, starting from highest and ending
    # with lowest. This is important because we will be merging points into
    # groups as we go, and so starting with the highest values means that
    # points will almost never have to be overwritten.
    sortedIndices = np.argsort(data)[::-1] # Reverse it, so it's highest -> lowest

    # Iterate over every index
    # It is important to note that we are iterating over in the order described above,
    # and so any indices calculated (for births, deaths, etc.) do not need to be first
    # passed through the sorted index array.
    # Eg. data[sortedIndices[peakMembership == 1]] would be incorrect to get all data
    # points in the peak labelled 1. Instead, data[peakMembership == 1] can be directly
    # used.
    for i in sortedIndices:
        # These bools represent whether the points to the left and right
        # (in the original data, not the height-sorted data) have been
        # assigned to a peak yet.
        leftIndex = i - 1
        rightIndex = i + 1
        if periodic:
            leftIndex = leftIndex % len(data)
            rightIndex = rightIndex % len(data)

        leftAssigned = (leftIndex >= 0 and peakMembership[leftIndex] >= 0)
        rightAssigned = (rightIndex < len(data) and peakMembership[rightIndex] >= 0)

        # If neither left or right have been processed, this point begins
        # a new peak (which may later be merged into another one)
        if not leftAssigned and not rightAssigned:
            # Assign this point to be included in a new peak
            peakMembership[i] = len(peakBoundsIndices)
            # Create a new peak, and record it's birth
            peakBoundsIndices.append([i,i])
            peakBirthIndices.append(i)
            # See note above about death
            #peakDeathIndices.append(None)

        # If either the left or right peak exists (but only one) we merge this
        # point with the existing one
        if leftAssigned and not rightAssigned:
            # Expand the peak bounds to include this new point
            # (1 index because the peak is expanding to the right)
            # We may have to wrap it around; we don't actually need to
            # check for the periodic condition, since it will only reach
            # the point of needing to be wrapped if this condition is true
            peakBoundsIndices[peakMembership[leftIndex]][1] = (peakBoundsIndices[peakMembership[leftIndex]][1] + 1) % len(data)
            peakMembership[i] = peakMembership[leftIndex]


        if rightAssigned and not leftAssigned:
            # Expand the peak bounds to include this new point
            # (0 index because the peak is expanding to the left)
            peakBoundsIndices[peakMembership[rightIndex]][0] = (peakBoundsIndices[peakMembership[rightIndex]][0] - 1) % len(data)
            peakMembership[i] = peakMembership[rightIndex]

        # If both left and right belong to peaks, that means one of them will
        # have to die, and the new point + the one who's group died get added
        # to the surviving group
        if rightAssigned and leftAssigned:
            # Determine which is taller
            if data[peakBirthIndices[peakMembership[leftIndex]]] > data[peakBirthIndices[peakMembership[rightIndex]]]:
                # If the left peak is higher, the right one dies
                # See note above about death
                #peakDeathIndices[peakMembership[rightIndex]] = i
                # Expand the bounds of the left peak to cover the right one
                # (1 index because the peak is expanding to the right)
                peakBoundsIndices[peakMembership[leftIndex]][1] = peakBoundsIndices[peakMembership[rightIndex]][1]
                # Save membership for current peak
                peakMembership[i] = peakMembership[leftIndex]
                # Save membership for right-most peak (that was just recently absorbed into this one).
                # You don't need to change every point in between, because those will never be
                # referenced again.
                peakMembership[peakBoundsIndices[peakMembership[leftIndex]][1]] = peakMembership[leftIndex]
            else:
                # If the right peak is higher, the left one dies
                # See note above about death
                #peakDeathIndices[peakMembership[i-1]] = i
                # Expand the bounds of the right peak to cover the left one
                # (0 index because the peak is expanding to the left)
                peakBoundsIndices[peakMembership[rightIndex]][0] = peakBoundsIndices[peakMembership[leftIndex]][0]
                # Save membership for current point
                peakMembership[i] = peakMembership[rightIndex]
                # Save membership for left-most point (that was just recently absorbed into this one).
                # You don't need to change every point in between, because those will never be
                # referenced again.
                peakMembership[peakBoundsIndices[peakMembership[rightIndex]][0]] = peakMembership[rightIndex]

    # This is the position of the proper crest for each peak
    peakPositions = np.array(peakBirthIndices)

    # Calculate the prevalence of each peak as the height of the birth points minus the
    # height of the death point
    # We could do this using the arrays we've stored along the way, but it's easier just to take the
    # max/min heights directly from the data

    # By this definition, you can't possibly have a peak that consists of only a
    # single point, but we might have gotten some of these from process above.
    # Honestly not really sure why they show up, but we have to check just in case.
    goodPeaks = [peakBoundsIndices[i][1] != peakBoundsIndices[i][0] for i in range(len(peakBoundsIndices))]
    
    peakPrevalences = np.array([data[peakPositions[i]] - np.min(data[peakMembership == i]) for i in range(len(peakPositions)) if goodPeaks[i]])
    peakPositions = peakPositions[goodPeaks]

    # Also note that I have made the decision here to normalize these prevalences by the total range
    # of the data, such that a prevalence of .6 means that the peak spans 60% of the range of the data.
    # This can be altered with the normalizePrevalence kwarg
    if normalizePrevalence:
        dataRange = np.max(data) - np.min(data)
        peakPrevalences /= dataRange

    # Cut off small peaks, if necessary
    if minPeakPrevalence is not None:
        peakPositions = peakPositions[peakPrevalences > minPeakPrevalence]
        peakPrevalences = peakPrevalences[peakPrevalences > minPeakPrevalence]

    # Sort the peaks by their prevalence
    order = np.argsort(peakPrevalences)[::-1]
    peakPositions = peakPositions[order]
    peakPrevalences = peakPrevalences[order]

    return (peakPositions, peakPrevalences)

