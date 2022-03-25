import numpy as np
from sklearn.neighbors import KDTree


def preserveOrderSort(oldValues, newValues, padMissingValues=False, maxDistance=None, periodic=False, fillNanSpots=True):
    """
    Order a set of values as similarly to a previous set of values as possible.
    That is, arange `newValues` such that the first element is the one that is
    closest to the first element of `oldValues`, and so on.

    If `padMissingValues=True`, then when `newValues` contains fewer items than
    `oldValues`, elements of `None` will be inserted to preserve the order.

    If `newValues` contains more items than `oldValues`, the novel elements will
    be appended at the end of the array in an arbitrary order.

    If the values are coordinates of the form `newValues=[[x1, y1, z1, ...], [x2, y2, z2, ...], ...]`
    then the proximity will be calculated as simple euclidiean distance.

    Parameters
    ----------

    oldValues : iterable
        A set of values whose order is to be preserved.

    newValues : iterable
        The new values that will be ordered based on proximity to the elements
        of `oldValues`.

    padMissingValues : bool
        Whether or not to leave placeholder values (`None`) in the final result
        for values that do not correspond to any of the original points (True)
        or to only return values that have pairs (False).

    maxDistance : float or None
        The maximum distance a point can be from its old counterpart and still be
        considered the same point. If set to None, every point in the smaller of the
        two arrays (new or old) will always have a partner; if a max distance is set,
        this is no longer guaranteed.

        Not recommended to be used if `padMissingValues=False`, since it will result
        in an array that has an arbitrary sorting order.
        
        eg. `[1, 2, 3]` sorted according to `[1, 3, 5]` would become `[1, None, 3, 2]` but
        then the `None` will be removed, resulting in `[1, 3, 2]` which likely isn't
        useful.

    periodic : bool
        Whether the values to be sorted are periodic such that pi and -pi are actually
        the same point (and should have a separation of 0, not 2pi.

    fillNanSpots : bool
        Whether to fill Nan spots in the old list with unmatched new values (True), or
        to preserve the position of the nan value and append unmatched new values to
        the end (False).

    Returns
    -------

    orderedValues : iterable
        The elements of `newValues` ordered according to proximity to the elements
        of `oldValues`.
    """

    # We already have preserveOrderArgsort written, so we can just wrap that
    order = preserveOrderArgsort(oldValues, newValues, padMissingValues, maxDistance, periodic, fillNanSpots)

    dim = len(newValues[0])

    if dim == 1:
        return np.array([newValues[i] if i is not None else np.nan for i in order])
    else:
        return np.array([newValues[i] if i is not None else np.repeat(np.nan, dim) for i in order])


def preserveOrderArgsort(oldValues, newValues, padMissingValues=False, maxDistance=None, periodic=False, fillNanSpots=True):
    """
    Return a set of indices that order a set of values as similarly to a previous
    set of values as possible. That is, a set of indices that arranges `newValues`
    such that the first element is the one that is closest to the first element of
    `oldValues`, and so on.

    If `padMissingValues=True`, then when `newValues` contains fewer items than
    `oldValues`, elements of `None` will be inserted to preserve the order.

    If `newValues` contains more items than `oldValues`, the novel elements will
    be appended at the end of the array in an arbitrary order.

    If the values are coordinates of the form `newValues=[[x1, y1, z1, ...], [x2, y2, z2, ...], ...]`
    then the proximity will be calculated as simple euclidiean distance.

    Parameters
    ----------

    oldValues : iterable
        A set of values whose order is to be preserved.

    newValues : iterable
        The new values that will be ordered based on proximity to the elements
        of `oldValues`.

    padMissingValues : bool
        Whether or not to leave placeholder values (`None`) in the final result
        for values that do not correspond to any of the original points (True)
        or to only return values that have pairs (False).

    maxDistance : float or None
        The maximum distance a point can be from its old counterpart and still be
        considered the same point. If set to None, every point in the smaller of the
        two arrays (new or old) will always have a partner; if a max distance is set,
        this is no longer guaranteed.

        Not recommended to be used if `padMissingValues=False`, since it will result
        in an array that has an arbitrary sorting order.
        
        eg. `[1, 2, 3]` sorted according to `[1, 3, 5]` would become `[1, None, 3, 2]` but
        then the `None` will be removed, resulting in `[1, 3, 2]` which likely isn't
        useful.

    periodic : bool
        Whether the values to be sorted are periodic such that pi and -pi are actually
        the same point (and should have a separation of 0, not 2pi.

    fillNanSpots : bool
        Whether to fill Nan spots in the old list with unmatched new values (True), or
        to preserve the position of the nan value and append unmatched new values to
        the end (False).

    Returns
    -------

    orderedIndices : iterable
        Indices corresponding to elements of `newValues` according to proximity
        to the elements of `oldValues`.
    """

    if maxDistance is None:
        maxDistance = np.inf

    # Make sure all of the values are np arrays
    npOldValues = np.array(oldValues)
    npNewValues = np.array(newValues)

    # This is required for the kdtree to work, since it is
    # generalized for higher dimensional spaces
    scalarValues = npOldValues.ndim == 1
    if scalarValues:
        npNewValues = npNewValues[:,None]
        npOldValues = npOldValues[:,None]

    # Convert all potential None values to np.nan (since they play nicer)
    # Numpy can only hold values of None in an array if the dtype is 'object',
    # which is inconvient for working with the kd tree
    npNewValues = np.array(np.where(npNewValues, npNewValues, np.nan), dtype=np.float64)
    npOldValues = np.array(np.where(npOldValues, npOldValues, np.nan), dtype=np.float64)

    # We want to have the longer list of items be the source points
    # for the kdtree, and the shorter be the query points. If they are
    # the same length, it doesn't matter.

    # Detect which values are valid aka not np.nan (see above about nan/None)
    oldValidIndices = np.unique(np.where(np.isnan(npOldValues) == False)[0])
    newValidIndices = np.unique(np.where(np.isnan(npNewValues) == False)[0])

    # If we have an empty old list, we just the original indexing
    if len(oldValidIndices) == 0:
        # If there just aren't any values in the old list
        # then we just return indexing for the new values
        if len(npOldValues) == 0:
            return [i for i in range(len(npNewValues))]
        else:
            # If there are some old values, but they are just all
            # None/nan, then we have to decide what to do based on
            # whether we want to fill in the nan spots or not
            if fillNanSpots:
                # Filling nan spots, we just return the same indexing for the new
                # list, potentially with some None values at the end
                return [i for i in range(len(npNewValues))] + [None]*(len(npOldValues)-len(npNewValues))
            else:
                # Otherwise, we return a similar thing, but with the None values at the
                # beginning, in the same spot as they were for the old list
                return [None]*len(npOldValues) + [i for i in range(len(npNewValues))]

    # If there are no new values, we return empty list (the new values)
    # or a list of None values
    if len(newValidIndices) == 0:
        if padMissingValues:
            return [None]*len(npOldValues)
        else:
            return [None]*len(npNewValues)

    # The list that we will be building
    # We'll append new entries on at the end
    addedIndices = [None for i in range(len(npOldValues))]

    kdTree = KDTree(npNewValues[newValidIndices], leaf_size=5)
    # This gives us the closest k points to each of the old values
    dist, ind = kdTree.query(npOldValues[oldValidIndices], k=len(newValidIndices))

    # Now convert the indices return from the kd tree (which excluded np.nan values)
    # back to the original order, which includes np.nan values
    ind = [[newValidIndices[ind[i][j]] for j in range(len(ind[i]))] for i in range(len(ind))]

    if periodic:
        # Points can't be more than pi away from each other on a circle (ignoring circulation
        # direction)
        dist = np.where(dist < np.pi, dist, 2*np.pi - dist)

    for i in range(len(oldValidIndices)):
        possiblePoints = [ind[i][j] for j in range(len(ind[i])) if (not ind[i][j] in addedIndices and dist[i][j] < maxDistance)]
        if len(possiblePoints) > 0:
            addedIndices[i] = possiblePoints[0]
        else:
            # Otherwise we don't need to do anything, because the extra values will be added
            # at the end anyway (unlike the previous case above)
            pass

    unaddedIndices = [i for i in range(len(newValidIndices)) if not i in addedIndices]

    if fillNanSpots:
        oldNanIndices = np.unique(np.where(np.isnan(npOldValues) == True)[0])
        for i in range(min(len(oldNanIndices), len(unaddedIndices))):
            addedIndices[oldNanIndices[i]] = unaddedIndices[i]

        unaddedIndices = unaddedIndices[len(oldNanIndices):] if len(oldNanIndices) < len(unaddedIndices) else []

    addedIndices += unaddedIndices
    
    if not padMissingValues:
        addedIndices = [i for i in addedIndices if i is not None]

    return addedIndices
