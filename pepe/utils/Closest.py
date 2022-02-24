import numpy as np
from sklearn.neighbors import KDTree


def preserveOrderSort(oldValues, newValues, padMissingValues=False, maxDistance=None, periodic=False):
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

    Returns
    -------

    orderedValues : iterable
        The elements of `newValues` ordered according to proximity to the elements
        of `oldValues`.
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

    # We want to have the longer list of items be the source points
    # for the kdtree, and the shorter be the query points. If they are
    # the same length, it doesn't matter.

    if len(npOldValues) > len(npNewValues):
        # If we have an empty old list, we just the (original) new list
        if len(npOldValues) == 0:
            return newValues

        # The list that we will be building
        addedIndices = [None for i in range(len(npOldValues))]

        kdTree = KDTree(npOldValues, leaf_size=5)
        # This gives us the closest k points to each of the old values
        dist, ind = kdTree.query(npNewValues, k=len(npOldValues))

        if periodic:
            # Points can't be more than pi away from each other on a circle (ignoring circulation
            # direction)
            dist = np.where(dist < np.pi, dist, 2*np.pi - dist)

        for i in range(len(npNewValues)):
            possiblePoints = [ind[i][j] for j in range(len(ind[i])) if (not addedIndices[ind[i][j]] is not None and dist[i][j] < maxDistance)]
            if len(possiblePoints) > 0:
                addedIndices[possiblePoints[0]] = i
            else:
                # If we didn't find any possible pairs for this point, it must have been too far from
                # any other point to correspond properly, so we add it at the end, denoting that this
                # represents a new point
                addedIndices.append(i)

        # We build the arrays out of the original elements (not the np ones) so that we return
        # an array of the exact same shape as the original, which would not be the case for
        # a list of scalar values (since we had to stack them for the kdtree)
        if padMissingValues:
            orderedValues = [newValues[i] if i is not None else None for i in addedIndices]
        else:
            orderedValues = [newValues[i] for i in addedIndices if i is not None]

        return orderedValues

    else:

        # If there are no new values, we return empty list (the new values)
        # or a list of None values
        if len(npNewValues) == 0:
            if padMissingValues:
                return [None for _ in range(len(npOldValues))]
            else:
                return newValues

        # The list that we will be building
        # We'll append new entries on at the end
        addedIndices = [None for i in range(len(npOldValues))]

        kdTree = KDTree(npNewValues, leaf_size=5)
        # This gives us the closest k points to each of the old values
        dist, ind = kdTree.query(npOldValues, k=len(npNewValues))

        if periodic:
            # Points can't be more than pi away from each other on a circle (ignoring circulation
            # direction)
            dist = np.where(dist < np.pi, dist, 2*np.pi - dist)

        for i in range(len(npOldValues)):
            possiblePoints = [ind[i][j] for j in range(len(ind[i])) if (not ind[i][j] in addedIndices and dist[i][j] < maxDistance)]
            if len(possiblePoints) > 0:
                addedIndices[i] = possiblePoints[0]
            else:
                # Otherwise we don't need to do anything, because the extra values will be added
                # at the end anyway (unlike the previous case above)
                pass

        orderedValues = [newValues[i] if i is not None else None for i in addedIndices]

        for i in range(len(npNewValues)):
            if not i in addedIndices:
                orderedValues.append(newValues[i])

        return orderedValues


def preserveOrderArgsort(oldValues, newValues, padMissingValues=False, maxDistance=None, periodic=False):
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
    # TODO: I never finished making this method compatable with nan/None values,
    # so that needs to be finished (or would be nice to have finished)
    #npNewValues = np.array(np.where(npNewValues, npNewValues, np.nan), dtype=np.float64)
    #npOldValues = np.array(np.where(npOldValues, npOldValues, np.nan), dtype=np.float64)

    # We want to have the longer list of items be the source points
    # for the kdtree, and the shorter be the query points. If they are
    # the same length, it doesn't matter.

    # If we have an empty old list, we just the original indexing
    if len(npOldValues) == 0:
        return [i for i in range(len(npNewValues))]

    if len(npOldValues) > len(npNewValues):

        # The list that we will be building
        addedIndices = [None for i in range(len(npOldValues))]

        # Detect if any values are None/nan (see above about nan/None)
        #noneIndices = np.unique(np.where(np.isnan(npOldValues))[0])

        kdTree = KDTree(npOldValues, leaf_size=5)
        # This gives us the closest k points to each of the old values
        dist, ind = kdTree.query(npNewValues, k=len(npOldValues))

        if periodic:
            # Points can't be more than pi away from each other on a circle (ignoring circulation
            # direction)
            dist = np.where(dist < np.pi, dist, 2*np.pi - dist)

        for i in range(len(npNewValues)):
            possiblePoints = [ind[i][j] for j in range(len(ind[i])) if (not addedIndices[ind[i][j]] is not None and dist[i][j] < maxDistance)]
            if len(possiblePoints) > 0:
                addedIndices[possiblePoints[0]] = i
            else:
                # If we didn't find any possible pairs for this point, it must have been too far from
                # any other point to correspond properly, so we add it at the end, denoting that this
                # represents a new point
                addedIndices.append(i)

        if not padMissingValues:
            addedIndices = [i for i in addedIndices if i is not None]

        return addedIndices

    else:
        # If there are no new values, we return empty list (the new values)
        # or a list of None values
        if len(npNewValues) == 0:
            if padMissingValues:
                return [None for _ in range(len(npOldValues))]
            else:
                return newValues

        # The list that we will be building
        # We'll append new entries on at the end
        addedIndices = [None for i in range(len(npOldValues))]

        kdTree = KDTree(npNewValues, leaf_size=5)
        # This gives us the closest k points to each of the old values
        dist, ind = kdTree.query(npOldValues, k=len(npNewValues))

        if periodic:
            # Points can't be more than pi away from each other on a circle (ignoring circulation
            # direction)
            dist = np.where(dist < np.pi, dist, 2*np.pi - dist)

        for i in range(len(npOldValues)):
            possiblePoints = [ind[i][j] for j in range(len(ind[i])) if (not ind[i][j] in addedIndices and dist[i][j] < maxDistance)]
            if len(possiblePoints) > 0:
                addedIndices[i] = possiblePoints[0]
            else:
                # Otherwise we don't need to do anything, because the extra values will be added
                # at the end anyway (unlike the previous case above)
                pass

        for i in range(len(npNewValues)):
            if not i in addedIndices:
                addedIndices.append(i)

        return addedIndices
