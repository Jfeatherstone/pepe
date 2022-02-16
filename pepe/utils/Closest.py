import numpy as np
from sklearn.neighbors import KDTree


def preserveOrderSort(oldValues, newValues, padMissingValues=False):
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

    Returns
    -------

    orderedValues : iterable
        The elements of `newValues` ordered according to proximity to the elements
        of `oldValues`.
    """

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
        # The list that we will be building
        addedIndices = [None for i in range(len(npOldValues))]

        kdTree = KDTree(npOldValues, leaf_size=5)
        # This gives us the closest k points to each of the old values
        dist, ind = kdTree.query(npNewValues, k=len(npOldValues))

        for i in range(len(npNewValues)):
            possiblePoints = [ind[i][j] for j in range(len(ind[i])) if not addedIndices[ind[i][j]] is not None]
            addedIndices[possiblePoints[0]] = i

        # We build the arrays out of the original elements (not the np ones) so that we return
        # an array of the exact same shape as the original, which would not be the case for
        # a list of scalar values (since we had to stack them for the kdtree)
        if padMissingValues:
            orderedValues = [newValues[i] if i is not None else None for i in addedIndices]
        else:
            orderedValues = [newValues[i] for i in addedIndices if i is not None]

        return orderedValues

    else:
        # The list that we will be building
        # We'll append new entries on at the end
        addedIndices = [None for i in range(len(npOldValues))]

        kdTree = KDTree(npNewValues, leaf_size=5)
        # This gives us the closest k points to each of the old values
        dist, ind = kdTree.query(npOldValues, k=len(npNewValues))

        for i in range(len(npOldValues)):
            possiblePoints = [ind[i][j] for j in range(len(ind[i])) if not ind[i][j] in addedIndices]
            addedIndices[i] = possiblePoints[0]

        orderedValues = [newValues[i] for i in addedIndices]

        for i in range(len(npNewValues)):
            if not i in addedIndices:
                orderedValues.append(newValues[i])

        return orderedValues


def preserveOrderArgsort(oldValues, newValues, padMissingValues=False):
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

    Returns
    -------

    orderedIndices : iterable
        Indices corresponding to elements of `newValues` according to proximity
        to the elements of `oldValues`.
    """

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
        # The list that we will be building
        addedIndices = [None for i in range(len(npOldValues))]

        kdTree = KDTree(npOldValues, leaf_size=5)
        # This gives us the closest k points to each of the old values
        dist, ind = kdTree.query(npNewValues, k=len(npOldValues))

        for i in range(len(npNewValues)):
            possiblePoints = [ind[i][j] for j in range(len(ind[i])) if not addedIndices[ind[i][j]] is not None]
            addedIndices[possiblePoints[0]] = i

        if not padMissingValues:
            addedIndices = [i for i in addedIndices if i is not None]

        return addedIndices

    else:
        # The list that we will be building
        # We'll append new entries on at the end
        addedIndices = [None for i in range(len(npOldValues))]

        kdTree = KDTree(npNewValues, leaf_size=5)
        # This gives us the closest k points to each of the old values
        dist, ind = kdTree.query(npOldValues, k=len(npNewValues))

        for i in range(len(npOldValues)):
            possiblePoints = [ind[i][j] for j in range(len(ind[i])) if not ind[i][j] in addedIndices]
            addedIndices[i] = possiblePoints[0]

        for i in range(len(npNewValues)):
            if not i in addedIndices:
                addedIndices.append(i)

        return addedIndices
