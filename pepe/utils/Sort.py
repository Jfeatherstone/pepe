"""
Proximity sort, in which a list of items is sorted to most closely match another list.
"""
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

    dim = len(newValues[0]) if hasattr(newValues[0], '__iter__') else 1

    # Slightly different behavior for d=1 vs. d>1, since we don't want
    # an extra dimension nested in there if our data is 1d
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
    # A note, since I have now spent a *very* long time figuring out this bug:
    # np.where(array) will give the indices of all values that are not None/np.nan
    # AND all values that are not 0. so this line used to be
    # ...np.where(npNewValues, npNewValues, np.nan)...
    # but this will convert any values that are exactly 0 to np.nan as wel, which is very bad.
    npNewValues = np.array(np.where(npNewValues != None, npNewValues, np.nan), dtype=np.float64)
    npOldValues = np.array(np.where(npOldValues != None, npOldValues, np.nan), dtype=np.float64)

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

    kdTree = KDTree(npOldValues[oldValidIndices], leaf_size=5)
    # This gives us the closest k points to each of the old values
    dist, ind = kdTree.query(npNewValues[newValidIndices], k=len(oldValidIndices))

    # Now convert the indices return from the kd tree (which excluded np.nan values)
    # back to the original order, which includes np.nan values
    ind = [[oldValidIndices[ind[i][j]] for j in range(len(ind[i]))] for i in range(len(ind))]

    if periodic:
        # Points can't be more than pi away from each other on a circle (ignoring circulation
        # direction)
        dist = np.where(dist < np.pi, dist, 2*np.pi - dist)
        # TODO: resort ind since the distances may have changed

    # Now for each new point, grab the closest old point that hasn't already
    # been added. The order we do this in is quite arbitrary (it follows the order
    # the user has passed in the values) which means that if particles move quite
    # large distances each frame, the algorithm may be incorrect. It would be more
    # apt to instead perform a minimization on the sum of the change in particle positions
    # for any possible pair configuration. TODO
    
    # When two new points have the same closest old point, one of them will have to
    # use their second closest old point instead (the farther one). This list records
    # the index their closest neighbor that they are able to use. eg. a value of 0
    # here means that that particular point used it's closest old point, a value of 1
    # means it had to use it's second closest point (because the first closest was
    # already taken).
    indexChoicePositions = [None]*len(newValidIndices)

    # This has to be a while loop with the list here because we may need
    # to recheck certain points (eg. if we first assign a point, and then realize
    # that there is actually a better match later on).
    newIndicesToAssign = list(np.arange(len(newValidIndices)))
    listIndex = 0
    while listIndex < len(newIndicesToAssign):
        # i is the index of the current new point
        i = newIndicesToAssign[listIndex]
        # The indices in possible points index the old array, so when we eventually
        # assign a pair, this index will be the **index** of the actual value. The value
        # will just be i, since that is what is iterating over the new list.

        # Checking to see that the addedIndices array is None is making sure that this new
        # point hasn't already been assigned
        # Check to make sure all of the possible points are within the max distance
        possiblePoints = [ind[i][j] for j in range(len(ind[i])) if dist[i][j] < maxDistance]
        # We don't need to explicitly check if len(possiblePoints) = 0, since the behavior
        # in that case should do nothing, as such points are dealt with at the end

        # We'll likely break out of this loop on the first iteration most of the time,
        # but we need to be able to handle if the first/second/etc. best options are already
        # taken
        for j in range(len(possiblePoints)):
            # It is possible that the closest old point to this new one has already
            # been assigned to another new point.
            if addedIndices[possiblePoints[j]] is None:
                # If it hasn't, of course that's no problem, and we can assign it
                addedIndices[possiblePoints[j]] = i
                indexChoicePositions[i] = j
                break
            else:
                # Otherwise, we have to check to see which is actually closer
                compareNewIndex = addedIndices[possiblePoints[j]]
                if dist[i][j] < dist[compareNewIndex][indexChoicePositions[compareNewIndex]]:
                    # If the new point is closer, we reassign the entry in addedIndices
                    # and change the previous best point to the next best (available) one
                    addedIndices[possiblePoints[j]] = i
                    indexChoicePositions[i] = j
                    
                    # And add the replaced index back into the list so
                    # we can find it a new match
                    indexChoicePositions[compareNewIndex] = None
                    newIndicesToAssign.append(compareNewIndex)
                    break
                else:
                    # If the new point is further, we just continue the j loop and try the
                    # next closest old point
                    continue

        listIndex += 1

    # Check which new indices haven't been assigned yet
    # Again, the values of addedIndices are the new indices, and the indices of
    # addedIndices are the old indices (yikes, am I right?)
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
