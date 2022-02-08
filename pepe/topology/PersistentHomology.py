import numpy as np


import matplotlib.pyplot as plt

def findPeaks2D():
    pass


def findPeaks(data, returnValues=False, minPeakPrevalence=None, normalizePrevalence=True):
    """
    Find peaks in one dimensional data using persistent homology.

    Peak prevalence is calculated as the age of topological features.

    Parameters
    ----------

    data : np.ndarray
        An array of data points within which to identify peaks.

    returnValues : bool
        Whether to return the indices of maxima (False) or the values
        at those maxima (True).

    minPeakPrevalence : float or None
        The minimum prevalence of a peak (maximum - minimum) that will be
        returned to the user.

    normalizePrevalence : bool
        Whether to normalize the prevalences such that they represent the
        percent of the data range that is spanned by a peak (True) or not (False).
        For example, if normalized, a prevalence of .4 would mean that the peak
        spans 40% of the total range of the data.

        Note that this normalization happens before using the minPeakPrevalence
        kwarg to clip smaller peaks, and thus the minimum value should be
        specified as a percent (eg. .3 for 30%) is the normalizePrevalence
        kwarg is True.
    """

    # Entries will be a list(2) in this list, with the first element as the
    # left index, and the second element as the right index
    peakBoundsIndices = []
    peakBirthIndices = []
    peakDeathIndices = []

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
        leftAssigned = (i > 0 and peakMembership[i-1] >= 0)
        rightAssigned = (i < len(data)-1 and peakMembership[i+1] >= 0)

        # If neither left or right have been processed, this point begins
        # a new peak (which may later be merged into another one)
        if not leftAssigned and not rightAssigned:
            # Assign this point to be included in a new peak
            peakMembership[i] = len(peakBoundsIndices)
            # Create a new peak, and record it's birth
            peakBoundsIndices.append([i,i])
            peakBirthIndices.append(i)
            peakDeathIndices.append(None)

        # If either the left or right peak exists (but only one) we merge this
        # point with the existing one
        if leftAssigned and not rightAssigned:
            # Expand the peak bounds to include this new point
            # (1 index because the peak is expanding to the right)
            peakBoundsIndices[peakMembership[i-1]][1] += 1
            peakMembership[i] = peakMembership[i-1]


        if rightAssigned and not leftAssigned:
            # Expand the peak bounds to include this new point
            # (0 index because the peak is expanding to the left)
            peakBoundsIndices[peakMembership[i+1]][0] -= 1
            peakMembership[i] = peakMembership[i+1]

        # If both left and right belong to peaks, that means one of them will
        # have to die, and the new point + the one who's group died get added
        # to the surviving group
        if rightAssigned and leftAssigned:
            # Determine which is taller
            if data[peakBirthIndices[peakMembership[i-1]]] > data[peakBirthIndices[peakMembership[i+1]]]:
                # If the left peak is higher, the right one dies
                peakDeathIndices[peakMembership[i+1]] = i
                # Expand the bounds of the left peak to cover the right one
                # (1 index because the peak is expanding to the right)
                peakBoundsIndices[peakMembership[i-1]][1] = peakBoundsIndices[peakMembership[i+1]][1]
                # Save membership for current peak
                peakMembership[i] = peakMembership[i-1]
                # Save membership for right-most peak (that was just recently absorbed into this one).
                # You don't need to change every point in between, because those will never be
                # referenced again.
                peakMembership[peakBoundsIndices[peakMembership[i-1]][1]] = peakMembership[i-1]
            else:
                # If the right peak is higher, the left one dies
                peakDeathIndices[peakMembership[i-1]] = i
                # Expand the bounds of the right peak to cover the left one
                # (0 index because the peak is expanding to the left)
                peakBoundsIndices[peakMembership[i+1]][0] = peakBoundsIndices[peakMembership[i-1]][0]
                # Save membership for current point
                peakMembership[i] = peakMembership[i+1]
                # Save membership for left-most point (that was just recently absorbed into this one).
                # You don't need to change every point in between, because those will never be
                # referenced again.
                peakMembership[peakBoundsIndices[peakMembership[i+1]][0]] = peakMembership[i+1]

    # This is the position of the proper crest for each peak
    peakPositions = np.array(peakBirthIndices)

    # Calculate the prevalence of each peak as the height of the birth points minus the
    # height of the death point
    # We could do this using the arrays we've stored along the way, but it's easier just to take the
    # max/min heights directly from the data
    peakPrevalences = np.array([data[peakPositions[i]] - np.min(data[peakMembership == i]) for i in range(len(peakBoundsIndices))])

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

    if returnValues:
        return data[peakPositions], peakPrevalences
    else:
        return peakPositions, peakPrevalences

