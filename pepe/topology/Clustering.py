"""
These methods aren't really 'topological', but I think this is
still the most apt place to put this code.
"""

import numpy as np
from scipy.spatial import KDTree

import matplotlib.pyplot as plt

def spatialClusterLabels(points, l=.001, randomize=False, wrapPoints=None):
    """
    Partition a set of points in clusters, and return the
    cluster label for each point.

    Method is very simplistic: the neighbors (points within
    a threshold distance) of a point are added to the same
    cluster, and the process is repeated until no points
    remain. This means that the size of clusters can be
    larger than this distance threshold (eg. think of a
    chain of points).

    Usage
    -----
    The partitioning process is done by iterating through all
    of the provided points, and identifying their neighbors. As
    such, this process can vary based on the order the points are
    provided in. There are two different ways to deal with this
    while still maintaining reproducible results.

    First, you can order the points in some meaningful way, with
    the more important (whatever that means for your case) points
    coming first. This should be done before calling this partitioning
    method. Second, you can use the `randomize=True` kwarg, and rely
    on the fact that this should lead to a statistically sound
    calculation.

    As an example, consider identifying clusters of points above some
    threshold (whether in brightness, \(G^2\), or whatever else) in
    an image. For the first approach, you might order the points by the
    magnitude of the parameter you are thresholding with respect to, since
    eg. it could be assumed that very bright points are more important
    than points that are just barely above the threshold. For the second
    approach, you might perform the clustering process M times to get
    a statistical ensemble, and then resolve the individual partitionings
    into a single combined one.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Array of N points in d-dimensions

    l : float
        Distance threshold to consider two points as being in the
        same cluster. Given as fraction of the total system size.

    randomize : bool
        The clustering processes is dependent on the order of
        the points; if `randomize=True`, the provided array will
        be indexed in a random order, otherwise it will be accessed
        exactly as provided.

    wrapPoints : numpy.ndarray[d] or float or None
        If the space is periodic, the size of each dimension. If
        a single value is given, this will be used for all dimensions.

        If None, no periodicity is assumed.

        Can have a mix of periodic and non-periodic dimensions,
        eg. spherical coordinates should have:
            `wrapPoints=[None, 2*np.pi, np.pi]`
        
    Returns
    -------
    labels : numpy.ndarray[N]
        Array of N labels (integers) denoting cluster
        assignment. Total number of clusters will be `max(labels)+1`.
    """
    d = np.shape(points)[-1]
    
    systemLengthScale = np.sqrt(np.sum([(np.max(points[:,i]) - np.min(points[:,i]))**2 for i in range(d)]))
    threshold = l*systemLengthScale
   
    # Generate a kd-tree
    # If we have a periodic space, we need to clean and pass that
    # information to the kd tree
    if hasattr(wrapPoints, '__iter__'):
        assert len(wrapPoints) == d, f'Wrap dimensions ({len(wrapPoints)}) do not match the dimension of the data ({d})!'

        boxSize = np.array([wrapPoints[i] if wrapPoints[i] is not None else np.max(points[:,i])*10 for i in range(d)])
        kdTree = KDTree(points, boxsize=boxSize)

    elif wrapPoints is not None:
        boxSize = np.repeat(wrapPoints, d)
        kdTree = KDTree(points, boxsize=boxSize)
    
    else:
        kdTree = KDTree(points)

    pointsList = points.tolist()

    randomOrder = np.arange(len(pointsList))

    if randomize:
        np.random.shuffle(randomOrder)
    
    labels = np.zeros(len(pointsList)) - 1
    labelsToMerge = []
    
    i = 0
    
    while len(np.where(labels == -1)[0]) > 0:
        # Grab a point, if it doesn't have a group already, create a new one
        if labels[randomOrder[i]] == -1:
            labels[randomOrder[i]] = np.max(labels)+1
        p = pointsList[randomOrder[i]]

        # Find its neighbors
        neighbors = kdTree.query_ball_point(p, r=threshold)

        # neighorLabels = [labels[n] for n in neighors]
        # uniqueNeighborLabels = np.unique(neighborLabels)
        # if len(uniqueNeighborLabels) == 1:
        #     # If all are unlabeled
        #     if uniqueNeighborLabels[0] == -1:
        #         # Create a new group
        #         labels[randomOrder[i]] = np.max(labels)+1

        #     # All are already in another group
        #     else:
        #         # Assign this point to that group as well
        #         labels[randomOrder[i]] = uniqueNeighborLabels[0]
        
        # Assign all of them to be the same group
        for index in neighbors:
            if index == randomOrder[i]:
                continue
                
            # If they already have a label, we will eventually want to merge the labels
            if labels[index] >= 0 and labels[index] != labels[randomOrder[i]]:
                # See if this is already in a merge group
                inGroup = False
                for j in range(len(labelsToMerge)):
                    # If it is already going to be merged, add the
                    # new label to this group
                    if labels[index] in labelsToMerge[j]:
                        inGroup = True
                        # If the new label is not in the merge group, add it
                        if not labels[randomOrder[i]] in labelsToMerge[j]:
                            labelsToMerge[j].append(labels[randomOrder[i]])
                            
                        break
                # If we aren't in a merge group, we should create
                # a new one
                if not inGroup:
                    labelsToMerge.append([labels[index], labels[randomOrder[i]]])
                    
            elif labels[index] == -1:
                labels[index] = labels[randomOrder[i]]
            
        i += 1

    mergedLabels = np.zeros_like(labels) - 1
    
    # Now we have to sort out merged clusters
    labelSets = []
    for i in range(int(np.max(labels))+1):
        # Generate a list of all of the labels that are identified together
        # including this value of i
        allLabels = [i]
        for j in range(len(labelsToMerge)):
            if i in labelsToMerge[j]:
                allLabels += labelsToMerge[j]

        # sort so that way we can do a unique check to remove
        # duplicates
        allLabels = sorted(np.unique(allLabels))
        # Python may automatically collapse the list if it is of length 1,
        # which we don't want.
        labelSets.append([allLabels] if len(allLabels) == 1 else allLabels)

    # Remove duplicate sets, creating disjoint sets of labels
    uniqueSets = np.unique(np.array(labelSets, list))

    # If there are no overlaps, then this above operation will
    # flatten the array to be one dimensional. We can't do anything
    # about that, except to check if it is the case. The good news is
    # that if this is the case, then we already have disjoint sets
    # and we can just immediately return our label array
    if len(np.shape(uniqueSets)) == 1:
        return labels

    # Relabel things with the new disjoint sets
    mergedLabels = np.zeros_like(labels)

    # TODO: Probably a much better way to do this
    for i in range(len(pointsList)):
        for j in range(len(uniqueSets)):
            if labels[i] in uniqueSets[j]:
                mergedLabels[i] = j
                break

    return mergedLabels

def spatialClusterCenters(points, l=.001, randomize=False, wrapPoints=None, pointWeights=None, return_weights=False):
    """
    Partition a set of points in clusters, and compute the
    center of each cluster.

    Generates clusters using `pepe.topology.spatialClusterLabels()`; see
    this method for more information.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Array of N points in d-dimensions

    l : float
        Distance threshold to consider two points as being in the
        same cluster. Given as fraction of the total system size.

    randomize : bool
        The clustering processes is dependent on the order of
        the points; if `randomOrder=True`, the provided array will
        be indexed in a random order, otherwise it will be accessed
        exactly as provided. 
        
        See documentation for `pepe.topology.spatialClusterLabels()`.

    wrapPoints : numpy.ndarray[d] or float or None
        If the space is periodic, the size of each dimension. If
        a single value is given, this will be used for all dimensions.

        If None, no periodicity is assumed.

        Can have a mix of periodic and non-periodic dimensions,
        eg. spherical coordinates should have:
            `wrapPoints=[None, 2*np.pi, np.pi]`

    pointWeights : numpy.ndarray[N] or None
        An array of weights to be used in finding the center of
        mass of each cluster. If `None`, every point will be
        weighted the same.

    return_weights : bool
        Whether to return the weight -- defined as the fraction of
        points included in that cluster -- alongside the centers.

        If a weight is given for each point using `pointWeights`, this
        will be used in calculating the weight of a cluster.
        
    Returns
    -------
    centers : numpy.ndarray[N, d]
        Array of N points in d-dimensions representing the detected
        clusters in the system.

    weights : numpy.ndarray[N]
        Array of weights -- defined as fraction of all points included
        in each cluster -- for each cluster. Only returned if
        `return_weights=True`.

    """
    labels = spatialClusterLabels(points, l=l, randomize=randomize, wrapPoints=wrapPoints)
    numLabels = int(np.max(labels))+1

    if hasattr(pointWeights, '__iter__'):
        individualWeights = pointWeights
    else:
        individualWeights = np.ones_like(labels)

    # Compute the center of each cluster
    weights = np.zeros(numLabels)
    centers = np.zeros((numLabels, np.shape(points)[-1]))
     
    # If our data is periodic, we can't just take the average of
    # the positions, we have to account for the possibility
    # that a cluster wraps around a boundary.
    if hasattr(wrapPoints, '__iter__') or wrapPoints is not None:
        for i in range(numLabels):
            indices = np.where(labels == i)[0]
            # We can check if we have a discontinuous jump by looking at the
            # sort changes in each dimension. If there is a jump that is larger
            # than half of the dimension size, this means that boundary needs to
            # be factored in.
            boundaryCrosses = [False]*np.shape(points)[-1]
            divideCenters = [np.nan]*np.shape(points)[-1]
            for j in range(np.shape(points)[-1]):
                oneDimPoints = np.sort(points[indices,j])
                diffArr = oneDimPoints[1:] - oneDimPoints[:-1]
                maxIndex = np.argmax(np.abs(diffArr))
                boundaryCrosses[j] = np.abs(diffArr[maxIndex]) > wrapPoints[j]/2
                # Record where the center of the gap between the two sides is
                divideCenters[j] = (oneDimPoints[maxIndex] + oneDimPoints[maxIndex+1])/2

            # Now we adjust the axes that were identified
            # Not the best naming but this array contains points that
            # are wrapped whereas wrapPoints contains the actual points
            # at which the space wraps around itself...
            wrappedPoints = np.array(points)[indices]
            for j in np.where(boundaryCrosses)[0]:
                # We have to find all of the points on one side of the wrap
                preWrapIndices = np.where(wrappedPoints[:,j] < divideCenters[j])
                # Move them to the other side of the wrap
                wrappedPoints[preWrapIndices,j] += wrapPoints[j]

            # Take the weighted average
            centers[i] = np.average(wrappedPoints, weights=individualWeights[indices], axis=0)
            weights[i] = np.sum(individualWeights[indices]) / np.sum(individualWeights)

            # Adjust in case we ended up outside on the wrong side
            # of the boundary
            # TODO
            axisNeedsAdjusting = (centers[i] / wrapPoints) > 1
            for j in np.where(axisNeedsAdjusting)[0]:
                centers[i][j] -= wrapPoints[j]

    else:
        for i in range(numLabels):
            indices = np.where(labels == i)
            # Weighted average
            centers[i] = np.average(np.array(points)[indices], weights=individualWeights[indices], axis=0)
            weights[i] = np.sum(individualWeights[indices]) / np.sum(individualWeights)

    order = np.argsort(weights)[::-1]
    centers = centers[order]
    weights = weights[order]
    
    if return_weights:
        return centers, weights
        
    return centers
