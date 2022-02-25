"""
Implementations of network analyses, like contact networks or adjacency matrices.
"""

import numpy as np

from sklearn.neighbors import KDTree 

from pepe.preprocess import circularMask
from pepe.analysis import gSquared

def adjacencyMatrix(centers, radii, contactPadding=5, neighborEvaluations=6):
    """
    Calculate the (unweighted) adjacency matrix, aka neighbor matrix,
    of a set of particles (centers and radii).

    For finding centers and radii, see `pepe.tracking`.

    Optimized for large numbers of particles by using a kd-tree to only
    evaluate potential contacts for nearby particles.

    Parameters
    ----------

    centers : np.ndarray[N,2] 
        A list of N centers of format [y, x].

    radii : np.ndarray[N]
        A list of N radii, corresponding to each particle center

    contactPadding : int
        Maximum difference between distance and sum of radii for which
        two particles will still be considered in contact.

    neightborEvaluations : int
        How many of the closest points to find via the kd tree and test
        as potential contacts. For homogeneous circles, or approximately
        homogenous (< 2:1 size ratio), 6 should be plenty.


    Returns
    -------

    adjMat : np.ndarray[N,N]
        Unweighted adjacency matrix
    """

    adjMatrix = np.zeros([len(centers), len(centers)])
    # Instead of going over every point and every other point, we can just take nearest neighbors, since
    # only neighbor particles can be in contact
    kdTree = KDTree(centers, leaf_size=10)
    
    # In 2D, 8 neighbors should be more than enough
    # +1 is so we can remove the actual point itself
    # Though if we have very few points, we may not even have 9 total
    dist, ind = kdTree.query(centers, k=min(neighborEvaluations+1, len(centers)))
   
    # See if the distance between nearby particles is less than the
    # sum of radii (+ the padding)
    for i in range(len(centers)):
        for j in range(len(ind[i])):
            if radii[i] + radii[ind[i][j]] + contactPadding > dist[i][j]: 
                adjMatrix[i][ind[i][j]] = 1
                adjMatrix[ind[i][j]][i] = 1

    return adjMatrix


def weightedAdjacencyMatrix(photoelasticSingleChannel, centers, radii, contactPadding=5, g2MaskPadding=1, contactThreshold=.1, neighborEvaluations=6):
    """
    Calculate a weighted adjacency matrix of a system of particles
    based on the average G^2 of each particle.

    For finding centers and radii, see `pepe.tracking`.

    For unweighted adjacency matrix, see `pepe.analysis.adjacencyMatrix()`.

    Parameters
    ----------

    photelasticSingleChannel : np.uint8[H,W]
        A single channel image in array form of the photoelastic response of particles.

    centers : np.ndarray[N,2] 
        A list of N centers of format [y, x].

    radii : np.ndarray[N]
        A list of N radii, corresponding to each particle center

    contactPadding : int
        Maximum difference between distance and sum of radii for which
        two particles will still be considered in contact.

    g2MaskPadding : int
        Number of pixels to ignore at the edge of each particle when calculating the average
        G^2.

    contactThreshold : float
        The neighbor weight value under which edges will be removed from the network,
        as they would be considered to weak to represent anything physical. This will
        help remove particle neighbors that are only barely touching, but not transmitting
        any real force.

    neightborEvaluations : int
        How many of the closest points to find via the kd tree and test
        as potential contacts. For homogeneous circles, or approximately
        homogenous (< 2:1 size ratio), 6 should be plenty.


    Returns
    -------

    adjMat : np.ndarray[N,N]
        Weighted adjacency matrix
    """

    # First, we calculate the unweighted network
    unweightedAdjMat = adjacencyMatrix(centers, radii, contactPadding, neighborEvaluations)

    # Calculate g^2 for entire image
    gSqr = gSquared(photoelasticSingleChannel)

    particleGSqrArr = np.zeros(len(centers))
    # Apply a circular mask to each particle, and sum the gradient squared
    # for each one
    for i in range(len(centers)):
        mask = circularMask(photoelasticSingleChannel.shape, centers[i], radii[i] - g2MaskPadding, channels=None)
        # Divide by radius squared, since we could have particles of different sizes
        particleGSqrArr[i] = np.sum(gSqr * mask) / np.sum(mask)

    # Outer product, so average gradients are all multiplied together
    weightedAdjMat = np.multiply.outer(particleGSqrArr, particleGSqrArr) * unweightedAdjMat

    # Clean up the matrix by normalizing the diagonals
    maxValue = np.max(weightedAdjMat)
    for i in range(len(centers)):
        weightedAdjMat[i,i] = maxValue

    weightedAdjMat /= np.max(weightedAdjMat)

    # And remove all edges below the cutoff threshold
    weightedAdjMat[weightedAdjMat < contactThreshold] = 0

    return weightedAdjMat
