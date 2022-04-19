"""
Method to solve D2min field between two different particle states,
and calculate the von Mises strain.

Copied directly from my standalone repo:
https://github.com/Jfeatherstone/D2min

Originally forked from:
https://github.com/Binxu-Stack/D2min
created by Bin Xu.

Modifications include more documentation, change of notation,
and change from bond-focused calculation to center-focused
calculation (see code for more details).

Intended for use within other projects, so CLI access has been
removed.
"""

import numpy as np
from numpy.linalg import inv

# For calculating which neighbors are within a certain distance
from sklearn.neighbors import KDTree

def calculateD2Min(initialCenters, finalCenters, refParticleIndex=0, interactionRadius=None, interactionNeighbors=None, normalize=True):
    """
    Calculate the d2min between an initial and final state of particles;
    a measure of how non-affine the transformation is, as originally described
    in [1].

    
    Parameters
    ----------

    initialCenters : np.ndarray[N,d] or list
        List of initial center points of N particles in d dimensions.

    finalCenters :  np.ndarray[N,d] or list
        List of final center points of N particles in d dimensions. Must be
        in the same order as initialCenters.

    refParticleIndex : int
        The index of the particle to treat as the reference particle (r_0 in Falk & Langer
        1998 eq. 2.11). If set to None, will calculate the D2min N times using each particle
        as the reference (and return types will have an extra first dimension N). Simiarly, can
        be a list of indices for which to calculate as the refernce indices.

    interactionRadius : float
        The maximum distance between particles that can be considered neighbors. Recommended to
        be set to around 1.1 - 1.5 times the mean particle radius of the system.

        If set to None, all other particles in the system will be considered neighbors. See interactionNeighbors
        for specifying a fixed number of neighbors. In the case that neither a radius or number of
        neighbors are specified, calculation will default to using all other particles as neighbors.

    interactionNeighbors : int
        As opposed to using an interactionRadius to define neighbors, a fixed number of neighbors can
        be specified here. This number of neighbors will be found using a kd-tree for the reference point(s).

        In the case that neither a radius or number of neighbors are specified, calculation will default
        to using all other particles as neighbors.

    normalize : bool
        Whether to divide the d2min by the number of neighbors used to calculate it (True) or not (False).
        For heterogeneous systems where the number of neighbors can vary significantly, recommend to set True.
        Will make little difference if a fixed number of neighbors (see interactionNeigbors) are used.

    Returns
    -------

    d2min : float
        The minimum value of D2 for the transition from the initial to final state. Units
        are a squared distance dimensionally the same as the initial and final centers (likely
        a pixel^2 value if tracked from images). Changing units to particle diameters afterwards
        may be necessary.

    epsilon : numpy.ndarray[d,d]
        The uniform strain tensor that minimizes D2; equation 2.14 in [1].


    In the case that `refParticleIndex=None`, the return will instead be a tuple of numpy arrays
    containing the same information but for every particle:

    d2minArr : np.ndarray[N]
        The minimum value of D2 for the transition from the initial to final state for
        every possible configuration.

    epsilonArr : np.ndarray[N,d,d]
        The uniform strain tensor that minimizes D2 for every possible configuration


    Examples
    --------

    See `test` folder in [standalone repository](https://github.com/Jfeatherstone/D2min).

 
    References
    ----------    

    [1] Falk, M. L., & Langer, J. S. (1998). Dynamics of viscoplastic deformation in amorphous solids. Physical Review E, 57(6), 7192–7205.
    [https://doi.org/10.1103/PhysRevE.57.7192](https://doi.org/10.1103/PhysRevE.57.7192)

    """

    # The number of particles and spatial dimension
    N, d = np.shape(initialCenters)

    # In the case that a single reference particle is defined, we just calculate exactly as described in the paper
    if not isinstance(refParticleIndex, list) and not isinstance(refParticleIndex, np.ndarray) and refParticleIndex != None:

        # Determine which particles are neighbors using the parameters supplied on the function call
        initialNeighbors = None
        finalNeighbors = None

        # If an interaction radius is supplied, we have to find the particles that are closest
        # in the initial state using a kd-tree
        if interactionRadius != None:
            kdTree = KDTree(initialCenters)
            ind = kdTree.query_radius([initialCenters[refParticleIndex]], interactionRadius)
            
            # We ignore the first element since it will always be the particle itself
            # And we have to sort we don't mess up the order
            ind = np.sort(ind[0][1:])

            # Make sure we actually found some particles
            if len(ind) == 0:
                print('Warning: no neighbors found within supplied interaction radius! Defaulting to all other particles')
            else:
                initialNeighbors = initialCenters[ind]
                finalNeighbors = finalCenters[ind]    

        # If a fixed number of neighbors is provided instead, we find those particles again with
        # a kd-tree
        elif interactionNeighbors != None:
            kdTree = KDTree(initialCenters)
            # Make sure we don't look for more particles than are in our system
            dist, ind = kdTree.query([initialCenters[refParticleIndex]], min(interactionNeighbors+1, N))
            
            # We ignore the first element since it will always be the particle itself
            # And we have to sort we don't mess up the order
            ind = np.sort(ind[0][1:])
            
            initialNeighbors = initialCenters[ind]
            finalNeighbors = finalCenters[ind]    

        # If no information is supplied, or we ran into issues, use every other particle
        if not isinstance(initialNeighbors, list) and not isinstance(initialNeighbors, np.ndarray):
            initialNeighbors = initialCenters[np.arange(N) != refParticleIndex]
            finalNeighbors = finalCenters[np.arange(N) != refParticleIndex]

        # Now onto the actual D2min calculation
        # Bin's original code defined the differences between centers in
        # Falk & Langer eq. 2.11 - 2.13 as "bonds"

        # We first calculate these bonds using our reference particle index
        # Do this by subtracting the ref particle center from the center of every other particle
        # Note that you could technically leave in the ref bond, since it will be 0 and not contribute,
        # but it is cleaner to just remove it

        initialBonds = initialNeighbors - initialCenters[refParticleIndex]
        finalBonds = finalNeighbors - finalCenters[refParticleIndex]

        # More succinct notation for doing the calculation, from Bin's original code
        # Converting to numpy matrices makes matrix multiplication happen automatically
        b0 = np.mat(initialBonds)
        b = np.mat(finalBonds)

        # Calculate the two functions used to minimize D2, X and Y (eq. 2.12 and 2.13 respectively)
        X = b0.transpose() * b
        Y = b0.transpose() * b0

        # Calculate the uniform strain tensor that minimizes D2 (eq. 2.14)
        # Note that we don't include the kronecker delta function since it will
        # be cancelled out when plugged into the D2min equation (eq. 2.11).
        # Also not that this is actually the transpose of the strain tensor as
        # it is defined in the paper, since it makes the matrix multiplication easier
        # in the next step
        epsilon = inv(Y) * X

        # Non-affine part, or the terms that are squared and summed over in eq. 2.11
        non_affine = b - b0*epsilon

        # The final value
        d2min = np.sum(np.square(non_affine))

        if normalize:
            d2min /= len(initialNeighbors)

        # Since we have been working with the transpose of the strain tensor,
        # we have to transpose to get the proper one
        return (d2min, np.array(epsilon.transpose()))

    # If we don't have a reference particle, or we are given multiple, we calculate for each of those
    if not isinstance(refParticleIndex, list) and not isinstance(refParticleIndex, np.ndarray):
        refParticleIndex = np.arange(N)

    # Now calculate for all of those possibilities
    d2minArr = np.zeros(len(refParticleIndex))
    epsilonArr = np.zeros([len(refParticleIndex), d, d])
    
    for i in range(len(refParticleIndex)):

        d2min, epsilon = calculateD2Min(initialCenters, finalCenters, refParticleIndex[i],
                                       interactionRadius, interactionNeighbors, normalize)
        d2minArr[i] = d2min
        epsilonArr[i] = epsilon

    return (d2minArr, epsilonArr)

        
def vonMisesStrain(uniformStrainTensor):
    """
    WIP.

    """
    # The number of spatial dimensions
    dimension = np.shape(uniformStrainTensor)[0]

    # Lagrangian strain matrix
    eta = 0.5 * (uniformStrainTensor * uniformStrainTensor.transpose() - np.eye(dimension))

    # von-Mises strain
    eta_m = 1.0/np.double(dimension) * np.trace(eta)
    tmp = eta - eta_m * np.eye(dimension)
    eta_s = np.sqrt(0.5*np.trace(tmp*tmp))

    return eta_s
