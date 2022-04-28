"""
Methods to gauge how well force balance is satisfied for an ensemble,
and to convert between polar and cartesian systems.
"""

import numpy as np

import numba

def polarToCartesian(force, alpha, beta, collapse=True):
    """
    Convert a set of forces defined in polar coordinates (f, a, b),
    to cartesian coordinates (f_y, f_x).

    Parameters
    ----------

    force : float or np.ndarray[F] or list[F]
        The force magnitude, or an array/list of F force magnitudes.

    alpha : float or np.ndarray[F] or list[F]
        The alpha angle, or an array/list of F alpha angles.

    beta : float or np.ndarray[F] or list[F]
        The beta angle, or an array/list of F beta angles.

    collapse : bool
        Whether to collapse the force index dimension in the case that
        only a single force is provided.

    Returns
    -------

    forceArr : np.ndarray[F,2]
        An array of the cartesian components (y,x) of the forces. 

        If only a single force is provided (ie. `force`, `alpha` and `beta` are all
        floats) the first dimension will be omitted, leaving just `[f_y, f_x]`. See
        `collapse` for more information.
    """
    # Check to see if we were given multiple forces, or just a single one
    if hasattr(force, '__iter__'):
        forceArr = np.array(force)
        alphaArr = np.array(alpha)
        betaArr = np.array(beta)
        singleForce = False
    else:
        forceArr = np.array([force])
        alphaArr = np.array([alpha])
        betaArr = np.array([beta])
        singleForce = True
    
    cartesianForceArr = np.zeros((forceArr.shape[0], 2))
    for i in range(cartesianForceArr.shape[0]):
        # Note that this expression is not exactly the same as in K. E. Daniels et al.
        # Rev. Sci. Inst. 88 (2017). There is an extra negative on the alphas, since mine
        # appear to be defined backwards.
        # F_y
        cartesianForceArr[i,0] = forceArr[i] * np.cos(-alphaArr[i] + betaArr[i]) #(np.cos(betaArr[i,j]) * np.cos(alphaArr[i,j]) + np.sin(betaArr[i,j]) * np.sin(alphaArr[i,j]))
        # F_x
        cartesianForceArr[i,1] = -forceArr[i] * np.sin(-alphaArr[i] + betaArr[i]) #(-np.sin(betaArr[i,j]) * np.cos(alphaArr[i,j]) + np.cos(betaArr[i,j]) * np.sin(alphaArr[i,j]))
       

    # If we only have a single force, we should collapse that first dimension
    if singleForce and collapse:
        return cartesianForceArr[0]

    return cartesianForceArr


def testForceBalance(forceArr, alphaArr, betaArr, collapse=True):
    """
    Sum each of the cartesian force components to see how
    well an ensemble of forces satisfies force balance.

    Parameters
    ----------

    forceArr : np.ndarray[F] or np.ndarray[T,F]
        An array/list of F force magnitudes, possibly for T timesteps.

    alphaArr : np.ndarray[F] or np.ndarray[T,F]
        An array/list of F alpha angles, possibly for T timesteps.

    betaArr : np.ndarray[F] or np.ndarray[T,F]
        An array/list of F beta angles, possibly for T timesteps.

    collapse : bool
        Whether to collapse the timestep dimension in the case that
        only a single timestep is provided.

    Returns
    -------

    forceSumArr : np.ndarray[T,2]
        An array of the sum of each cartesian component (y,x) of the forces at each timestep.

        If only a single timestep is provided (ie. `forceArr`, `alphaArr` and `betaArr` are all
        1D arrays) the first dimension will be omitted, leaving just `[sum_f_y, sum_f_x]`. See
        `collapse` for more information.
    """
    # Check if we were given a single timestep, or multiple
    if len(np.shape(forceArr)) == 2:
        singleTimestep = False
        multiForceArr = np.array(forceArr)
        multiAlphaArr = np.array(alphaArr)
        multiBetaArr = np.array(betaArr)
    else:
        singleTimestep = True
        # TODO: Might need a transpose here
        multiForceArr = np.array([forceArr])
        multiAlphaArr = np.array([alphaArr])
        multiBetaArr = np.array([betaArr])

    forceSumArr = np.zeros((multiForceArr.shape[1], 2)) 
    # Sum up forces for each timestep
    for i in range(multiForceArr.shape[1]):
        cartForces = polarToCartesian(multiForceArr[:,i], multiAlphaArr[:,i], multiBetaArr[:,i], collapse=False)

        # sum_y
        forceSumArr[i,0] = np.sum(cartForces[:,0])
        # sum_x
        forceSumArr[i,1] = np.sum(cartForces[:,1])

    if singleTimestep and collapse:
        return forceSumArr[0]

    return forceSumArr


@numba.jit(nopython=True)
def singleParticleForceBalance(forceArr, alphaArr, betaArr):
    """
    **Does not currently work! Any calls to this function will just return the original
    arrays**

    Takes a set of forces acting on a single particle and ensures they obey
    force balance.

    The majority of this method is transpiled directly from Jonathan Kollmer's
    implementation:
    https://github.com/jekollmer/PEGS

    Parameters
    ----------

    forceArr : np.ndarray[N]
        Array of force magnitudes at each contact point.

    alphaArr : np.ndarray[N]
        Array of angles that define the direction of force at each contact point

    betaArr : np.ndarray[N]
        Array of angles that define the contact point of the forces, and therefore are
        not adjusted in the force balancing process

    Returns
    -------

    np.ndarray[N] : Magnitude of balanced forces

    np.ndarray[N] : Balanced contact angles alpha

    """

    # TODO: Get this function working
    print("Warning: force balance is not yet implemented, do not call the singleParticleForceBalance function!")
    return forceArr, alphaArr

    # Number of contacts (coordination number, often denoted by z)
    numContacts = len(forceArr)

    if numContacts < 2:
        # Can't do anything with only a single force
        return forceArr, alphaArr
    elif numContacts == 2:
        # For 2 forces, there is a unique process

        # The two force magnitudes must be equal
        balancedForceArr = np.array([forceArr[0], forceArr[0]])

        balancedAlphaArr = np.zeros(2) 
        dBeta = (betaArr[0] - betaArr[1]) / 2
        balancedAlphaArr[0] = np.arccos(np.sin(dBeta))
        if balancedAlphaArr[0] > np.pi/2:
            balancedAlphaArr[0] = np.arccos(np.sin(-dBeta))
        
        # And the other angle must be the opposite
        balancedAlphaArr[1] = - balancedAlphaArr[0]

        return balancedForceArr, balancedAlphaArr

    elif numContacts > 2:
        # We solve any z>2 contacts the same way
        balancedForceArr = np.zeros_like(forceArr)
        balancedAlphaArr = np.zeros_like(alphaArr)

        # To calculate the new force magnitudes, we add up vertical and
        # horizontal components of the other forces
        for i in range(numContacts):
            # These initializations are to not count the case where j = i
            sum1 = -forceArr[i] * np.sin(alphaArr[i])
            sum2 = -forceArr[i] * np.cos(alphaArr[i])
            for j in range(numContacts):
                sum1 += forceArr[j] * np.sin(alphaArr[j] + betaArr[j] - betaArr[i])
                sum2 += forceArr[j] * np.cos(alphaArr[j] + betaArr[j] - betaArr[i])

            balancedForceArr[i] = np.sqrt(sum1**2 + sum2**2)

        # To calculate new alpha values, we 
        for i in range(numContacts):
            sum3 = -balancedForceArr[i] * np.sin(alphaArr[i])
            for j in range(numContacts):
                sum3 += balancedForceArr[j] * np.sin(alphaArr[j])

            balancedAlphaArr[i] = np.arcsin(-sum3/balancedForceArr[i])


        return balancedForceArr, balancedAlphaArr

