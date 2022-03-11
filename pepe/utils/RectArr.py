import numpy as np

from pepe.utils import preserveOrderArgsort

def rectangularizeForceArrays(forceArr, alphaArr, betaArr, centerArr, radiusArr, maxBetaDisplacement=.3, forcePadValue=np.nan, continuousBetas=False):
    """
    Take the ragged arrays representing particles and forces and convert them to
    particle-wise rectangular arrays.

    The force input arrays should be 3 dimensional, with indices as follows:

        `forceArr[timestep index]
                [particle index]
                [force index]`

    The center array should be 3 dimensional as well, with indices as follows:

        `centerArr[timestep index]
                 [particle index]
                 [coordination (x or y) index]`

    The radius array should be 2 dimensional:

        `radiusArr[timestep index]
                 [particle index]`

    The final returned result will be a list of rectangular arrays of the form:

        `rectForceArr[particle index]
                    [force index]
                    [timestep index]`

    There are several advantages to restructing like this:

        1.  The evolution of a single force can be plotted with something
            like `plt.plot(rectForceArr[0][0])`. To do this with the input arrays
            would require quite a bit more work since they are lists, and cannot
            be sliced.
        
        2.  While the top-layer index corresponds to a python list, all of the particle-
            wise arrays are numpy objects, which allows them to be sliced/saved easily.

        3.  Due to potentially having different numbers of particles per frame and different
            numbers of forces acting on each particle, the original arrays and very ragged.
            This makes any sane indexing very hard to do, which is fixed by reshaping the
            arrays. If a force only exists for a subset of the time steps, the rectangular
            result will still identify it as a unique force in all timesteps, just with a
            magnitude of 0 (or `np.nan`; see `forcePadValue`).


    Parameters
    ----------

    forceArr : list[T,P,F]
        Force magnitudes for T timesteps, with P(T) particles per timestep, and F(P(T)) forces
        per particle.

    alphaArr : list[T,P,F]
        Alpha angles for T timesteps, with P(T) particles per timestep, and F(P(T)) forces
        per particle.

    betaArr : list[T,P,F]
        Beta angles for T timesteps, with P(T) particles per timestep, and F(P(T)) forces
        per particle.

    centerArr : list[T,P,2]
        List of particle centers (y,x) for T timesteps, with P(T) particles per timestep.

    radiusArr : list[T,P]
        List of particle radii for T timesteps, with P(T) particles per timestep.

    maxBetaDisplacement : float
        The maximum difference of beta angle between timesteps for which a force will
        still be identified with the previous step. The difference is calculated between
        the current timestep and whatever the last timestep for which that force existed.

    forcePadValue : float
        The value that is recorded for a force that isn't current active for a given frame.
        Common choices would be `np.nan` -- such that the force won't appear when it is
        plotted -- or `0` -- which is a little more physically-inspired.

    continuousBetas : bool
        When a force is not active for a number of timesteps, should the beta value (position)
        of that force be set to `np.nan` (`False`) or should the previous position of the
        force be carried over (`True`)? This is the difference between the force not existing
        for these timesteps, versus the force existing but having a magnitude of 0.

    Returns
    -------

    rectForceArr : list[P](np.ndarray[F,T])
        A list of arrays representing the force magnitude for each force on each particle.
        See description for more info.

    rectAlphaArr : list[P](np.ndarray[F,T])
        A list of arrays representing the alpha angle for each force on each particle.
        See description for more info.

    rectBetaArr : list[P](np.ndarray[F,T])
        A list of arrays representing the beta angle for force on each particle.
        See description for more info.

    rectCenterArr : np.ndarray[P,T,2]
        Particle centers for each timestep. Elements take on a value of `[np.nan, np.nan]`
        if the particle does not exist for a given timestep.

    rectRadiusArr : np.ndarray[P,T]
        Particle radii for each timestep. Elements take on a value of `np.nan` if the particle
        does not exist for a given timestep.
    """
    rectForceArr = []
    rectBetaArr = []
    rectAlphaArr = []

    # Scalar
    # TODO Make this capable of following same process as for forces, where
    # identities are maintained even if multiple particles appear/disappear
    maxNumParticles = np.max([len(betaArr[i]) for i in range(len(betaArr))])
    numTimesteps = len(forceArr)
    # First, make the centers array look nice, which we then use to identify
    # particles
    rectCenterArr = np.zeros((maxNumParticles, numTimesteps, 2))
    rectRadiusArr = np.zeros((maxNumParticles, numTimesteps))

    # We have to initialize the first element so that we can then use the
    # preserveOrderSort function to make sure the identities stay
    # consistent
    rectCenterArr[:,0] = list(centerArr[0]) + [[np.nan, np.nan]]*(maxNumParticles - len(centerArr[0]))
    rectRadiusArr[:,0] = list(radiusArr[0]) + [np.nan]*(maxNumParticles - len(centerArr[0]))

    particleOrder = np.zeros((numTimesteps, maxNumParticles), dtype=np.int16)
    particleOrder[0] = np.arange(len(particleOrder[0]))
    particleExists = np.zeros((numTimesteps, maxNumParticles), dtype=np.int16)

    for i in range(1, numTimesteps):
        currOrder = preserveOrderArgsort(rectCenterArr[:,i-1], centerArr[i], padMissingValues=True, fillNanSpots=True)
        # Convert all None to np.nan (since None can't be turned into an integer)
        particleOrder[i] = [ci if ci is not None else np.nan for ci in currOrder]
        rectCenterArr[:,i] = [centerArr[i][particleOrder[i,j]] if not np.isnan(particleOrder[i,j]) else [np.nan, np.nan] for j in range(len(particleOrder[i]))]
        rectRadiusArr[:,i] = [radiusArr[i][particleOrder[i,j]] if not np.isnan(particleOrder[i,j]) else np.nan for j in range(len(particleOrder[i]))]

    # We now have linked the particles from frame to frame, and can
    # rectangularize the other quantities on a particle-by-particle basis
    for i in range(maxNumParticles):
        # Unfortunately, we cannot determine the maximum number of forces just by doing a
        # call to np.max(len(...)) or something, because eg. two frames could have 3 and 4
        # forces in each, but those could represent up to 7 unique forces total

        # This will be triangular array representing the indices of each force in the final
        # rectangular array
        triSortedBetaArr = [betaArr[0][particleOrder[0,i]]]
        triSortedForceOrderArr = [list(np.arange(len(triSortedBetaArr[0])))]

        for j in range(1, numTimesteps):
            particleIndex = particleOrder[j,i]
            # Correlate the particles between this frame and the last
            #print(triSortedBetaArr[-1])
            order = preserveOrderArgsort(triSortedBetaArr[-1], betaArr[j][particleIndex], padMissingValues=True, maxDistance=maxBetaDisplacement, periodic=True)
            # If we "lose" a force, carry the old beta value through to the new array, so
            # that we can identify it if it ever pops up again.
            betaValues = [betaArr[j][particleIndex][order[k]] if order[k] is not None else triSortedBetaArr[-1][k] for k in range(len(order))]
            #print(betaValues)
            triSortedBetaArr.append(betaValues)
            #triSortedForceOrderArr.append([i for i in order if i is not None])
            triSortedForceOrderArr.append(order)

        # Since the triangular array is only allowed to grow, the length of the last
        # entry will be the maximum number of *unique* forces.
        #plt.plot([len(ele) for ele in triSortedBetaArr])
        #plt.show()
        maxNumForces = len(triSortedBetaArr[-1])

        # Now we can finally create some nice rectangular arrays
        singleParticleForceArr = np.zeros((maxNumForces, numTimesteps))
        singleParticleBetaArr = np.zeros((maxNumForces, numTimesteps))
        singleParticleAlphaArr = np.zeros((maxNumForces, numTimesteps))

        for j in range(numTimesteps):
            # We want to remove the carry over beta values for forces that aren't actually there
            # now, and then pad with nan values for forces that haven't appeared yet

            # Not sure why, but it may be helpful to have the previous beta values carried over,
            # so that can be done by passing a kwarg
            if continuousBetas:
                singleParticleBetaArr[:,j] = [triSortedBetaArr[j][k] for k in range(len(triSortedBetaArr[j]))] + [np.nan]*(maxNumForces - len(triSortedBetaArr[j]))
            else:
                singleParticleBetaArr[:,j] = [betaArr[j][particleIndex][triSortedForceOrderArr[j][k]] if triSortedForceOrderArr[j][k] is not None else np.nan for k in range(len(triSortedBetaArr[j]))] + [np.nan]*(maxNumForces - len(triSortedBetaArr[j]))

            # I apologize to anyone who has to read these lines... They do exactly what is stated above,
            # but boy do they look terrible
            singleParticleForceArr[:,j] = [forceArr[j][particleIndex][triSortedForceOrderArr[j][k]] if triSortedForceOrderArr[j][k] is not None else forcePadValue for k in range(len(triSortedBetaArr[j]))] + [forcePadValue]*(maxNumForces - len(triSortedBetaArr[j]))
            singleParticleAlphaArr[:,j] = [alphaArr[j][particleIndex][triSortedForceOrderArr[j][k]] if triSortedForceOrderArr[j][k] is not None else np.nan for k in range(len(triSortedBetaArr[j]))] + [np.nan]*(maxNumForces - len(triSortedBetaArr[j]))

        # Append all of the nice rectangular arrays
        rectForceArr.append(singleParticleForceArr)
        rectAlphaArr.append(singleParticleAlphaArr)
        rectBetaArr.append(singleParticleBetaArr)

    return rectForceArr, rectAlphaArr, rectBetaArr, rectCenterArr, rectRadiusArr

