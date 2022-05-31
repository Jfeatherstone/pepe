"""
Tracking particles and forces across timesteps to turn triangular arrays into rectangular ones.
"""
import numpy as np
import matplotlib.pyplot as plt

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

    angleArr : list[T,P]
        List of particle rotations for T timesteps, with P(T) particles per timestep.

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

    rectAngleArr : np.ndarray[P,T]
        Particle angles for each timestep. Elements take on a value of `np.nan` if the particle
        does not exist for a given timestep.
    """
    rectForceArr = []
    rectBetaArr = []
    rectAlphaArr = []

    # Scalar
    # TODO Make this capable of following same process as for forces, where
    # identities are maintained even if multiple particles appear/disappear
    # To determine the maximum number of particles, we have to go through each frame
    # and give unique identities to each one. We can't just take the max length of the
    # list, since one frame could have 3 particles, and another 2 particles, but that
    # could be a total of 5 unique particles (or it could be 3 unique particles).
    
    numTimesteps = len(forceArr)

    # We'll build this list as we go, which will be triangular (ie. a 2d list in which
    # each sublist could have a different number of elements). That being said, by keeping
    # placeholders for particles that disappear, we will ensure that this is upper triangular,
    # meaning it will only every grow in size.

    # Initialize the first step
    triSortedParticleOrderArr = [[i for i in range(len(centerArr[0]))]]
    triSortedCenterArr = [[c for c in centerArr[0]]]

    for j in range(1, numTimesteps):

        pOrder = preserveOrderArgsort(triSortedCenterArr[-1], centerArr[j], padMissingValues=True, fillNanSpots=True, maxDistance=np.mean(radiusArr[j]))
        # If a particle disappears, we carry through the old center, so it can be identified again
        # in the future
        centers = [centerArr[j][pOrder[k]] if pOrder[k] is not None else triSortedCenterArr[-1][k] for k in range(len(pOrder))]

        triSortedCenterArr.append(centers)
        triSortedParticleOrderArr.append(pOrder)

    #print(triSortedParticleOrderArr)

    maxNumParticles = len(triSortedCenterArr[-1])

    # First, make the centers array look nice, which we then use to identify
    # particles
    rectCenterArr = np.zeros((maxNumParticles, numTimesteps, 2))
    rectRadiusArr = np.zeros((maxNumParticles, numTimesteps))
    particleOrder = np.zeros((numTimesteps, maxNumParticles), dtype=np.int16)

    for j in range(numTimesteps):
        particleOrder[j] = [pI if pI is not None else -1 for pI in triSortedParticleOrderArr[j]] + [-1]*(maxNumParticles - len(triSortedParticleOrderArr[j]))
        # Now just copy the triangular data into a rectangular array, padding at the end as necessary
        # (and replacing invalid data with np.nan as necessary).
        rectCenterArr[:,j] = [triSortedCenterArr[j][k] if triSortedParticleOrderArr[j][k] is not None else [np.nan, np.nan] for k in range(len(triSortedParticleOrderArr[j]))] + [[np.nan, np.nan] for k in range(maxNumParticles - len(triSortedParticleOrderArr[j]))]
        rectRadiusArr[:,j] = [radiusArr[j][triSortedParticleOrderArr[j][k]] if triSortedParticleOrderArr[j][k] is not None else np.nan for k in range(len(triSortedParticleOrderArr[j]))] + [np.nan for k in range(maxNumParticles - len(triSortedParticleOrderArr[j]))]

    # We have to initialize the first element so that we can then use the
    # preserveOrderSort function to make sure the identities stay
    # consistent
#    rectCenterArr[:,0] = list(centerArr[0]) + [[np.nan, np.nan]]*(maxNumParticles - len(centerArr[0]))
#    rectRadiusArr[:,0] = list(radiusArr[0]) + [np.nan]*(maxNumParticles - len(centerArr[0]))
#
#    particleOrder = np.zeros((numTimesteps, maxNumParticles), dtype=np.int16)
#    particleOrder[0] = np.arange(len(particleOrder[0]))
#    particleExists = np.zeros((numTimesteps, maxNumParticles), dtype=np.int16)
#
#    for i in range(1, numTimesteps):
#        currOrder = preserveOrderArgsort(rectCenterArr[:,i-1], centerArr[i], padMissingValues=True, fillNanSpots=True, maxDistance=np.mean(rectRadiusArr[:,i-1]))
#        # Convert all None to -1 (since None can't be turned into an integer)
#        particleOrder[i] = [ci if ci is not None else -1 for ci in currOrder]
#        rectCenterArr[:,i] = [centerArr[i][particleOrder[i,j]] if particleOrder[i,j] >= 0 else [np.nan, np.nan] for j in range(len(particleOrder[i]))]
#        rectRadiusArr[:,i] = [radiusArr[i][particleOrder[i,j]] if particleOrder[i,j] >= 0 else np.nan for j in range(len(particleOrder[i]))]


    # We now have linked the particles from frame to frame, and can
    # rectangularize the other quantities on a particle-by-particle basis
    for i in range(maxNumParticles):
        # Unfortunately, we cannot determine the maximum number of forces just by doing a
        # call to np.max(len(...)) or something, because eg. two frames could have 3 and 4
        # forces in each, but those could represent up to 7 unique forces total

        # This will be triangular array representing the indices of each force in the final
        # rectangular array
        # We have to make sure this particle exists at the first timestep as well
        #print(particleOrder[0])
        #print(betaArr[0])
        if particleOrder[0,i] >= 0 and particleOrder[0,i] < len(betaArr[0]):
            triSortedBetaArr = [betaArr[0][particleOrder[0,i]]]
            triSortedForceOrderArr = [list(np.arange(len(triSortedBetaArr[0])))]
        else:
            # Otherwise, we just use an empty list
            triSortedBetaArr = [[]]
            triSortedForceOrderArr = [[]]


        for j in range(1, numTimesteps):
            particleIndex = particleOrder[j,i]
   
            # If less than 0, that means this particle is not actually present in this
            # frame, so we just continue. Note that we do have to carry forward the
            # beta values from the previous frame, since otherwise we will run into
            # issues later on.
            if particleIndex < 0:
                triSortedBetaArr.append(triSortedBetaArr[-1])
                triSortedForceOrderArr.append([None for i in range(len(triSortedBetaArr[-1]))])
                continue

            # Correlate the forces between this frame and the last
            #print(triSortedBetaArr[-1])
            order = preserveOrderArgsort(triSortedBetaArr[-1], betaArr[j][particleIndex], padMissingValues=True, maxDistance=maxBetaDisplacement, periodic=True)
            # If we "lose" a force, carry the old beta value through to the new array, so
            # that we can identify it if it ever pops up again.
            betaValues = [betaArr[j][particleIndex][order[k]] if order[k] is not None else triSortedBetaArr[-1][k] for k in range(len(order))]

            #print(f'{betaValues}     {order}     {betaArr[j][particleIndex]}')

            triSortedBetaArr.append(betaValues)
            #triSortedForceOrderArr.append([i for i in order if i is not None])
            triSortedForceOrderArr.append(order)

        #plt.plot([triSortedBetaArr[j][0] for j in range(len(triSortedBetaArr))])
        #plt.show()

        # Since the triangular array is only allowed to grow, the length of the last
        # entry will be the maximum number of *unique* forces.
        #plt.plot([len(ele) for ele in triSortedBetaArr]) # Check that it only grows
        #plt.show()
        maxNumForces = len(triSortedBetaArr[-1])

        if maxNumForces == 0:
            continue

        # Now we can finally create some nice rectangular arrays
        singleParticleForceArr = np.zeros((maxNumForces, numTimesteps))
        singleParticleBetaArr = np.zeros((maxNumForces, numTimesteps))
        singleParticleAlphaArr = np.zeros((maxNumForces, numTimesteps))

        for j in range(numTimesteps):
            particleIndex = particleOrder[j,i]
            
            # If particle doesn't exist for this frame, fill with nan values and move
            # on
            if particleIndex < 0:
                singleParticleBetaArr[:,j] = np.repeat(np.nan, maxNumForces)
                singleParticleForceArr[:,j] = np.repeat(forcePadValue, maxNumForces)
                singleParticleAlphaArr[:,j] = np.repeat(np.nan, maxNumForces)
                continue

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

