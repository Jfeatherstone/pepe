import numpy as np

from pepe.preprocess import circularMask, rectMask, ellipticalMask, mergeMasks
from pepe.analysis import gSquared, adjacencyMatrix
from pepe.simulate import genSyntheticResponse
from pepe.utils import outerSubtract

import numba

from lmfit import minimize, Parameters, fit_report
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

def initialForceSolve(photoelasticSingleChannel, centers, radii, fSigma, pxPerMeter, contactPadding=10, g2MaskPadding=1, contactThreshold=.1, contactMaskRadius=40, neighborEvaluations=4, boundaryMask=None, ignoreBoundary=True, brightfield=False):
    """
    Calculate the approximate forces on each particle based on the photoelastic response.
    No optimization/gradient descent is done in this method; this should be used either where
    rough estimates of the forces are fine, or as an initial condition for a fitting function.

    Heavily based on James Puckett's PhD thesis (2012) and Jonathan Kollmer's PeGS:
    https://github.com/jekollmer/PEGS

    Daniels, K. E., Kollmer, J. E., & Puckett, J. G. (2017). Photoelastic force
    measurements in granular materials. Review of Scientific Instruments, 88(5),
    051808. https://doi.org/10.1063/1.4983049

    Outline of the process:

    0. Track particles to get centers and radii (external)
    1. Calculate average gradient squared for each particle
    2. Find which particles are in contact with each other
    3. Determine if there are any boundary contacts (optional)
    4. Calculate the position of each (boundary or interparticle) force for each particle
    5. Use the average gradient from step 2 to estimate the magnitude of force at each position

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

    contactMaskRadius : float
        The radius of the circular mask applied over a contact to estimate the magnitude
        of the force there. Should be smaller than the radius, usually 25% of the radius is fine.
        For higher forces, larger mask may be needed.

    neightborEvaluations : int
        How many of the closest points to find via the kd tree and test
        as potential contacts. For homogeneous circles, or approximately
        homogenous (< 2:1 size ratio), 4 should be plenty.

    boundaryMask : np.uint8[H,W]
        Optional mask removing uneccesary parts of the image, that is used
        to detect boundaries. Locations with a value of 0 will be assumed to be
        solid walls, that a particle could potentially be in contact with. If
        not provided, or set to None, system will be assumed to be in a rectangular
        box defined the most extreme top/bottom/left/right particles.

    ignoreBoundary : bool
        Whether to not attempt to find any contacts with the boundary. This will take effect
        regardless of whether a boundary mask is passed as a parameter.

    brightfield : bool
        Whether the photoelastic response is seen through a brightfield (True)
        polariscope or darkfield (False) polariscope.
    """

    # We are passed centers and radii
    numParticles = len(centers)
    # This should not include any channels, and should just be h x w
    imageSize = photoelasticSingleChannel.shape
   

    # Triangular arrays are a little wonky, but this type of initialization works
    betaGuessArr = [np.empty(0) for i in range(numParticles)]

    # Find neighbors
    # (This is the unweighted adjacency matrix, so a 1 if two particles are neighbors, 0 otherwise)
    # np.eye is the identity, which removes all diagonal "contacts" (a particle with itself)
    adjMat = adjacencyMatrix(centers, radii, contactPadding, neighborEvaluations) - np.eye(numParticles)

    # This includes duplicates, but that is ok because as long as we keep
    # the index order consistent, each force will show up only once for each particle
    # (but twice overall, which is what we want)
    contacts = np.transpose(np.where(adjMat))

    # Each pair of indices
    for p in contacts:
        betaGuessArr[p[0]] = np.append(betaGuessArr[p[0]], np.arctan2(centers[p[1]][1] - centers[p[0]][1], centers[p[1]][0] - centers[p[0]][0]))

    # Find potential wall contacts (optional)
    if not ignoreBoundary:
        # If no boundary is given, but ignoreBoundary = False, we just make a box around
        # our system
        if boundaryMask is None:
            # Working in image conventions, so top left corner is 0,0
            # So y value increases as you move downwards
            topLeftCorner = np.array([np.min(centers[:,0] - radii), np.min(centers[:,1] - radii)])
            dimensions = np.array([np.max(centers[:,0] + radii) - topLeftCorner[0],
                                   np.max(centers[:,1] + radii) - topLeftCorner[1]])

            boundaryMask = rectMask(photoelasticSingleChannel.shape, topLeftCorner, dimensions, channels=None)

        # See detectWallContacts for more information
        numWallContacts, wallBetaArr, wallG2AvgArr = detectWallContacts(centers, radii, boundaryMask, photoelasticSingleChannel, contactPadding=contactPadding)
        for i in range(numParticles):
            for j in range(numWallContacts[i]):
                betaGuessArr[i] = np.append(betaGuessArr[i], wallBetaArr[i][j])

    # Alpha is very easy: we just assume all forces are radial
    # This creates an empty array the same shape as the beta one,
    # but with all of the values set to 0 (again, triangular arrays are wonky)
    alphaGuessArr = np.zeros(len(betaGuessArr), dtype='object')
    for i in range(len(alphaGuessArr)):
        alphaGuessArr[i] = np.zeros(len(betaGuessArr[i]))

    # Initialize force the same way, but this will end up with
    # actual values
    forceGuessArr = np.zeros(len(betaGuessArr), dtype='object')
    for i in range(len(forceGuessArr)):
        forceGuessArr[i] = np.zeros(len(betaGuessArr[i]))

    # Calculate G^2 of the image
    gSqr = gSquared(photoelasticSingleChannel)

    # Find G2 calibration value
    # Mean of the radii should be fine, though TODO would be a good idea to have options here
    g2Cal = g2ForceCalibration(fSigma, np.mean(radii), pxPerMeter, brightfield=brightfield)

    for i in range(numParticles):
        g2Mask = circularMask(photoelasticSingleChannel.shape, centers[i], radii[i] - g2MaskPadding)[:,:,0]

        # This is the average g2 of the whole particle
        avgGSqr = np.sum(gSqr * g2Mask) / np.sum(g2Mask)

        # Now allocate this force to each contact
        contactG2Arr = np.zeros(len(forceGuessArr[i]))
        # We have to find the average gsqr at each contact
        for j in range(len(forceGuessArr[i])):
            contactMask = circularMask(photoelasticSingleChannel.shape,
                                       centers[i] + radii[i]*np.array([np.cos(betaGuessArr[i][j]), np.sin(betaGuessArr[i][j])]),
                                      contactMaskRadius)[:,:,0]
            # Now make sure it stays within the particle
            contactMask = 1 - np.int16((contactMask + g2Mask - 2) > 0)

            # This is average g2 of each contact
            contactG2Arr[j] = np.sum(gSqr * contactMask) / np.sum(contactMask)

        forceGuessArr[i][:] = avgGSqr * contactG2Arr / np.sum(contactG2Arr) / g2Cal

    # while (!solved)
    
        # Few iterations of least squares for each remaining particle
        # TODO

        # Find the ones with the least error, and perform a many more iterations
        # TODO

        # Apply force balance to those particles (optional)
        # TODO

        # Fix those contacts, and carry over the force to their neighbors
        # TODO

        # Remove fixed particles from the queue
        # TODO

    return forceGuessArr, alphaGuessArr, betaGuessArr


def forceOptimize(forceGuessArr, betaGuessArr, alphaGuessArr, radius, center, realImage, fSigma, pxPerMeter, brightfield, parametersToFit=['f', 'a'], method='nelder', maxEvals=300, forceBounds=(0, 5), betaBounds=(-np.pi, np.pi), alphaBounds=(0, np.pi), forceTol=.5, betaTol=.5, alphaTol=.1, useTolerance=True, returnOptResult=False, allowAddForces=True, allowRemoveForces=True, minForceThreshold=.01, newBetaContactMaskRadius=30, newBetaMinSeparation=.4, newBetaG2Height=.0005, missingForceChiSqrThreshold=2.1e8):
    """
    Optimize an initial guess for the forces acting on a particle using
    a nonlinear minimization function.

    Parameters
    ----------

    forceGuessArr : np.ndarray[Z]
        The initial guess for magnitudes of each force that is acting on the particle.

    betaGuessArr : np.ndarray[Z]
        The initial guess for the positional angles where each force is acting (in radians).

    alphaGuessArr : np.ndarray[Z]
        The initial guess for the directional angles for each force acting on the particle (in radians).

    radius : float
        The radius of the particle, in pixels.

    center : [int, int]
        The coordinates of the center, [y,x], of the particle.

    realImage : np.ndarray[H,W]
        The photoelastic channel that the optimizer will compare against. Can be a composite image
        include other particles, since a mask will be applied around the given particle.

    parametersToFit : ['f'[[, 'b'], 'a']]
        Which parameters should be allowed to be varied in the process of minimizing the function; 'f' for
        force magnitude, 'b' for beta positional angle, 'a' for alpha directional angle. Must include at least one.

        Can also specify to perform multiple optimizations on different variables by providing a 2D list of
        parameters keys (any of 'f', 'a', 'b') e.g. [ ['f', 'b'], ['f', 'a'] ] will first optimize the force
        and beta values, then optimize the force and alpha values. When performing multiple optimizations,
        the result of the previous one is used as initial condition for the following one, though the residual
        array includes information about all optimizations. 

        Values for `maxEvals`, `forceTol`, `betaTol`, `alphaTol`, and `method` can be provided as lists of
        acceptable values to specify different options for each optimization e.g. `method=['nelder', 'cobyla']`
        will use the Nelder-Mead optimization scheme on the first pass, and then COBYLA on the second one. If
        a single value is provided for any of these parameters, that will be used for all optimizations.

    method : ['nelder', 'bfgsb', 'powell', 'cobyla']
        The method to use for optimization. See wiki page on Force Solver for more information on selecting
        an appropriate one. If in doubt, 'nelder' is usually a safe bet.

        Can be provided as a list of values when performing multiple optimizations; see `parametersToFit`.

    maxEvals : int
        The maximum number of function evaluations before the minimizer exits. Convergence can occur before
        this number of evaluations, but process will be interrupted at this value regardless of whether
        or not the result has converged.

        Can be provided as a list of values when performing multiple optimizations; see `parametersToFit`.

    forceBounds : (float, float)
        The upper and lower limits for the values of force magnitudes that the minimizer can explore, 
        assuming that `useTolerance` is set to False.

    betaBounds : (float, float)
        The upper and lower limits for the values of beta that the minimizer can explore, 
        assuming that `useTolerance` is set to False.

    alphaBounds : (float, float)
        The upper and lower limits for the values of alpha that the minimizer can explore, 
        assuming that `useTolerance` is set to False.

    forceTol : float
        If `useTolerance` is set to True, this value will be used to calculate the bounds for each
        force individually, as the initial value minus this tolerance, and the initial value plus
        this tolerance.

        Can be provided as a list of values when performing multiple optimizations; see `parametersToFit`.

    betaTol : float
        If `useTolerance` is set to True, this value will be used to calculate the bounds for each
        beta individually, as the initial value minus this tolerance, and the initial value plus
        this tolerance.

        Can be provided as a list of values when performing multiple optimizations; see `parametersToFit`.

    alphaTol : float
        If `useTolerance` is set to True, this value will be used to calculate the bounds for each
        alpha individually, as the initial value minus this tolerance, and the initial value plus
        this tolerance.

        Can be provided as a list of values when performing multiple optimizations; see `parametersToFit`.

    useTolerance : bool
        Whether to calculate the bounds for each parameters as the initial value plus/minus a
        certain tolerance (True) or to use the set intervals provided via the `forceBounds`, 
        `betaBounds`, and `alphaBounds` parameters (False).

    returnOptResult : bool
        Whether to return the optimizer result object, which includes all parameters values and
        potentially uncertainties (True) or just the optimized values of the forces, betas, alphas,
        and residuals (False). Note that if multiple optimizations are being performed, only the
        final result object will be returned.
    
    minForceThreshold : float
        The minimizer will automatically remove any forces whose magnitude is lower than
        this value between fittings and after the final one.
    """

    residuals = []

    # If we are passed a 2d list of parametersToFit, that means we 
    # are to perform multiple minimizations, likely with different
    # parameters being optimized each time.

    # It is easier to set up this method to run for an arbitrary number
    # of optimizations, even though most of the time that number of
    # optimizations will be one

    # So if we aren't given a 2d list structure, we have to make one
    if type(parametersToFit[0]) is not list:
        parametersToFitList = [parametersToFit]
    else:
        parametersToFitList = parametersToFit

    numFits = len(parametersToFitList)

    # Different numbers of evaluations can be provided for each optimization
    # or just the one value can be used for all of them
    if type(maxEvals) is not list:
        maxEvalsList = [maxEvals for i in range(numFits)]
    else:
        maxEvalsList = maxEvals

    # The difference minimizations can use different methods
    if type(method) is not list:
        methodList = [method for i in range(numFits)]
    else:
        maxEvalsList = method
    
    # Same deal for all of the tolerances (we don't do bound intervals, since tolerances
    # are the recommended way to handle parameters bounds)
    if type(forceTol) is not list:
        forceTolList = [forceTol for i in range(numFits)]
    else:
        forceTolList = forceTol

    if type(betaTol) is not list:
        betaTolList = [betaTol for i in range(numFits)]
    else:
        betaTolList = betaTol

    if type(alphaTol) is not list:
        alphaTolList = [alphaTol for i in range(numFits)]
    else:
        alphaTolList = alphaTol

    # Make sure everything is the same length
    assert len(parametersToFitList) == numFits, 'Invalid parametersToFit provided'
    assert len(maxEvalsList) == numFits, 'Invalid maxEvals provided'
    assert len(methodList) == numFits, 'Invalid method provided'
    assert len(forceTolList) == numFits, 'Invalid forceTol provided'
    assert len(betaTolList) == numFits, 'Invalid betaTol provided'
    assert len(alphaTolList) == numFits, 'Invalid alphaTol provided'

    # Setup our function based on what parameters we are fitting
    # We want to avoid any if statements within the function itself, since
    # that will be evaluated many many times.
    # lmfit has a nice setup in that you can denote whether a variable can be
    # changed or not, which means we don't actually have to change which variables
    # are passed to the function.
    def objectiveFunction(params, trueImage, z, radius, center):
        forceArr = np.array([params[f"f{j}"] for j in range(z)])
        betaArr = np.array([params[f"b{j}"] for j in range(z)])
        alphaArr = np.array([params[f"a{j}"] for j in range(z)])

        synImage = genSyntheticResponse(forceArr, alphaArr, betaArr, fSigma, radius, pxPerMeter, brightfield, imageSize=trueImage.shape, center=center)
        residuals.append(np.sum(np.abs(synImage - trueImage)))
        return np.sum(np.abs(synImage - trueImage))


    # Mask our real image
    particleMask = circularMask(realImage.shape, center, radius)[:,:,0]
    maskedImage = realImage * particleMask

    # Since we may be doing multiple fits, we want to set up the initial conditions
    # such that each fit uses the result of the previous one (and the first one
    # of course uses what is provided to the function)
    forceArr = np.array(forceGuessArr.copy())
    betaArr = np.array(betaGuessArr.copy())
    alphaArr = np.array(alphaGuessArr.copy())

    # Now that we have all of that bookkeeping done, we can actually get on
    # to doing the minimization
    result = None
    i = 0
    # We use a while loop here because we may want to repeat a fit if
    # we add a new force, which isn't possible in a for loop
    while i < numFits:

        z = len(forceArr)

        # Out fitting parameters
        # if vary kwarg is false, that value won't be fit
        params = Parameters()

        for j in range(z):
            if useTolerance:
                # Have to make sure that certain values aren't allowed to go negative, but
                # otherwise the bounds are just the initial value +/- the tolerances
                params.add(f'f{j}', value=forceArr[j], vary='f' in parametersToFitList[i], min=max(forceArr[j]-forceTolList[i], 0), max=forceArr[j]+forceTolList[i])
                params.add(f'b{j}', value=betaArr[j], vary='b' in parametersToFitList[i], min=max(betaArr[j]-betaTolList[i], -np.pi), max=min(betaArr[j]+betaTolList[i], np.pi))
                params.add(f'a{j}', value=alphaArr[j], vary='a' in parametersToFitList[i], min=max(alphaArr[j]-alphaTolList[i], 0), max=alphaArr[j]+alphaTolList[i])
            else:
                params.add(f'f{j}', value=forceArr[j], vary='f' in parametersToFitList[i], min=forceBounds[0], max=forceBounds[1])
                params.add(f'b{j}', value=betaArr[j], vary='b' in parametersToFitList[i], min=betaBounds[0], max=betaBounds[1])
                params.add(f'a{j}', value=alphaArr[j], vary='a' in parametersToFitList[i], min=alphaBounds[0], max=alphaBounds[1])


        # Now do the optimization
        result = minimize(objectiveFunction, params,
                         args=(maskedImage, z, radius, center),
                         method=methodList[i], nan_policy='omit', max_nfev=maxEvalsList[i])

        # Copy over the new values of the forces, alphas, and betas
        for j in range(z):
            forceArr[j] = result.params[f"f{j}"] 
            betaArr[j] = result.params[f"b{j}"] 
            alphaArr[j] = result.params[f"a{j}"] 

        # ---------------------
        # Detect missing forces
        # ---------------------
        # If the code detects there is a missing force (no idea how yet)
        if result.chisqr > missingForceChiSqrThreshold and allowAddForces:
            # We sweep around the edge of the particle to see if there
            # are any regions that look like they could have a force
            # (denoted by a particularly high g2 value, or rather a peak)
            testBetaCount = 30
            avgG2Arr = np.zeros(testBetaCount)
            newBetaArr = np.linspace(-np.pi, np.pi, testBetaCount)

            # Calculate all of the g2s around the edge of the particle
            gSqr = gSquared(realImage)
            for j in range(testBetaCount):
                contactPoint = center + radius * np.array([np.cos(newBetaArr[j]), np.sin(newBetaArr[j])])
                
                # Create a mask just over the small area inside of the particle
                contactMask = circularMask(realImage.shape, contactPoint, newBetaContactMaskRadius)[:,:,0]
                contactMask = (contactMask + particleMask) == 2

                avgG2Arr[j] = np.sum(contactMask * gSqr) / np.sum(contactMask)

            # Identify any peaks in the average g2s
            peakIndArr = find_peaks(avgG2Arr, height=newBetaG2Height)[0]
            peakIndArr = np.sort(peakIndArr)

            # Make sure that there aren't any artifacts of periodicity
            # Usually this isn't actually a problem, because the peak
            # finding algorithm requires a proper peak, which can only
            # be on one side (but we'll leave it here just in case)
            if np.arange(3).any() in peakIndArr and np.arange(len(avgG2Arr)-3, len(avgG2Arr)).any() in peakIndArr:
                # Remove last entry
                peakIndArr = peakIndArr[:-1]

            peakBetaArr = newBetaArr[peakIndArr]
            
            # Now we have a list of likely points, we need to see if our original
            # list is missing any of these.
            differenceArr = np.abs(np.subtract.outer(peakBetaArr, betaArr))

            # Check to see if there is a new peak that doesn't have
            # a previous force close to it
            for j in range(len(peakBetaArr)):
                if np.min(differenceArr[j]) > newBetaMinSeparation:
                    # Add the new force
                    betaArr = np.append(betaArr, peakBetaArr[j])
                    forceArr = np.append(forceArr, .1) # Value isn't too imporant here
                    alphaArr = np.append(alphaArr, 0.)


            # If we have added a force, we should run the optimization again, and see if it improves
            if len(forceArr) > z:
                print(f'Added {len(forceArr) - z} force(s).')
                # We also want to make sure we're allowed to vary beta on the next iteration
                parametersToFitList[i] += ['b'] # (It's okay that it might be in there twice)
                # This skips the i += 1 at the end of the loop, and makes the optimization run again
                continue


        # ------------------------------------
        # Remove forces that don't do anything
        # ------------------------------------
        if len(forceArr[forceArr < minForceThreshold]) > 0 and allowRemoveForces:
            # Remove forces that aren't actually doing anything
            betaArr = betaArr[forceArr > minForceThreshold]
            alphaArr = alphaArr[forceArr > minForceThreshold]
            # This one has to be done last for the other indexing to work
            forceArr = forceArr[forceArr > minForceThreshold]
            print(f'Removed {z - len(forceArr)} force(s).')
       
        # Iterate (since we have a while not a for)
        i += 1

    if returnOptResult:
        return result

    else:
        return forceArr, betaArr, alphaArr, residuals



@numba.jit(nopython=True)
def singleParticleForceBalance(forceArr, alphaArr, betaArr):
    """
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


@numba.jit(nopython=True)
def g2ForceCalibration(fSigma, radius, pxPerMeter=1., g2Padding=1, alphaArr=np.array([0., 0.]), betaArr=np.array([0., -np.pi]), forceSteps=100, forceBounds=np.array([.01, 1.]), brightfield=True):
    """
    Use synthetic photoelastic response to fit the conversion constant between
    gradient squared value and force (in Newtons), assuming a linear relationship.

    Note that this computes the least squares of force (N) vs **average** gradient
    squared. The Matlab just uses the sum of gradient squared, but this is not invariant
    under changes of resolution, so I have opted to use the average. Because of this, a
    calibration value calculated in Matlab will **not** work here unless you divide out
    the number of points it is summed over first.

    Parameters
    ----------

    fSigma : float
        Stress optic coefficient, relating to material thickness, wavelength of light
        and other material property (C).

    radius : float
        Radius of the particle that is being simulated in pixels. If pxPerMeter is
        not provided (or set to 1), this value will be assumed to already have been converted to meters.

    contactMaskRadius : float
        Radius of the circular mask that is applied over each contact to find the average gradient
        squared.

    pxPerMeter : float
        The number of pixels per meter for the simulated image. If not provided, or set to 1, the radius
        value will be assumed to already have been converted to meters.

    g2Padding : int
        Number of pixels to ignore at the edge of the particle. We don't expect any boundary
        artifacts in our synthetic data, but we will eventually do this for the real data,
        so it is important to keep the size of the particles the same throughout.

    alphaArr : np.ndarray[Z]
        Array of angles representing force contact angles.

    betaArr : np.ndarray[Z]
        Array of angles representing force contact positions.

    forceSteps : int
        The number of points to use for fitting our line of g^2 vs. force.

    forceBounds : [float, float]
        The minimum and maximum value of force applied to calculate the calibration value.

    brightfield : bool
        Whether the intensity should be simulated as seen through a brightfield (True)
        polariscope or darkfield (False) polariscope.

    Returns
    -------

    g2ForceSlope : float
        Slope found via linear regression to convert average g^2 to force.
    """

    # The magnitude of the forces that will be acting at each step
    forceValues = np.linspace(forceBounds[0], forceBounds[1], forceSteps)
    gSqrAvgArr = np.zeros(forceSteps)

    imageSize = (np.int16(radius*2)+11, np.int16(radius*2)+11)
    center = np.array([imageSize[0]/2, imageSize[1]/2], dtype=np.int64)
    particleMask = circularMask(imageSize, center, radius - g2Padding)[:,:,0]

    # The contact mask is a circle placed over the edge of the particle where the force is applied
    #contactMask1 = circularMask(imageSize,
    #                            np.array([imageSize[0]/2 + radius*np.cos(betaArr[0]), imageSize[1]/2 + radius*np.sin(betaArr[0])]),
    #                            contactMaskRadius)[:,:,0]

    #contactMask2 = circularMask(imageSize,
    #                            np.array([imageSize[0]/2 + radius*np.cos(betaArr[1]), imageSize[1]/2 + radius*np.sin(betaArr[1])]),
    #                            contactMaskRadius)[:,:,0]

    # Get rid of the parts outside of the circle
    #contactMask1 = contactMask1 * particleMask
    #contactMask2 = contactMask2 * particleMask

    # Add them together
    #contactMask = contactMask1 + contactMask2

    # To divide out the number of points
    #numPoints = np.sum(contactMask)
    numPoints = np.sum(particleMask)

    for i in range(forceSteps):

        # Assume two forces acting on the particle with equal magnitude
        forceArr = np.array([forceValues[i], forceValues[i]])
       
        # Create a synthetic photoelastic response
        particleImg = genSyntheticResponse(forceArr, alphaArr, betaArr, fSigma, radius, pxPerMeter, brightfield, imageSize, center)

        # Calculate the gradient
        gSqr = gSquared(particleImg)

        # Multiply by the mask to avoid weird edge effects
        gSqrAvgArr[i] = np.sum(gSqr * particleMask) / numPoints

    # Now fit a straight line to the data
   
    # Create col vector of forces
    # Note that we multiply by 2 here
    # Since we are calculating g2 over the entire particle, we should
    # compare it to the force on the entire particle, not just one side.
    forceColMat = forceValues.reshape((forceSteps, 1)) * 2
    # Perform least squares
    solution = np.linalg.lstsq(forceColMat, gSqrAvgArr)

    return solution[0][0]


def g2ForceCalibrationDebug(fSigma, radius, pxPerMeter, alphaArr=np.array([0., 0.]), betaArr=np.array([0., -np.pi]), forceSteps=100, forceBounds=np.array([.01, 1.]), brightfield=True):
    # TODO: Also find the point at which the linear fit breaks down, so we can
    # return the max amount of force that can be converted using this method
    # before it stops working
    """
    Use synthetic photoelastic response to fit the conversion constant between
    gradient squared value and force (in Newtons), assuming a linear relationship.

    Returns the X and Y values that a line would be fit to, instead of the
    fit paramters themselves. See pepe.analysis.g2ForceCalibration().

    Note that this computes the least squares of force (N) vs **average** gradient
    squared. The Matlab just uses the sum of gradient squared, but this is not invariant
    under changes of resolution, so I have opted to use the average. Because of this, a
    calibration value calculated in Matlab will **not** work here unless you divide out
    the number of points it is summed over first.

    Parameters
    ----------

    fSigma : float
        Stress optic coefficient, relating to material thickness, wavelength of light
        and other material property (C).

    radius : float
        Radius of the particle that is being simulated in pixels. If pxPerMeter is
        not provided (or set to 1), this value will be assumed to already have been converted to meters.

    contactMaskRadius : float
        Radius of the circular mask that is applied over each contact to find the average gradient
        squared.

    pxPerMeter : float
        The number of pixels per meter for the simulated image. If not provided, or set to 1, the radius
        value will be assumed to already have been converted to meters.

    alphaArr : np.ndarray[Z]
        Array of angles representing force contact angles.

    betaArr : np.ndarray[Z]
        Array of angles representing force contact positions.

    forceSteps : int
        The number of points to use for fitting our line of g^2 vs. force.

    forceBounds : [float, float]
        The minimum and maximum value of force applied to calculate the calibration value.

    brightfield : bool
        Whether the intensity should be simulated as seen through a brightfield (True)
        polariscope or darkfield (False) polariscope.

    Returns
    -------

    forceArr : np.ndarray[forceSteps]
        Force applied (in Newtons) at each point

    g2AvgArr : np.ndarray[forceSteps]
        Resultant average gradient squared across the particle at each point.
    """

    # The magnitude of the forces that will be acting at each step
    forceValues = np.linspace(forceBounds[0], forceBounds[1], forceSteps)
    gSqrAvgArr = np.zeros(forceSteps)

    imageSize = np.array([np.int(radius*2.2), np.int(radius*2.2)])
    particleMask = circularMask(imageSize, imageSize/2, radius)[:,:,0]

    # The contact mask is a circle placed over the edge of the particle where the force is applied
    #contactMask1 = circularMask(imageSize,
    #                            np.array([imageSize[0]/2 + radius*np.cos(betaArr[0]), imageSize[1]/2 + radius*np.sin(betaArr[0])]),
    #                            contactMaskRadius)[:,:,0]
    
    #contactMask2 = circularMask(imageSize,
    #                            np.array([imageSize[0]/2 + radius*np.cos(betaArr[1]), imageSize[1]/2 + radius*np.sin(betaArr[1])]),
    #                            contactMaskRadius)[:,:,0]

    # Get rid of the parts outside of the circle
    #contactMask1 = contactMask1 * particleMask
    #contactMask2 = contactMask2 * particleMask

    # Add them together
    #contactMask = contactMask1 + contactMask2

    # To divide out the number of points
    #numPoints = np.sum(contactMask)
    numPoints = np.sum(particleMask)

    for i in range(forceSteps):

        # Assume two forces acting on the particle with equal magnitude
        forceArr = np.array([forceValues[i], forceValues[i]])
       
        # Create a synthetic photoelastic response
        particleImg = genSyntheticResponse(forceArr, alphaArr, betaArr, fSigma, radius, pxPerMeter, brightfield, imageSize, imageSize/2)

        # Calculate the gradient
        gSqr = gSquared(particleImg)

        # Multiply by the mask to avoid weird edge effects
        gSqrAvgArr[i] = np.sum(gSqr * particleMask) / numPoints

    # We multiply by 2 because that is the total force acting on the particle
    return 2*forceValues, gSqrAvgArr


#@numba.jit(nopython=True)
def detectWallContacts(centers, radii, boundaryMask, photoelasticSingleChannel=None, contactPadding=10, g2EdgePadding=.95, angleClusterThreshold=.2, contactMaskRadius=50, maxContactExtent=.75):
    """
    Detect potential particle contacts with the wall.

    Parameters
    ----------

    centers : np.ndarray[N,2] 
        A list of N centers of format [y, x].

    radii : np.ndarray[N]
        A list of N radii, corresponding to each particle center

    boundaryMask : np.uint8[H,W]
        Mask removing uneccesary parts of the image, that is used
        to detect boundaries. Locations with a value of 0 will be assumed to be
        solid walls, that a particle could potentially be in contact with.

    photelasticSingleChannel : np.uint8[H,W]
        A single channel image in array form of the photoelastic response of particles.
        Is used to determine if a contact is actually force bearing or not. If not 
        provided, all potential contacts will be returned.

    contactPadding : int
        Maximum difference between distance between a particle's edge and the wall
        that will still be tested as a wall contact.

    g2EdgePadding : int or float
        Number of pixels to ignore at the edge of each particle when calculating the average
        G^2. If float value < 1 is passed, gradient mask radius will be taken as that percent
        of the full particle radius.

    angleClusterThreshold : float
        The minimum difference in consecutive angle (beta) for which two wall contacts
        will be considered unique (and not merged into the same contact).

    contactMaskRadius : int
        The size of the circular mask that is used to determine average gradient squared
        value around detected contacts.

    maxContactExtent : float
        The maximum range of angles that can be included in a single cluster. If any particular
        cluster exceeds this value, it will be divided up into multiple new clusters.

    Returns
    -------

    numWallContacts : np.ndarray[N]
        The number of wall contacts for each particle (denoted as Z_i in other return values).
    
    betaArr : np.ndarray[N, Z_i]
        A list of angles for each particle in which there likely is a wall contact.

    contactG2Arr : np.ndarray[N, Z_i]
        A list of average gradient squared values for each wall contact, to be
        multiplied by the g2 calibration value to get the initial force magnitude guess.
    """

    # Figure out how much of the particle we will be calculating the g2 over
    g2MaskRadii = radii.astype(np.float64)
    if g2EdgePadding < 1.:
        g2MaskRadii = (radii.astype(np.float64) * g2EdgePadding)
    elif g2EdgePadding > 1:
        g2MaskRadii = (radii - g2EdgePadding).astype(np.float64)

    # Things that will be returned
    numWallContacts = np.zeros(len(centers), dtype=np.int16)
    betaArr = []
    contactG2Arr = []

    # Iterate over each particle
    for i in range(len(centers)):

        # Create a mask slightly larger than the particle
        wallContactMask = circularMask(boundaryMask.shape, centers[i], radii[i] + contactPadding)

        # Find the overlap (similarity) of this enlarged particle mask and the boundary mask
        invBoundaryMask = 1 - boundaryMask
        similarity = np.floor((invBoundaryMask + wallContactMask).astype(np.double)/2).astype(np.uint8)[:,:,0]

        # Determine if there is a chance of a wall contact
        if np.sum(similarity) > 0:
            # ------------------------------
            # Find every point that overlaps
            # ------------------------------

            # Normally we could just do this:
            #points = np.transpose(np.where(cMask > 0))
            # but numba doesn't quite like this way, so we have to be 
            # a little more creative

            whereIndices = np.where(similarity > 0)
            points = np.zeros((len(whereIndices[0]), 2), dtype=np.int16)
            # There is a chance that these indices are backwards, but
            # because we have rotational symmetry, it doesn't really matter...
            # BUT if there is ever some weird anisotropy bug or something,
            # try switching these indices
            points[:,0] = whereIndices[0]
            points[:,1] = whereIndices[1]

            # ------------------------------
            # Cluster them based on position
            # ------------------------------

            # Convert to angles
            angles = np.arctan2(points[:,1] - centers[i][1], points[:,0] - centers[i][0])
            
            # Sort the angles, since they may not exactly be in order
            sortedIndices = np.argsort(angles)
            angles = np.sort(angles)

            # To apply it to the original points, we would have to reverse the sorting
            # we did earlier, or much easier, just sort the original points the same way
            points = points[sortedIndices]

            # Calculate the difference between angles
            # There should be a jump once we move to a new cluster
            dAngle = angles[1:] - angles[:-1]

            # Add 1 to these, because we started at 1
            # These are the indices of places that have large jumps in angle
            clusterSeparation = np.where(dAngle > angleClusterThreshold)[0].astype(np.int16) + 1

            # Generate an array with labels for each angle
            numClusters = len(clusterSeparation) + 1
            labels = np.zeros(len(points))
            startingIndex = 0

            for j in range(numClusters-1):
                labels[startingIndex:clusterSeparation[j]] = j
                startingIndex = clusterSeparation[j]
            # And the final cluster
            labels[startingIndex:] = numClusters-1

            # Check to see if we have any artifacts of the periodicity of the angle
            # (and fix them by setting the last cluster to be equal to the first one)
            #print(angles[0], angles[-1])
            if numClusters > 1 and abs(angles[0] - angles[-1]) < angleClusterThreshold or abs(abs(angles[0] - angles[-1]) - 2*np.pi) < angleClusterThreshold:
                labels[labels == np.max(labels)] = 0
                numClusters -= 1

            # --------------------------------------
            # Calculate the centroid of each cluster
            # --------------------------------------
            clusterCentroids = np.zeros((numClusters, 2))
            for j in range(numClusters):
                clusterCentroids[j] = [np.mean(points[labels == j][:,0]), np.mean(points[labels == j][:,1])]
 

            # --------------------------------------------------------------
            # Calculate the angle with respect to the particle center (beta)
            # --------------------------------------------------------------
            clusterBetas = np.zeros(numClusters)
            for j in range(numClusters):
                clusterBetas[j] = np.arctan2(clusterCentroids[j,1] - centers[i][1], clusterCentroids[j,0] - centers[i][0])

            # -------------------------------------
            # Divide up big clusters (if necessary)
            # -------------------------------------
            newBetas = np.zeros(0)
            for j in range(numClusters):
                # First, calculate the extent of the cluster
                # This isn't as simple as subtract max from min, because
                # of the periodicity, so the most reliable method is as follows

                # Locate every unique angle in this cluster (in order)
                uniqueBetas = np.sort(np.unique(angles[labels == j]))

                # If you only have 1 beta, then clearly we don't need to divide
                # this cluster up
                if len(uniqueBetas) < 2:
                    newBetas = np.append(newBetas, clusterBetas[j])
                    continue

                clusterBounds = np.array([np.max(np.array([uniqueBetas[0], uniqueBetas[-1]])), np.min(np.array([uniqueBetas[0], uniqueBetas[-1]]))])
                clusterExtent = clusterBounds[0] - clusterBounds[1]

                # This is usually a good way to identify that the region
                # passes across the top of the circle
                if clusterExtent < .01 or 2*np.pi - clusterExtent < .01:
                #if (clusterBetas[j] < clusterBounds[0] and clusterBetas[j] > clusterBounds[1]):
                    clusterBounds = [np.max(uniqueBetas[uniqueBetas < 0]), np.min(uniqueBetas[uniqueBetas > 0])]
                    clusterExtent = 2*np.pi - (clusterBounds[1] - clusterBounds[0])

                if clusterExtent > maxContactExtent:
                    numNewClusters = np.int16(np.ceil(clusterExtent / maxContactExtent))
                    dBeta = clusterExtent/numNewClusters
                    newBetas = np.append(newBetas, np.linspace(clusterBetas[j] + clusterExtent/2., clusterBetas[j] - clusterExtent/2., numNewClusters))
                else:
                    newBetas = np.append(newBetas, clusterBetas[j])
                    
            #print(newBetas)

            clusterBetas = newBetas.copy()
            numClusters = len(clusterBetas)

            # Now we want to recalculate our centroids, since there are
            # potentially some new ones
            clusterCentroids = np.zeros((numClusters, 2))
            for j in range(numClusters):
                clusterCentroids[j] = centers[i] + radii[i] * np.array([np.cos(clusterBetas[j]), np.sin(clusterBetas[j])])

            # --------------------------------------------------------------------------
            # Apply a mask to get the magnitude of the average g2 value for that contact
            # --------------------------------------------------------------------------
            # We only do this if we are provided a photoelastic image
            clusterAvgG2 = np.zeros(numClusters)

            if photoelasticSingleChannel is not None:
                # Calculate G2
                gSqr = gSquared(photoelasticSingleChannel)

                for j in range(numClusters):
                    # Create an elliptical mask around the area the photoelastic response would
                    # be in. This is actually more a half ellipse, since we want a reasonably
                    # wide area around the contact itself, and only a small bit of the center
                    # of the particle.
                    #majorAxisDir = clusterCentroids[j] - centers[i]
                    #majorAxisDir /= np.sqrt(majorAxisDir[0]**2 + majorAxisDir[1]**2)
                    #ellipseEndpoint = majorAxisDir * 2 *radii[i] + centers[i]
                    #contactMask = ellipticalMask(photoelasticSingleChannel.shape, centers[i], ellipseEndpoint, radii[i]/2)

                    # Just kidding, a circular mask seems to work better
                    contactMask = circularMask(photoelasticSingleChannel.shape, clusterCentroids[j], contactMaskRadius)

                    # Now cut off the part outside of the particle
                    contactMask = (contactMask + circularMask(photoelasticSingleChannel.shape, centers[i], g2MaskRadii[i]))[:,:,0] == 2

                    # Calculate average G2
                    clusterAvgG2[j] = np.sum(contactMask * gSqr) / np.sum(contactMask)


            # ---------------------------
            # Save all of the information
            # ---------------------------            
            numWallContacts[i] = len(clusterBetas)
            betaArr.append(clusterBetas)
            contactG2Arr.append(clusterAvgG2)

        else:
            # No possible wall contacts
            # numWallContacts is initialized at 0, so don't need to set it
            # Numba doesn't allow untyped lists, so this is a little trick
            # to get an empty list into this array
            betaArr.append(np.array([np.double(x) for x in range(0)]))
            contactG2Arr.append(np.array([np.double(x) for x in range(0)]))
            pass


    return numWallContacts, betaArr, contactG2Arr

