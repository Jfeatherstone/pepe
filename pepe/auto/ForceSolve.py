import numpy as np
import os
import cv2
import time
import progressbar
import pickle
import inspect

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from IPython.display import clear_output

from pepe.preprocess import checkImageType, lightCorrectionDiff, circularMask
from pepe.analysis import initialForceSolve, forceOptimize, gSquared, g2ForceCalibration, singleParticleForceBalance
from pepe.tracking import houghCircle, convCircle
from pepe.simulate import genSyntheticResponse
from pepe.utils import preserveOrderArgsort, rectangularizeForceArrays, explicitKwargs
from pepe.visualize import genColors, visCircles, visForces, visContacts

# Decorator that allows us to identify which keyword arguments were explicitly
# passed to the function, and which were left as default values. See beginning
# of method code for more information/motivation.
@explicitKwargs()
def forceSolve(imageDirectory, guessRadius=0.0, fSigma=0.0, pxPerMeter=0.0, brightfield=True, contactPadding=15, g2MaskPadding=2, contactMaskRadius=30, lightCorrectionImage=None, lightCorrectionHorizontalMask=None, lightCorrectionVerticalMask=None, g2CalibrationImage=None, g2CalibrationCutoffFactor=.9, maskImage=None, peBlurKernel=3, imageExtension='bmp', requireForceBalance=False, imageStartIndex=None, imageEndIndex=None, carryOverAlpha=True, carryOverForce=True, showProgressBar=True, circleDetectionMethod='convolution', circleTrackingKwargs={}, circleTrackingChannel=0, maxBetaDisplacement=.5, photoelasticChannel=1, forceNoiseWidth=.03, optimizationKwargs={}, debug=False, saveMovie=False, outputRootFolder='./', inputSettingsFile=None, pickleArrays=True, genFitReport=True, outputExtension=''):
    """
    Complete pipeline to solve for forces and particle positions for all image files
    in a directory. Results will be returned and potentially written to various files.
    See `Returns` section for more information

    Expects all particles to be the same (or very similar) sizes. This assumption is made
    by the calculation of the gradient squared calibration value, which is computed just
    once using the guess of the radii. This should not be a problem if the radii are only
    slightly varied (~10 pixels or something) but any more than that and errors will begin
    to accumulate.

    This method has **a lot** of arguments; it is intended to be used once reasonable
    values for all of these have already been found. While the `debug` option for this function
    is very helpful, it is recommended to utilize the various notebooks/examples to find good
    choices for parameters first.

    The output readme file can also serve as a cache of the parameter values/settings, which
    can be passed back to this method using the `inputSettingsFile` argument.

    Parameters
    ----------

    imageDirectory : str
        An absolute or relative path to the directory containing the images that are
        to be analyzed. The names of the images need not follow any particular naming
        scheme, but they should be such that sorted the list alphabetically will give
        the images in their proper order.

    guessRadius : float
        The radius of the particles to be detected, in pixels. As of now, this value will
        be taken as the particle radius, but future versions may be able to vary this to
        find the optimal value.

        Currently no support for particles of different sizes.

    fSigma : float
        Stress optic coefficient, relating to material thickness, wavelength of light and
        other material property (C).

    pxPerMeter : float
        The number of pixels per meter in the images. Depends on the camera, lens, and 
        zoom settings used to capture the images.

    brightfield : bool
        Whether the images are captured using a brightfield polariscope (`True`) or
        a darkfield polariscope (`False`).

    contactPadding : int
        Maximum distance (in pixels) between a particle's edge and the wall or the edges of two
        particles that will still be considered a potential force-bearing contact.

    g2MaskPadding : int or float
        Number of pixels to ignore at the edge of each particle when calculating the average G^2.
        If float value < 1 is passed, gradient mask radius will be taken as that percent of the full
        particle radius. A value of 0 means no padding is included.

    contactMaskRadius : float
        The radius of the circular mask that will be constructed around each contact to estimate
        the magnitude of the force using the gradient squared in that region.

    lightCorrectionImage : str or np.ndarray[H,W[,C]]
        The path to an image (or an array representing an image) that contains no particles, and
        can be used to correct for any light gradients present in the actual data.

    lightCorrectionHorizontalMask : str or np.ndarray[H,W[,C]]
        A mask array, containing values of `0` or `1`, with the latter representing areas over which
        the horizontal light correction should be calculated. Can also be a path to an image.

    lightCorrectionVerticalMask : str or np.ndarray[H,W[,C]]
        A mask array, containing values of `0` or `1`, with the latter representing areas over which
        the vertical light correction should be calculated. Can also be a path to an image.

    g2CalibrationImage : str or np.ndarray[H,W,C]
        An image with a single particle (or at least one particle) that has no force acting on it.
        Used to determine the base level of gradient squared (due to noise) for a free particle. Can
        also be a path to an image.

    g2CalibrationCutoffFactor : float
        The factor that is multipled by the mean gradient squared value of the particles in the 
        calibration image. Any particle that has an average gradient squared value below the 
        calibration value multipled by this factor will be assumed to have no forces acting on it.

    maskImage : str or np.ndarray[H,W,C]
        A mask array, containing values of `0` or `1`, with the latter representing the regions of
        importance for the image. Used in detecting particles, generating initial guesses, and
        calculating error during non-linear optimization. can also be a path to an image.

    peBlurKernel : int
        The kernel size that will be used for bluring the photoelastic channel of each image, to
        reduce noise. Should be an odd integer.

    imageExtension : str
        The extension of the image files that will be read in from `imageDirectory`. Should not include
        the '.' before the extension.

    requireForceBalance : bool
        Whether to impose particle-wise force balance at each step (`True`) or to take the results of
        the optimization process as they are (`False`).

        Currently WIP, and does not do anything.

    imageStartIndex : int or None
        The index of which image to start at when analyzing the files in `imageDirectory`. Value
        of `None` will start at the first (alphabetically sorted) image.

    imageEndIndex : int or None
        The index of which image to end at when analyzing the files in `imageDirectory`. Value
        of `None` will end at the last (alphabetically sorted) image.

    showProgressBar : bool
        Whether to show a progress bar throughout the analysis (`True`) or not (`False`). Uses
        `progressbar2` library.

    circleDetectionMethod : ['convolution' or 'hough']
        Whether to use the convolution or hough circle detection method to identify particles.

        See `pepe.tracking.convCircle()` and `pepe.tracking.houghCircle()` for more information.

    circleTrackingKwargs : **kwargs
        Keyword arguments to be passed to the selected circle tracking function.

        See `pepe.tracking.convCircle()` and `pepe.tracking.houghCircle()` for more information.

    circleTrackingChannel : int
        The channel of the image that will be used to track the particles. `0` for red, `1` for
        green, and `2` for blue.

    maxBetaDisplacement : float
        The maximum distance (angle) that a force can move between frames and still be identified
        as the same force. If a force moves more than this value, it will still be recorded as a force,
        but will be considered a new and independent force from any of the ones in the previous frame.

    photoelasticChannel : int
        The channel of the image that will be used to gauge the photoelastic response. `0` for red, `1` for
        green, and `2` for blue.

    forceNoiseWidth : float
        The width of the gaussian distribution (centered at 0) that noise is sampled from to add to the
        force magnitudes from the previous frame. This is done to avoid getting stuck in a local
        minimum for too long (adds some Monte-Carlo-esque behavior to the solving).

    optimizationKwargs : **kwargs
        Keyword arguments to be passed to the optimization process.

        For more information, see `pepe.analysis.forceOptimize()`.

    debug : bool
        Whether to print progress updates for each frame to the screen (`True`) or not (`False`).

    saveMovie : bool
        Whether to save a compiled gif of the reconstructed forces at each frame at the end (`True`)
        or not (`False`).

    outputRootFolder : str
        The location where the output folder (potentially containg the movie, pickle files, readme, etc.)
        will be created. Output folder itself will be named after the `imageDirectory`, with '_Synthetic'
        appended to the end.

    pickleArrays : bool
        Whether to save the forces, betas, alphas, centers, and radii as pickle files (`True`) or not (`False`).
        Files will be located in the output folder (see `outputRootFolder`).

    inputSettingsFile : str
        Path to a readme file containg parameters for the solving process, likely generated from
        a previous iteration of the program. Explicitly passed arguments will override those that
        are included in the settings file.

        Currently WIP and does not do anything.

    genFitReport : bool
        Whether or not to generate a fit report of the results, including errors per frame,
        examinations of all particles/forces, and settings, compiled in a latex pdf.

        Will generate both the compiled file 'FitReport.pdf' and the source directory
        'FitReport_src/'.
    
    Returns
    -------

    rectForceArr : list[P](np.ndarray[F,T])
        A list of arrays representing the force magnitude for each force on each particle.

    rectAlphaArr : list[P](np.ndarray[F,T])
        A list of arrays representing the alpha angle for each force on each particle.

    rectBetaArr : list[P](np.ndarray[F,T])
        A list of arrays representing the beta angle for force on each particle.

    rectCenterArr : np.ndarray[P,T,2]
        Particle centers for each timestep. Elements take on a value of `[np.nan, np.nan]`
        if the particle does not exist for a given timestep.

    rectRadiusArr : np.ndarray[P,T]
        Particle radii for each timestep. Elements take on a value of `np.nan` if the particle
        does not exist for a given timestep.

    
    Depending on kwarg values, several quantities may be written into the output folder,
    which will be located in `outputRootFolder` and named according to `imageDirectory`,
    with '_Synthetic' appended.

    """
   
    overallStartTime = time.perf_counter()

    # For the sake of saving the options to a readme file (and potentially)
    # reading them back out, it is easiest to keep all of the settings in a
    # dictionary

    # We have 3 layers of precedence for reading in settings:
    # 1. Explicitly passed kwarg
    # 2. Read in from settings file
    # 3. Default value of a kwarg
    # So we assign the elements of our settings dict in opposite order

    # 3. All of the default values
    settings = {"imageDirectory": os.path.abspath(imageDirectory) + '/', # Convert to absolute path
                "imageExtension": imageExtension,
                "imageEndIndex": imageEndIndex,
                "imageStartIndex": imageStartIndex,
                "carryOverAlpha": carryOverAlpha,
                "carryOverForce": carryOverForce,
                "showProgressBar": showProgressBar,
                "lightCorrectionImage": lightCorrectionImage,
                "lightCorrectionVerticalMask": lightCorrectionVerticalMask,
                "lightCorrectionHorizontalMask": lightCorrectionHorizontalMask,
                "g2CalibrationImage": g2CalibrationImage,
                "g2CalibrationCutoffFactor": g2CalibrationCutoffFactor,
                "maskImage": maskImage,
                "circleDetectionMethod": circleDetectionMethod,
                "guessRadius": guessRadius,
                "fSigma": fSigma,
                "pxPerMeter": pxPerMeter,
                "brightfield": brightfield,
                "contactPadding": contactPadding,
                "g2MaskPadding": g2MaskPadding,
                "contactMaskRadius": contactMaskRadius,
                "peBlurKernel": peBlurKernel,
                "requireForceBalance": requireForceBalance,
                "circleTrackingChannel": circleTrackingChannel,
                "photoelasticChannel": photoelasticChannel,
                "maxBetaDisplacement": maxBetaDisplacement,
                "forceNoiseWidth": forceNoiseWidth,
                "saveMovie": saveMovie,
                "pickleArrays": pickleArrays,
                "outputRootFolder": outputRootFolder,
                "outputExtension": outputExtension,
                "genFitReport": genFitReport,
                "debug": debug}

    # 2. Anything read in from a settings file
    # Note that it works to our advantage that we already have values for most entries,
    # since the settings file doesn't include type information, so we need the old
    # values to cast properly. 1. is actually contained in here as well, because
    # we can just check to see if that variable was explicitly passed before overwriting it.

    # 1. The kwargs that are explicitly passed
    # This one is a little tricky, because there isn't a super great way by default
    # to differentiate whether a kwarg is explicitly passed or is its default value
    # (without just keeping a list of default values). I also don't want to
    # replace the entire function signature with (*args, **kwargs) because then the
    # documentation would not be as good (I think). So the solution here is to define
    # a decorator that has the (*args, **kwargs) signature, and to create an attribute
    # of this method that is a list of the kwargs that are explicitly passed to the
    # decorator. See `pepe.utils.explicitKwargs()` for more info.
    if inputSettingsFile is not None:
        if os.exists(inputSettingsFile):
            fileObj = open(inputSettingsFile, 'r')
            for line in fileObj:
                # Check each line and see if it looks like a dictionary value
                split = line.split(':')
                if len(split) == 2 and split[0] in settings.keys() and not split[0] in forceSolve.explicit_kwargs:
                    # Cast to the type of the value already in the dict
                    settings[split[0].strip()] = type(settings[split[0].strip()])(split[1].strip())
        else:
            print(f'Warning: provided settings file does not exist! Attempting to run regardless...')

    # While the following variables all have a default value of 0, they cannot actually
    # be left as this value. The reason they have a default value is so that if these
    # values are indicated by a settings file, we don't want to have to enter them again.
    # So here, we make sure we have values for them all, either explicitly passed or read in.
    requiredVars = ["guessRadius", "fSigma", "pxPerMeter"]
    for r in requiredVars:
        assert settings[r] != 0, f'Error: {r} value not supplied explicitly or implicitly!'

    # Now carry over the kwargs that are sent to the optimization procedure into that
    # dictionary. We can find the names of arguments by using the `inspect` library
    possibleOptimKwargs = list(inspect.signature(forceOptimize).parameters.keys())
    for pkw in possibleOptimKwargs:
        if pkw in settings.keys():
            optimizationKwargs[pkw] = settings[pkw]

    # We want to do the same thing for the circle tracking function, but we don't
    # yet know which circle tracking function we are using yet, so we'll carry
    # that over a bit later.

    # TODO: Make this more robust
    centerColors = genColors(10)

    # Find all images in the directory
    imageFiles = os.listdir(settings["imageDirectory"])

    # This goes before the sorting/extension filtering so we can get more specific
    # error messages (and we have another one of these below)
    if len(imageFiles) < 1:
        print(f'Error: directory {imageDirectory} contains no files!')
        return None

    imageFiles = np.sort([img for img in imageFiles if img[-len(settings["imageExtension"]):] == settings["imageExtension"]])

    # We have to do the end index first, so it doesn't mess up the start one
    if settings["imageEndIndex"] is not None:
        imageFiles = imageFiles[:min(settings["imageEndIndex"], len(imageFiles))]

    if settings["imageStartIndex"] is not None:
        imageFiles = imageFiles[max(settings["imageStartIndex"], 0):]

    # Make sure we still have some proper images
    if len(imageFiles) < 1:
        print(f'Error: directory \'{settings["imageDirectory"]}\' contains no files with extension \'{settings["imageExtension"]}\'!')
        return None

    if settings["showProgressBar"]:
        bar = progressbar.ProgressBar(max_value=len(imageFiles))

    # This will calculation the light correction across the images 
    if settings["lightCorrectionImage"] is not None:
        cImageProper = checkImageType(settings["lightCorrectionImage"])
        vMask = checkImageType(settings["lightCorrectionVerticalMask"])
        hMask = checkImageType(settings["lightCorrectionHorizontalMask"])

        if vMask.ndim == 3:
            vMask = vMask[:,:,0]
        if hMask.ndim == 3:
            hMask = hMask[:,:,0]
        lightCorrection = lightCorrectionDiff(cImageProper, vMask, hMask)
    else:
        # It probably isn't great hygiene to have this variableflip between a single
        # value and an array, but you can always add a scalar to a numpy array, so
        # this is the easiest way (since we haven't loaded any images yet)
        lightCorrection = 0

    # Load up the mask image, which will be used to remove parts of the images
    # that we don't care about, and also potentially indicate which particles
    # are close to the boundary.
    if settings["maskImage"] is not None:
        maskArr = checkImageType(settings["maskImage"])
        ignoreBoundary = False

    else:
        # Same deal as above: scalar multiplication functions exactly how we want
        # in the case that we don't have a mask, so it's just easier to do this.
        maskArr = 1
        ignoreBoundary = True

    # Which method we will be using to detect circles
    if settings["circleDetectionMethod"] == 'convolution':
        circFunc = convCircle
    elif settings["circleDetectionMethod"] == 'hough':
        circFunc = houghCircle
    else:
        print(f'Error: circle detection option \'{settings["circleDetectionMethod"]}\' not recognized!')
        return None

    # Now that we have a circle tracking function, we can carry over any possible kwargs
    possibleCircleKwargs = list(inspect.signature(circFunc).parameters.keys())
    for pkw in possibleCircleKwargs:
        if pkw in settings.keys():
            circleTrackingKwargs[pkw] = settings[pkw]

    # Calculate the lowest g2 value that we care about, so we can throw everything
    # that is below that away when solving (optional)
    checkMinG2 = False
    if settings["g2CalibrationImage"] is not None:
        g2CalImage = checkImageType(settings["g2CalibrationImage"])

        g2CalPEImage = cv2.blur((g2CalImage[:,:,settings["photoelasticChannel"]] + lightCorrection).astype(np.float64) / 255, (settings["peBlurKernel"],settings["peBlurKernel"]))
        # Locate particles
        centers, radii = circFunc(g2CalImage[:,:,settings["circleTrackingChannel"]] * maskArr[:,:,0], settings["guessRadius"], **circleTrackingKwargs)
        # There should only be 1 particle in the calibration image
        if len(centers) < 0:
            print(f'Warning: Gradient-squared calibration image does not contain any particles! Ignoring...')
        else:
            particleMask = circularMask(g2CalPEImage.shape, centers[0], radii[0])[:,:,0]
            gSqr = gSquared(g2CalPEImage)
            minParticleG2 = np.sum(gSqr * particleMask) / np.sum(particleMask) * settings["g2CalibrationCutoffFactor"]
            checkMinG2 = True


    # The arrays that we will be building for each timestep. It is better to just
    # use an untyped list since the arrays are all triangular and whatnot.
    centersArr = []
    radiiArr = []

    forceArr = []
    betaArr = []
    alphaArr = []

    imageArr = []
    errorArr = []

    # For keeping track of time (though will only be display if debug=True)
    trackingTimes = np.zeros(len(imageFiles))
    initialGuessTimes = np.zeros(len(imageFiles))
    optimizationTimes = np.zeros(len(imageFiles))
    miscTimes = np.zeros(len(imageFiles))
    totalFailedParticles = 0

    errorMsgs = []

    # Calculate the gradient-squared-to-force calibration value
    g2Cal = g2ForceCalibration(fSigma, guessRadius, pxPerMeter)

    # The big loop that iterates over every image
    for i in range(len(imageFiles)):

        image = checkImageType(settings["imageDirectory"] + imageFiles[i])
        # Convert to floats on the domain [0,1], so we can compare to the output of 
        # genSyntheticResponse()
        peImage = cv2.blur((image[:,:,settings["photoelasticChannel"]] + lightCorrection).astype(np.float64) / 255, (settings["peBlurKernel"],settings["peBlurKernel"]))

        # -------------
        # Track circles
        # -------------
        start = time.perf_counter()
        centers, radii = circFunc(image[:,:,settings["circleTrackingChannel"]] * maskArr[:,:,0], settings["guessRadius"], **circleTrackingKwargs)

        # We do some indexing using the centers/radii, so it is helpful
        # to have them as an integer type
        centers = centers.astype(np.int64)
        radii = radii.astype(np.int64)

        # We want to keep the order of particles constant, so we make sure
        # that they are (to whatever extent possible) in the same order
        # as the previous frame. This involves finding the closest neighbor
        # from the previous frame.
        if len(centersArr) > 0:
            centerOrder = preserveOrderArgsort(centersArr[-1], centers, padMissingValues=False)
            centers = centers[centerOrder]
            radii = radii[centerOrder]

        trackingTimes[i] = time.perf_counter() - start

        # ----------------------
        # Generate initial guess
        # ----------------------
        # We rerun the initial guess mostly when we can't carry over the values from
        # the previous step
        if len(centersArr) == 0:
            forceGuessArr, alphaGuessArr, betaGuessArr = initialForceSolve(peImage,
                                                        centers, radii, settings["fSigma"], settings["pxPerMeter"],
                                                        settings["contactPadding"], settings["g2MaskPadding"],
                                                        contactMaskRadius=settings["contactMaskRadius"],
                                                        boundaryMask=maskArr, ignoreBoundary=ignoreBoundary, g2Cal=g2Cal)

        else:
        #elif len(centers) != len(centersArr[-1]):
            # If we have added/lost particles, we want to carry over the previous values where
            # possible, and otherwise take the results of initialForceSolve
            forceGuessArr, alphaGuessArr, betaGuessArr = initialForceSolve(peImage,
                                                        centers, radii, settings["fSigma"], settings["pxPerMeter"],
                                                        settings["contactPadding"], settings["g2MaskPadding"],
                                                        contactMaskRadius=settings["contactMaskRadius"],
                                                        boundaryMask=maskArr, ignoreBoundary=ignoreBoundary, g2Cal=g2Cal)

            # This variable will certainly exist (even though it was defined in an if statement
            for j in range(len(betaArr[-1])):
                if len(betaArr[-1][j]) > 0:
                    # Centers should already be in the same order, so now we just have to order
                    # the forces in case one was added/removed
                    forceOrder = preserveOrderArgsort(betaGuessArr[j], betaArr[-1][j], padMissingValues=True, maxDistance=settings["maxBetaDisplacement"])
                    for k in range(len(forceGuessArr[j])):
                        if forceOrder[k] is not None:
                            if settings["carryOverForce"]:
                                forceGuessArr[j][k] = forceArr[-1][j][forceOrder[k]]
                            if settings["carryOverAlpha"]:
                                alphaGuessArr[j][k] = alphaArr[-1][j][forceOrder[k]]


            # In this case, we want to add a small randomly generated contribution
            # so that the algorithm doesn't get stuck in some incorrect loop and so that it
            # explores a little more of the parameter space to find a nice minimum at each step
            forceGuessArr = [np.abs(np.array(f) + np.random.normal(0, settings["forceNoiseWidth"], size=len(f))) for f in forceGuessArr]


        initialGuessTimes[i] = time.perf_counter() - trackingTimes[i] - start

        # -------------------------------
        # Optimize each particle's forces
        # -------------------------------
        optimizedForceArr = []
        optimizedBetaArr = []
        optimizedAlphaArr = []
        failed = [False for i in range(len(centers))]

        # Drop forces on any particles whose g2 is lower than the min value
        skipParticles = [False for i in range(len(centers))]
        if checkMinG2:
            gSqr = gSquared(peImage)
            for j in range(len(centers)):
                cMask = circularMask(peImage.shape, centers[j], radii[j])[:,:,0]
                avgG2 = np.sum(gSqr * cMask) / np.sum(cMask)
                skipParticles[j] = avgG2 < minParticleG2

        for j in range(len(centers)):
            if not skipParticles[j]:
                try:
                    # We don't need to pass fSigma, pxPerMeter, or brightfield to the method
                    # because they will get added to optimizationKwargs automatically.
                    optForceArr, optBetaArr, optAlphaArr, res = forceOptimize(forceGuessArr[j], betaGuessArr[j], alphaGuessArr[j], radii[j], centers[j], peImage,
                                                                              #settings["fSigma"], settings["pxPerMeter"], settings["brightfield"],
                                                                              **optimizationKwargs)
                    optimizedForceArr.append(optForceArr)
                    optimizedBetaArr.append(optBetaArr)
                    optimizedAlphaArr.append(optAlphaArr)
                except Exception as ex:
                    print(ex)
                    errorMsgs.append(f'File {imageFiles[i]}: ' + str(ex) + '\n')
                    failed[j] = True
                    totalFailedParticles += 1
                    # Append empty lists (ie say there are no forces) 
                    #optimizedForceArr.append(forceGuessArr[j])
                    #optimizedBetaArr.append(betaGuessArr[j])
                    #optimizedAlphaArr.append(alphaGuessArr[j])
                    optimizedForceArr.append([])
                    optimizedBetaArr.append([])
                    optimizedAlphaArr.append([])
            else:
                optimizedForceArr.append([])
                optimizedBetaArr.append([])
                optimizedAlphaArr.append([])

        # If necessary, impose force balance on all particles
        if requireForceBalance:
            for j in range(len(centers)):
                optimizedForceArr[j], optimizedAlphaArr[j] = singleParticleForceBalance(optimizedForceArr[j], optimizedAlphaArr[j], optimizedBetaArr[j])

        optimizationTimes[i] = time.perf_counter() - initialGuessTimes[i] - trackingTimes[i] - start

        # Save all of our values
        forceArr.append(optimizedForceArr)
        betaArr.append(optimizedBetaArr)
        alphaArr.append(optimizedAlphaArr)
        centersArr.append(centers)
        radiiArr.append(radii)

        if settings["debug"] or settings["saveMovie"] or settings["genFitReport"]:
            estimatedPhotoelasticChannel = np.zeros_like(peImage, dtype=np.float64)    
            for j in range(len(centers)):
                estimatedPhotoelasticChannel += genSyntheticResponse(np.array(forceGuessArr[j]),
                                                                     np.array(alphaGuessArr[j]),
                                                                     np.array(betaGuessArr[j]),
                                                                     settings["fSigma"], radii[j],
                                                                     settings["pxPerMeter"], settings["brightfield"], imageSize=peImage.shape,
                                                                     center=centers[j])
            
            

            optimizedPhotoelasticChannel = np.zeros(peImage.shape)
            for j in range(len(centers)):
                optimizedPhotoelasticChannel += genSyntheticResponse(np.array(optimizedForceArr[j]),
                                                                     np.array(optimizedAlphaArr[j]),
                                                                     np.array(optimizedBetaArr[j]),
                                                                     settings["fSigma"], radii[j],
                                                                     settings["pxPerMeter"], settings["brightfield"], imageSize=peImage.shape,
                                                                     center=centers[j])
            

            # Just simple mean-squared error
            errorArr.append(np.sqrt(np.sum((optimizedPhotoelasticChannel - peImage)**2)))

            imgArr = np.zeros((*optimizedPhotoelasticChannel.shape, 3))
            
            img = Image.fromarray(optimizedPhotoelasticChannel*255)
            img = img.convert('RGB')
            drawObj = ImageDraw.Draw(img)
            for j in range(len(centers)):
                leftUpPoint = (centers[j][1]-radii[j], centers[j][0]-radii[j])
                rightDownPoint = (centers[j][1]+radii[j], centers[j][0]+radii[j])
                twoPointList = [leftUpPoint, rightDownPoint]
                color =  '#FF0000' if failed[j] else '#00AAAA'
                drawObj.ellipse(twoPointList, outline=color, fill=None, width=3)

        if debug:

            clear_output(wait=True)
            fig, ax = plt.subplots(1, 3, figsize=(12,4))
            
            ax[0].imshow(maskArr * image)
            ax[0].set_title('Tracked Particles')
            for j in range(len(centers)):
                c = plt.Circle(centers[j][::-1], radii[j], label='Detected particles', color='teal', fill=False, linewidth=1)
                ax[0].add_artist(c)
                # Now add contacts
                for k in range(len(betaGuessArr[j])):
                    contactPoint = centers[j] + radii[j] * np.array([np.cos(betaGuessArr[j][k]), np.sin(betaGuessArr[j][k])])
                    cc = plt.Circle(contactPoint[::-1], 12, color='red', fill=False, linewidth=1)
                    ax[1].add_artist(cc)
                    
                # Now plot past center positions
                for k in range(len(centersArr)):
                    if len(centersArr[k]) >= j:
                        cc = plt.Circle(centersArr[k][j][::-1], 5, color=centerColors[j], fill=True)
                        ax[0].add_artist(cc)
                    

            ax[1].imshow(estimatedPhotoelasticChannel)
            ax[1].set_title('Initial Guess for Optimizer\n(known forces)')
            
            
            ax[2].imshow(img)
            ax[2].set_title('Optimized Forces\n(known forces)')
            
            fig.suptitle(imageFiles[i])
            fig.tight_layout()
            plt.show()

        if saveMovie:
            imageArr.append(img)

        miscTimes[i] = time.perf_counter() - optimizationTimes[i] - initialGuessTimes[i] - trackingTimes[i] - start 

        if debug: 
            print(f'Took {time.perf_counter() - start:.5}s to solve frame:')
            print(f'{5*" "}Tracking:         {trackingTimes[i]:.3}s')
            print(f'{5*" "}Initial guess:    {initialGuessTimes[i]:.3}s')
            print(f'{5*" "}Optimization:     {optimizationTimes[i]:.3}s')
            print(f'{5*" "}Misc. processes:  {miscTimes[i]:.3}s')

        if showProgressBar:
            bar.update(i)
   
    
    if showProgressBar:
        bar.update(len(imageFiles))

    # Reuse the name of the folder the images come from as a part of
    # the output folder name
    # [-2] element for something of form 'path/to/final/folder/' will be 'folder'
    # If we are missing the final /, you have to take just the [-1] element
    if imageDirectory[-1] == '/':
        outputFolderPath = outputRootFolder + imageDirectory.split('/')[-2] + f'_Synthetic{settings["outputExtension"]}/'
    else:
        outputFolderPath = outputRootFolder + imageDirectory.split('/')[-1] + f'_Synthetic{settings["outputExtension"]}/'

    if not os.path.exists(outputFolderPath):
        os.mkdir(outputFolderPath)

    if saveMovie:
        imageArr[0].save(outputFolderPath + 'Synthetic.gif', save_all=True, append_images=imageArr[1:], duration=30, optimize=False, loop=1)

    # Write a readme file that contains all of the parameters that the solving used
    lines = ['#####################\n',
             '#    README FILE    #\n',
             '#####################\n']

    lines += [f'Generated: {time.ctime()}\n\n']
    lines += ['Note: this file was autogenerated by the `pepe.auto.forceSolve()` function\n',
              '      and it is not recommended to be manually edited. To reuse the settings\n',
              '      and parameters that were used here, the path of this file\n',
             f'      (\'{outputFolderPath}readme.txt\') \n',
              '      can be passed via the \'settingsFile\' keyword argument of `pepe.auto.forceSolve()`.\n',
              '      In this case, explictly passed arguments will override the values in the settings file.\n']

    lines += ['\n## Runtime Information\n',
              f'Total runtime: {time.perf_counter() - overallStartTime:.6}s\n',
              f'Mean tracking time: {np.mean(trackingTimes):.4}s\n',
              f'Mean guess generation time: {np.mean(initialGuessTimes):.4}s\n',
              f'Mean optimization time: {np.mean(optimizationTimes):.4}s\n',
              f'Mean misc. time: {np.mean(miscTimes):.4}s\n',
              f'Number of failed particles: {totalFailedParticles}\n']

    settings.update(circleTrackingKwargs)
    settings.update(optimizationKwargs)

    lines += ['\n## Settings\n']
    for k,v in settings.items():
        lines += [f'{k}: {v}\n']
   
    lines += ['\n## Errors\n']
    if len(errorMsgs) > 0:
        lines += errorMsgs
    else:
        lines += ['None :)']

    with open(outputFolderPath + 'readme.txt', 'w') as readmeFile:
        readmeFile.writelines(lines)

    # Restructure the arrays to make them more friendly, and to track forces/particles across timesteps
    rectForceArr, rectAlphaArr, rectBetaArr, rectCenterArr, rectRadiusArr = rectangularizeForceArrays(forceArr, alphaArr, betaArr, centersArr, radiiArr)

    # Save the arrays to pickle files (optional)
    if settings["pickleArrays"]:
        with open(outputFolderPath + 'forces.pickle', 'wb') as f:
            pickle.dump(rectForceArr, f)

        with open(outputFolderPath + 'alphas.pickle', 'wb') as f:
            pickle.dump(rectAlphaArr, f)

        with open(outputFolderPath + 'betas.pickle', 'wb') as f:
            pickle.dump(rectBetaArr, f)

        with open(outputFolderPath + 'centers.pickle', 'wb') as f:
            pickle.dump(rectCenterArr, f)

        with open(outputFolderPath + 'radii.pickle', 'wb') as f:
            pickle.dump(rectRadiusArr, f)

    # Generate a fit report (optional)
    # This include informtaion about the error for each frame, all of the forces/alphas/betas/
    # centers/radii for each particle at each timestep, and all settings in a nicely compiled
    # (via latex) pdf.
    if settings["genFitReport"]:
        # Make the source directory
        if not os.path.exists(outputFolderPath + 'FitReport_src'):
            os.mkdir(outputFolderPath + 'FitReport_src')

        # First, generate a plot of the error
        fig, ax = plt.subplots()

        ax.plot(errorArr)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Mean-squared error')
        ax.set_title('Difference Between Optimized Result and Real Image')

        fig.savefig(outputFolderPath + 'FitReport_src/error.pdf')
        plt.close(fig)

        # Draw all of the circles, with their labeled numbers
        fig, ax = plt.subplots(1, 2, figsize=(8,3))
        # First timestep
        visCircles([rectCenterArr[i][0] for i in range(len(rectCenterArr))], [rectRadiusArr[i][0] for i in range(len(rectRadiusArr))],
                   ax=ax[0], annotations=np.arange(len(rectCenterArr)), setBounds=True) 
        # Last timestep
        visCircles([rectCenterArr[i][-1] for i in range(len(rectCenterArr))], [rectRadiusArr[i][-1] for i in range(len(rectRadiusArr))],
                   ax=ax[1], annotations=np.arange(len(rectCenterArr)), setBounds=True) 

        for i in range(2):
            ax[i].set_xlabel('X [px]')
            ax[i].set_ylabel('Y [px]')
            ax[i].invert_yaxis()

        ax[0].set_title('First Frame')
        ax[1].set_title('Last Frame')

        fig.savefig(outputFolderPath + 'FitReport_src/particle_identities.pdf')
        plt.close(fig)

        # Next, draw the forces/betas/alphas/centers for each particle
        # through time
        for i in range(len(rectForceArr)):
            fig, ax = visForces(rectForceArr[i], rectAlphaArr[i], rectBetaArr[i], rectCenterArr[i])
            fig.suptitle(f'Particle {i}')
            fig.savefig(outputFolderPath + f'FitReport_src/particle_{i}_forces.pdf')
            plt.close(fig)


    # Create gifs of the contacts
    forceColors = genColors(len(rectBetaArr))
    # The list comprehension is to make sure that we index a particle that actually has forces acting
    # on it.
    tSteps = len([b for b in betaArr if len(b) > 0][0])
    contactPointImages = [None for i in range(tSteps)]
    contactAngleImages = [None for i in range(tSteps)]

    for i in range(tSteps):
        # Have to do this, because the settings variable could be None 
        startI = settings["imageStartIndex"] if settings["imageStartIndex"] is not None else 0

        # First, just the contact points
        fig, ax = plt.subplots()
        visCircles([rectCenterArr[p][i] for p in range(len(rectCenterArr))], [rectRadiusArr[p][i] for p in range(len(rectRadiusArr))], ax=ax)

        for particleIndex in range(len(rectBetaArr)):
            visContacts(rectCenterArr[particleIndex][i], rectRadiusArr[particleIndex][i],
                        rectBetaArr[particleIndex][:,i], ax=ax, forceColors=forceColors[particleIndex])

        ax.set_xlim([0, 1280])
        ax.set_ylim([0, 1024])
        ax.set_aspect('equal')
        ax.set_title(f'Frame {i + startI}')
        ax.invert_yaxis()

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        contactPointImages[i] = Image.frombytes('RGB', canvas.get_width_height(), 
                                    canvas.tostring_rgb())

        plt.close(fig)
   
        # Now the one with angles
        fig, ax = plt.subplots()
        visCircles([rectCenterArr[p][i] for p in range(len(rectCenterArr))], [rectRadiusArr[p][i] for p in range(len(rectRadiusArr))], ax=ax)

        for particleIndex in range(len(rectBetaArr)):
            visContacts(rectCenterArr[particleIndex][i], rectRadiusArr[particleIndex][i],
                        rectBetaArr[particleIndex][:,i], ax=ax, forceColors=forceColors[particleIndex], alphaArr=rectAlphaArr[particleIndex][:,i])
                                            
        ax.set_xlim([0, 1280])
        ax.set_ylim([0, 1024])
        ax.set_aspect('equal')
        ax.set_title(f'Frame {i + startI}')
        ax.invert_yaxis()

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        contactAngleImages[i] = Image.frombytes('RGB', canvas.get_width_height(), 
                                    canvas.tostring_rgb())

        plt.close(fig)

    contactPointImages[0].save(outputFolderPath + 'FitReport_src/contact_points.gif', save_all=True,
                               append_images=contactPointImages[1:], duration=25, optimize=True, loop=True)

    contactAngleImages[0].save(outputFolderPath + 'FitReport_src/contact_angles.gif', save_all=True,
                               append_images=contactAngleImages[1:], duration=25, optimize=True, loop=True)
    return rectForceArr, rectAlphaArr, rectBetaArr, rectCenterArr, rectRadiusArr
