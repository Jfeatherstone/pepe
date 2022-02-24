import numpy as np
import os
import cv2
import time
import progressbar
import pickle

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from IPython.display import clear_output

from pepe.preprocess import checkImageType, lightCorrectionDiff, circularMask
from pepe.analysis import initialForceSolve, forceOptimize, gSquared, g2ForceCalibration, singleParticleForceBalance
from pepe.tracking import houghCircle, convCircle
from pepe.simulate import genSyntheticResponse
from pepe.utils import genRandomColors, preserveOrderArgsort, rectangularizeForceArrays

def forceSolve(imageDirectory, guessRadius, fSigma, pxPerMeter, brightfield, contactPadding=15, g2MaskPadding=2, contactMaskRadius=30, lightCorrectionImage=None, lightCorrectionHorizontalMask=None, lightCorrectionVerticalMask=None, g2CalibrationImage=None, g2CalibrationCutoffFactor=.9, maskImage=None, peBlurKernel=3, imageExtension='bmp', requireForceBalance=False, imageStartIndex=None, imageEndIndex=None, showProgressBar=True, circleDetectionMethod='convolution', circleTrackingKwargs={}, circleTrackingChannel=0, maxBetaDisplacement=.5, photoelasticChannel=1, forceNoiseWidth=.03, optimizationKwargs={}, debug=False, saveMovie=False, outputRootFolder='./', inputSettingsFile=None, pickleArrays=True):
    """
    Complete pipeline to solve for forces and particle positions for all image files
    in a directory. Results will be saved to a text file and optionally compiled into
    images/movies of the system.

    Expects all particles to be the same (or very similar) sizes.

    This method has **a lot** of arguments; it is intended to be used once reasonable
    values for all of these have already been found. While the `debug` option for this function
    is very helpful, it is recommended to utilize the various notebooks/examples to find good
    choices for parameters first.
    """
   
    overallStartTime = time.perf_counter()

    # TODO: Possibly read in settings from a readme file
    if inputSettingsFile is not None:
        if os.exists(inputSettingsFile):
            pass
        else:
            print(f'Warning: provided settings file does not exist! Attempting to run regardless...')

    # For the sake of saving the options to a readme file (and potentially)
    # reading them back out, it is easiest to keep all of the settings in a
    # dictionary
    settings = {"imageDirectory": os.path.abspath(imageDirectory) + '/', # Convert to absolute path
                "imageExtension": imageExtension,
                "imageEndIndex": imageEndIndex,
                "imageStartIndex": imageStartIndex,
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
                "outputRootFolder": outputRootFolder}


    centerColors = genRandomColors(10)

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

    # TODO: make sure all settings exist

    # The arrays that we will be building for each timestep. It is better to just
    # use an untyped list since the arrays are all triangular and whatnot.
    centersArr = []
    radiiArr = []

    forceArr = []
    betaArr = []
    alphaArr = []

    imageArr = []

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
            for j in range(len(centers)):
                if len(betaArr[-1][j]) > 0:
                    # Centers should already be in the same order, so now we just have to order
                    # the forces in case one was added/removed
                    forceOrder = preserveOrderArgsort(betaGuessArr[j], betaArr[-1][j], padMissingValues=True, maxDistance=settings["maxBetaDisplacement"])
                    for k in range(len(forceGuessArr[j])):
                        if forceOrder[k] is not None:
                            forceGuessArr[j][k] = forceArr[-1][j][forceOrder[k]]
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
                    optForceArr, optBetaArr, optAlphaArr, res = forceOptimize(forceGuessArr[j], betaGuessArr[j], alphaGuessArr[j], radii[j], centers[j], peImage,
                                                                              settings["fSigma"], settings["pxPerMeter"], settings["brightfield"], **optimizationKwargs)
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

        if debug or saveMovie:
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
   
    # Reuse the name of the folder the images come from as a part of
    # the output folder name
    # [-2] element for something of form 'path/to/final/folder/' will be 'folder'
    # If we are missing the final /, you have to take just the [-1] element
    if imageDirectory[-1] == '/':
        outputFolderPath = outputRootFolder + imageDirectory.split('/')[-2] + '_Synthetic/'
    else:
        outputFolderPath = outputRootFolder + imageDirectory.split('/')[-1] + '_Synthetic/'

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
              '      In this case, explictly passed arguments will override the values in the settings file.\n\n']

    lines += ['## Runtime Information\n',
              f'Total runtime: {time.perf_counter() - overallStartTime:.6}s\n',
              f'Mean tracking time: {np.mean(trackingTimes):.4}s\n',
              f'Mean guess generation time: {np.mean(initialGuessTimes):.4}s\n',
              f'Mean optimization time: {np.mean(optimizationTimes):.4}s\n',
              f'Mean misc. time: {np.mean(miscTimes):.4}s\n',
              f'Number of failed particles: {totalFailedParticles}\n\n']

    settings.update(circleTrackingKwargs)
    settings.update(optimizationKwargs)

    lines += ['## Settings\n']
    for k,v in settings.items():
        lines += [f'{k}: {v}\n']
   
    lines += ['## Errors\n']
    lines += errorMsgs

    with open(outputFolderPath + 'readme.txt', 'w') as readmeFile:
        readmeFile.writelines(lines)

    # Restructure the arrays to make them more friendly, and to track forces/particles across timesteps
    rectForceArr, rectAlphaArr, rectBetaArr, rectCenterArr, rectRadiusArr = rectangularizeForceArrays(forceArr, alphaArr, betaArr, centersArr, radiiArr)

    if settings["pickleArrays"]:
        with open(outputFolderPath + 'forces.pickle', 'wb') as f:
            pickle.dump(rectForceArr, f)

        with open(outputFolderPath + 'betas.pickle', 'wb') as f:
            pickle.dump(rectAlphaArr, f)

        with open(outputFolderPath + 'alphas.pickle', 'wb') as f:
            pickle.dump(rectBetaArr, f)

        with open(outputFolderPath + 'centers.pickle', 'wb') as f:
            pickle.dump(rectCenterArr, f)

        with open(outputFolderPath + 'radii.pickle', 'wb') as f:
            pickle.dump(rectRadiusArr, f)

    return rectForceArr, rectAlphaArr, rectBetaArr, rectCenterArr, rectRadiusArr
