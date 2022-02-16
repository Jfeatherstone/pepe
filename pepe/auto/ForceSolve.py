import numpy as np
import os
import cv2
import time
import progressbar

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from IPython.display import clear_output

from pepe.preprocess import checkImageType
from pepe.analysis import initialForceSolve, forceOptimize, gSquared, g2ForceCalibration, singleParticleForceBalance
from pepe.tracking import houghCircle, convCircle
from pepe.simulate import genSyntheticResponse
from pepe.utils import genRandomColors, preserveOrderArgsort

def forceSolve(imageDirectory, guessRadius, fSigma, pxPerMeter, brightfield, contactPadding=15, g2MaskPadding=2, contactMaskRadius=30, correctionImage=None, lightCorrectionHorizontalMask=None, lightCorrectionVerticalMask=None, maskImage=None, peBlurKernel=3, imageExtension='bmp', requireForceBalance=False, imageStartIndex=None, imageEndIndex=None, showProgressBar=True, circleDetectionMethod='convolution', circleTrackingKwargs={}, circleTrackingChannel=0, photoelasticChannel=1, optimizationKwargs={}, debug=False, saveMovie=False, outputFolder='./'):
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
    centerColors = genRandomColors(10)

    # Find all images in the directory
    imageFiles = os.listdir(imageDirectory)

    # This goes before the sorting/extension filtering so we can get more specific
    # error messages (and we have another one of these below)
    if len(imageFiles) < 1:
        print(f'Error: directory {imageDirectory} contains no files!')
        return None

    imageFiles = np.sort([img for img in imageFiles if img[-len(imageExtension):] == imageExtension])

    # We have to do the end index first, so it doesn't mess up the start one
    if imageEndIndex is not None:
        imageFiles = imageFiles[:min(imageEndIndex, len(imageFiles))]

    if imageStartIndex is not None:
        imageFiles = imageFiles[max(imageStartIndex, 0):]

    # Make sure we still have some proper images
    if len(imageFiles) < 1:
        print(f'Error: directory \'{imageDirectory}\' contains no files with extension \'{imageExtension}\'!')
        return None

    if showProgressBar:
        bar = progressbar.ProgressBar(max_value=len(imageFiles))

    # This will calculation the light correction across the images 
    if correctionImage is not None:
        cImageProper = checkImageType(correctionImage)
        lightCorrection = lightCorrectionDiff(cImageProper, lightCorrectionVerticalMask, lightCorrectionHorizontalMask)
    else:
        # It probably isn't great hygiene to have this variableflip between a single
        # value and an array, but you can always add a scalar to a numpy array, so
        # this is the easiest way (since we haven't loaded any images yet)
        lightCorrection = 0

    # Load up the mask image, which will be used to remove parts of the images
    # that we don't care about, and also potentially indicate which particles
    # are close to the boundary.
    if maskImage is not None:
        maskArr = checkImageType(maskImage)
        ignoreBoundary = False

    else:
        # Same deal as above: scalar multiplication functions exactly how we want
        # in the case that we don't have a mask, so it's just easier to do this.
        maskArr = 1
        ignoreBoundary = True

    # Which method we will be using to detect circles
    if circleDetectionMethod == 'convolution':
        circFunc = convCircle
    elif circleDetectionMethod == 'hough':
        circFunc = houghCircle
    else:
        print(f'Error: circle detection option \'{circleDetectionMethod}\' not recognized!')
        return None

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

    # Calculate the gradient-squared-to-force calibration value
    g2Cal = g2ForceCalibration(fSigma, guessRadius, pxPerMeter)

    # The big loop that iterates over every image
    for i in range(len(imageFiles)):

        image = checkImageType(imageDirectory + imageFiles[i])
        # Convert to floats on the domain [0,1], so we can compare to the output of 
        # genSyntheticResponse()
        peImage = cv2.blur((image[:,:,photoelasticChannel] + lightCorrection).astype(np.float64) / 255, (peBlurKernel,peBlurKernel))

        # -------------
        # Track circles
        # -------------
        start = time.perf_counter()
        centers, radii = circFunc(image[:,:,circleTrackingChannel] * maskArr[:,:,0], guessRadius, **circleTrackingKwargs)

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
                                                        centers, radii, fSigma, pxPerMeter,
                                                        contactPadding, g2MaskPadding,
                                                        contactMaskRadius=contactMaskRadius,
                                                        boundaryMask=maskArr, ignoreBoundary=ignoreBoundary, g2Cal=g2Cal)

        else:
        #elif len(centers) != len(centersArr[-1]):
            # If we have added/lost particles, we want to carry over the previous values where
            # possible, and otherwise take the results of initialForceSolve
            # TODO: Make this behavior properly different from previous statement
            forceGuessArr, alphaGuessArr, betaGuessArr = initialForceSolve(peImage,
                                                        centers, radii, fSigma, pxPerMeter,
                                                        contactPadding, g2MaskPadding,
                                                        contactMaskRadius=contactMaskRadius,
                                                        boundaryMask=maskArr, ignoreBoundary=ignoreBoundary, g2Cal=g2Cal)

            # This variable will certainly exist (even though it was defined in an if statement
            for j in range(len(centers)):
                # Centers should already be in the same order, so now we just have to order
                # the forces in case one was added/removed
                forceOrder = preserveOrderArgsort(betaGuessArr[j], betaArr[-1][j], padMissingValues=True)
                for k in range(len(forceGuessArr[j])):
                    if forceOrder[k] is not None:
                        forceGuessArr[j][k] = forceArr[-1][j][forceOrder[k]]
                        alphaGuessArr[j][k] = alphaArr[-1][j][forceOrder[k]]


            # In this case, we want to add a small randomly generated contribution
            # so that the algorithm doesn't get stuck in some incorrect loop and so that it
            # explores a little more of the parameter space to find a nice minimum at each step
            forceGuessArr = [np.abs(np.array(f) + np.random.normal(0, .03, size=len(f))) for f in forceGuessArr]
        #else:
        #    # Note that we do want to update the beta values based on the current
        #    # contact network, so we still have to run the initial force solve here.
        #    # TODO: It would be a good idea to write a faster method that just recalculates
        #    # the beta values
        #    forceGuessArr, alphaGuessArr, betaGuessArr = initialForceSolve(peImage,
        #                                                centers, radii, fSigma, pxPerMeter,
        #                                                contactPadding, g2MaskPadding,
        #                                                contactMaskRadius=contactMaskRadius,
        #                                                boundaryMask=maskArr, ignoreBoundary=ignoreBoundary, g2Cal=g2Cal)
        #    forceGuessArr = forceArr[-1]
        #    alphaGuessArr = alphaArr[-1]
        #    #betaGuessArr = betaArr[-1]
        #
        #    # In this case, we want to add a small randomly generated contribution
        #    # so that the algorithm doesn't get stuck in some incorrect loop and so that it
        #    # explores a little more of the parameter space to find a nice minimum at each step
        #    forceGuessArr = [np.abs(np.array(f) + np.random.normal(0, .03, size=len(f))) for f in forceGuessArr]

        initialGuessTimes[i] = time.perf_counter() - trackingTimes[i] - start

        # -------------------------------
        # Optimize each particle's forces
        # -------------------------------
        optimizedForceArr = []
        optimizedBetaArr = []
        optimizedAlphaArr = []
        failed = [False for i in range(len(centers))]

        for j in range(len(centers)):
            try:
                optForceArr, optBetaArr, optAlphaArr, res = forceOptimize(forceGuessArr[j], betaGuessArr[j], alphaGuessArr[j], radii[j], centers[j], peImage,
                                                                          fSigma, pxPerMeter, brightfield, **optimizationKwargs)
                optimizedForceArr.append(optForceArr)
                optimizedBetaArr.append(optBetaArr)
                optimizedAlphaArr.append(optAlphaArr)
            except Exception as ex:
                print(ex)
                failed[j] = True
               
                optimizedForceArr.append(forceGuessArr[j])
                optimizedBetaArr.append(betaGuessArr[j])
                optimizedAlphaArr.append(alphaGuessArr[j])

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
                                                                     fSigma, radii[j],
                                                                     pxPerMeter, brightfield, imageSize=peImage.shape,
                                                                     center=centers[j])
            
            

            optimizedPhotoelasticChannel = np.zeros(peImage.shape)
            for j in range(len(centers)):
                optimizedPhotoelasticChannel += genSyntheticResponse(np.array(optimizedForceArr[j]),
                                                                     np.array(optimizedAlphaArr[j]),
                                                                     np.array(optimizedBetaArr[j]),
                                                                     fSigma, radii[j], pxPerMeter, brightfield,
                                                                     imageSize=peImage.shape, center=centers[j])

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
       
    if saveMovie:
        # TODO: Make naming more resilient to missing / at end of directory
        imageArr[0].save(outputFolder + imageDirectory.split('/')[-2].replace('/', '') + '_Synthetic.gif', save_all=True, append_images=imageArr[1:], duration=30, optimize=False, loop=1)


    return forceArr, betaArr, alphaArr, centersArr, radiiArr
