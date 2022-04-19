"""
Particle position tracking via convolution.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

import numba
from scipy.fft import fft2, ifft2

from pepe.preprocess import circularMask, rectMask
from pepe.topology import findPeaks2D
from pepe.utils import lorentzian
from pepe.analysis import adjacencyMatrix

from lmfit import minimize, Parameters

def convCircle(singleChannelFrame, radius, radiusTolerance=None, offscreenParticles=False, outlineOnly=False, outlineThickness=.05, negativeHalo=False, haloThickness=.03, negativeInside=False, peakDownsample=10, minPeakPrevalence=.1, intensitySoftmax=1.2, intensitySoftmin=.1, invert=False, allowOverlap=False, fitPeaks=True, debug=False):
    """
    Perform convolution circle detection on the provided image.

    A brief description of the technique is described below; for
    a more comprehensive discussion, see [1].

    The convolution method identifies circles in an image by convolving
    a kernel (representing what a particle should look like) with the
    image within which you would like to detect circles. This essentially
    finds the likelihood of any particle location being the center for
    such an object. By performing peak detection on this 2D quantity,
    we can identify the most likely places for particles to be.

    While this technique can be used for non-circular objects, this 
    implementation makes use of several simplifications/approximations
    that make it suitable only for radially symmetric objects.

    Tends to give more accurate and consistent results than the Hough
    technique (see `pepe.tracking.houghCircle()`), though evaluation
    times can be longer. When the radius of the circle is not well known
    (with no initial guess), the Hough method should be used. That being said,
    this implementation can fine tune the provided radius, but this requires
    the provided value to be reasonably close to the actual value (within 
    ~10 pixels is ideal).

    The ability to get subpixel precision by interpolating/fitting curves to the
    convolution is currently WIP, and currently the method can, at best, give results
    accurate to within a single pixel.

    
    Parameters
    ----------
 
    singleChannelFrame : np.uint8[H, W]
        The frame to perform circle detection on. Should not include multiple
        color channels. If it does, the channels will be averaged to form a
        grayscale image.

    radius : float or [float, float]
        The radius to look for, or min and max radii to constrain the circle
        detection. If a single value is provided, a range will be constructed
        around that value with the `radiusTolerance` parameter.

        Note that this method is more sensitive to the initial guess of radius
        than `pepe.tracking.houghCircle()`. Ideally, the radius should be within
        ~10 pixels of the true radius (or a range of radii should only be ~15 pixels wide).

        NOTE: Varying the radius is still under development, and likely does not work that great.

    radiusTolerance : float or None
        The half-width of the range of radii that will be tested for detecting particles.
        ie. the range will be calculated as `[radius + radiusTolerance, radius - radiusTolerance]`.

        If set to `None`, radius will be fixed at whatever value is provided via `radius` argument.

        NOTE: Varying the radius is still under development, and likely does not work that great.

    offscreenParticles : bool
        Whether or not the algorithm should look for particles whose centers are outside of
        the image domain (but some parts of the particles still show up in the image).

        This expands the padding provided to the fft/ifft calculations, so as to avoid cutting
        off what might be a potential particle. As a result, a value of True will increase computation times.

    outlineOnly : bool
        Whether to look for only a particle outline (True), or to look for a filled-in circle (False).

        See `outlineThickness`.

    outlineThickness : float or int
        Thickness of the particle outline in the kernel; only relevant if `outlineOnly=True`.

        If float, will be taken as fraction of the radius; if integer, will be taken as number
        of pixels.

    negativeHalo : bool
        Whether to surround the particle mask with a region of negative value, which
        further promotes identifying distinct objects as grains.

    haloThickness : float or int
        Thickness of the negative halo in the kernel; only relevant if `negativeHalo=True`.

        If float, will be taken as fraction of the radius; if integer, will be taken as number
        of pixels.

    negativeInside : bool
        Whether to fill the inside of the particle mask with a negative value, which 
        further promotes identifying outlines of circles.

        Has no effect unless `outlineOnly=True`.

    peakDownsample : int
        The factor by which to downsample the image when performing the initial peak
        detection. The final peak detection will always be performed at the original
        resolution, so this should not affect the accuracy of the detected circles,
        though it is possible that some particles in a densely packed system may
        not be detected. False

        Set to `1` or `None` to perform no downsampling, though this is not recommended,
        as the peak finding algorithm is O(N) where N is the total number of grid points.

    minPeakPrevalence : float
        The minimum prevalence (or persistence) of peaks that will be considered as
        particles. Should be a float in the range `[0, 1.]`, representing the percent
        of the total intensity data range that must be spanned by a feature.

        See `pepe.topology.findPeaks2D()` for more information.

    fitPeaks : bool
        Whether or not to fit each peak in the convolution with a lorentzian
        form and use that center as the location of the particle (instead of
        the simple maximum point).

    intensitySoftmax : float
        Value at which to cap the intensity values in the image, calculated as
        this value multipled by the mean of the image.

        Set to `None` to skip this preprocessing step.

    intensitySoftmin : float
        Zero-out all values that have an intensity below this factor times the mean of
        the image.

        Set to `None` to skip this preprocessing step.
        
    debug : bool
        Whether or not to draw debug information on a plot. 

    invert : bool
        If particles appear dark on a bright background, this will invert the
        image (and convolution peak finding will go from looking for maxima
        to looking for minima).
   
    Returns
    -------

    centers : np.ndarray[N,2]
        The detected circles' centers.

    radii : np.ndarray[N]
        The detected circles' radii.


    References
    ----------

    [1] Franklin, S. V., & Shattuck, M. D. (Eds.). (2016). Handbook
    of Granular Materials. Chapter 2: Experimental Techniques. p.63-68.
    CRC Press. https://doi.org/10.1201/b19291

    """

    # General housekeeping of arguments, see if we need to parse anything
    if offscreenParticles:
        paddingFactor = 2.
    else:
        paddingFactor = 1.1

    if peakDownsample is None:
        peakDownsample = 1
 
    # Check if we are given a range of radii or just a single value
    possibleRadii = None

    if hasattr(radius, '__iter__'):
        if len(radius) == 2 and radius[0] < radius[1]:
           possibleRadii = np.arange(radius[0], radius[1]+1)
        else:
            raise Exception(f'Invalid radius provided to convCircle()!\n Expected scalar or [min, max], but got {radius}!')
    elif radiusTolerance is not None:
        # Use the tolerance
        possibleRadii = np.arange(radius - radiusTolerance, radius + radiusTolerance)
    else:
        # Just the single value
        possibleRadii = np.array([radius])

    # Start with just the mean of the possible radii options
    initialRadius = np.mean(possibleRadii)

    if singleChannelFrame.ndim > 2:
        imageArr = np.mean(singleChannelFrame, axis=-1)
    else:
        imageArr = np.copy(singleChannelFrame)
            
    if invert:
        imageArr = np.max(imageArr) - imageArr

    if intensitySoftmax is not None:
        softmax = np.mean(imageArr) * intensitySoftmax
        if not invert:
            imageArr[imageArr >= softmax] = softmax
        #else:
        #    imageArr[255 - imageArr >= softmax] = softmax

    if intensitySoftmin is not None:
        softmin = np.mean(imageArr) * intensitySoftmin
        if not invert:
            imageArr[imageArr <= softmin] = 0
        #else:
        #    imageArr[255 - imageArr <= softmin] = 0

    # Calculate convolution
    convArr = circularKernelFind(imageArr, initialRadius, fftPadding=int(initialRadius*paddingFactor),
                                 outlineOnly=outlineOnly, outlineThickness=outlineThickness,
                                 negativeHalo=negativeHalo, haloThickness=haloThickness,
                                 negativeInside=negativeInside, debug=debug)

    if debug:
        plt.show()

    # Downsample data by 5-10x and run peak detection
    downsampledConvArr = cv2.resize(cv2.blur(convArr, (peakDownsample,peakDownsample)), (0,0),
                                    fx=1/peakDownsample, fy=1/peakDownsample, interpolation=cv2.INTER_CUBIC)

    peakPositions, peakPrevalences = findPeaks2D(downsampledConvArr, minPeakPrevalence=minPeakPrevalence)

    lorentzPositions = np.zeros((len(peakPositions), 2))
    cleanedConvArr = np.zeros_like(convArr)

    # We have the option to fit lorentzian peaks to each maximum in the convolution
    # matrix, or we can just take the peak positions as they are
    if fitPeaks:
        def objectiveFunction(params, localImage):
            # Create a small grid across the area
            Y,X = np.ogrid[:localImage.shape[0], :localImage.shape[1]]
            Y += int(params["center_y"] - localImage.shape[0]/2)
            X += int(params["center_x"] - localImage.shape[1]/2)

            lorentzArr = lorentzian([Y,X], [params["center_y"], params["center_x"]], params["width"], params["amp"], params["offset"])
            
            return np.sum((localImage - lorentzArr)**2)

        # Now we fit an appropriate function to each peak to see if we can refine the center
        for i in range(len(peakPositions)):
            upsampledPosition = np.array([peakPositions[i][0]*peakDownsample, peakPositions[i][1]*peakDownsample])
            imagePadding = int(np.mean(possibleRadii))
            # Crop out the small area of the image around the peak
            localConv = convArr[max(upsampledPosition[0]-imagePadding, 0):min(upsampledPosition[0]+imagePadding,convArr.shape[0]-1), max(upsampledPosition[1]-imagePadding, 0):min(upsampledPosition[1]+imagePadding, convArr.shape[1]-1)]


            #plt.imshow(localConv)
            #plt.show()
            # Now setup a fit to this function
            params = Parameters()

            params.add('center_y', value=localConv.shape[0]/2, vary=True)
            params.add('center_x', value=localConv.shape[1]/2, vary=True)
            params.add('width', value=imagePadding/2, vary=False)
            params.add('amp', value=np.max(localConv)-np.min(localConv), vary=False)
            params.add('offset', value=np.min(localConv), vary=False)
        
            result = minimize(objectiveFunction, params, args=[localConv], method='nelder', max_nfev=10000)
            
            if result is not None:
                lorentzPositions[i,0] = upsampledPosition[0] - localConv.shape[0]/2 + result.params["center_y"]
                lorentzPositions[i,1] = upsampledPosition[1] - localConv.shape[1]/2 + result.params["center_x"]

                Y,X = np.ogrid[:convArr.shape[0], :convArr.shape[1]]
                cleanedConvArr += lorentzian([Y,X], lorentzPositions[i], result.params["width"], result.params["amp"], result.params["offset"])

            else:
                lorentzPositions[i] = upsampledPosition

    else:
        for i in range(len(peakPositions)):
            upsampledPosition = np.array([peakPositions[i][0]*peakDownsample, peakPositions[i][1]*peakDownsample])
            lorentzPositions[i] = upsampledPosition

    # Look around each peak to find the real peak in the full-resolution image
    refinedPeakPositions = []
    refinedRadii = []

    localPadding = int(peakDownsample*1.5)
    for i in range(len(peakPositions)):
        # We want to vary the radius here as well, so we can maybe get a slightly more accurate result.
        # Note that it is much more efficient to perform the kernel finding on a small image many times
        # than to work on a large image just a few times. So we recalculate the kernel finding for each
        # radii for each peak.
        # TODO: This part is still WIP, because it doens't quite work correctly
        if len(possibleRadii) > 1:
            maximumValues = np.zeros(len(possibleRadii))
            maximumPositions = np.zeros((len(possibleRadii), 2))

            #upsampledPosition = np.array([peakPositions[i][0]*peakDownsample, peakPositions[i][1]*peakDownsample])
            upsampledPosition = np.int16(lorentzPositions[i])
            for j in range(len(possibleRadii)):
                # This is the point in the real image (not the conv arr)
                imagePadding = int(possibleRadii[j])
                # Crop out the small area of the image that we are working with
                localImage = imageArr[max(upsampledPosition[0]-imagePadding, 0):min(upsampledPosition[0]+imagePadding,imageArr.shape[0]-1),max(upsampledPosition[1]-imagePadding, 0):min(upsampledPosition[1]+imagePadding, imageArr.shape[1]-1)]

                # Perform another convolution
                # Don't need as much padding for this one, since we are quite sure the particle is somewhere
                # near the center
                localConvPadding = int(possibleRadii[j])
                localConvArr = circularKernelFind(localImage, possibleRadii[j], fftPadding=localConvPadding,
                                                  outlineOnly=outlineOnly, outlineThickness=outlineThickness,
                                                  negativeHalo=negativeHalo, haloThickness=haloThickness,
                                                  negativeInside=negativeInside, debug=False)
                # Required if you set debug=True in above line
                #plt.show()

                # Add a *very* small weighting that favors circles closer to the center of the
                # image. This is needed because otherwise the algorithm may place circles more
                # offscreen than they actually are, because it would mean fewer pixels are included
                # (since some are cropped off)
                #Y = np.arange(localImage.shape[0]).reshape((localImage.shape[0], 1)) # Column vector
                #X = np.arange(localImage.shape[1]).reshape((1, localImage.shape[1])) # Row vector
                #localConvArr += 1e-3*np.exp(- ( (Y - localImage.shape[0]/2)**2 + (X - localImage.shape[1]/2)**2 ) / max(localImage.shape))

                # Now setup a fit to this function
                params = Parameters()

                params.add('center_y', value=localImage.shape[0]/2, vary=True)
                params.add('center_x', value=localImage.shape[1]/2, vary=True)
                params.add('width', value=imagePadding, vary=True)
                params.add('amp', value=np.max(localConvArr)/2, vary=True)
                params.add('offset', value=np.mean(localConvArr), vary=True)
                
                try:
                    result = minimize(objectiveFunction, params, args=[localConvArr], method='nelder', max_nfev=1000)
                except:
                    result = None
                
                if result is not None:
                    realLocalMax = np.array([int(result.params["center_y"]), int(result.params["center_x"])])
                else:
                    realLocalMax = np.zeros(2)

                #realLocalMax = np.unravel_index(np.argmax(localConvArr.flatten()), localConvArr.shape)
                # We want to convert these back to real-space coordinates here
                #maximumPositions[j] = upsampledPosition - realLocalMax - np.repeat(localConvPadding, 2)
                maximumPositions[j,0] = upsampledPosition[0] - localConvArr.shape[0]/2 + realLocalMax[0]
                maximumPositions[j,1] = upsampledPosition[1] - localConvArr.shape[1]/2 + realLocalMax[1]

                # Need to convert to int16 since we might have negative values
                pMask = circularMask(singleChannelFrame.shape, maximumPositions[j], possibleRadii[j])[:,:,0].astype(np.int16)

                # The next couple if statements are for only doing an outline, adding a halo, removing the center, etc.
                # If we only want the outline, we subtract another circular mask from the above one
                if outlineOnly:
                    # Should be 2 if negative inside is true, 1 otherwise
                    negativeInsideFactor = 1 + int(negativeInside)
                    if outlineThickness < 1:
                        innerRadius = np.ceil((1 - outlineThickness) * possibleRadii[j])
                    else:
                        innerRadius = possibleRadii[j] - outlineThickness
                        pMask -= negativeInsideFactor * circularMask(singleChannelFrame.shape, maximumPositions[j], innerRadius)[:,:,0]

                    if negativeHalo:
                        pMask = 2*pMask - (circularMask(singleChannelFrame.shape, maximumPositions[j], possibleRadii[j]+2)[:,:,0].astype(np.int16) - circularMask(singleChannelFrame.shape, maximumPositions[j], possibleRadii[j])[:,:,0].astype(np.int16))

                elif negativeHalo:
                    pMask = 2*pMask - circularMask(singleChannelFrame.shape, maximumPositions[j], possibleRadii[j]+2)[:,:,0].astype(np.int16)


                if np.sum(pMask) == 0:
                    maximumValues[j] = 0
                else:
                    maximumValues[j] = np.sum(singleChannelFrame * pMask) / np.sum(pMask)
           
            

            # Noise can cause some slight variations in the maximum values making different
            # radii give different maximum values, when they should give actually the same
            # value (and then the largest one should be taken). We fix this by rounding
            # after multiply by a large number. .5e3 was chosen by experimenting

            maximumValues = [int(m*1e2) if not np.isnan(m) else 0 for m in maximumValues]

            #plt.plot(possibleRadii, maximumValues)
            #plt.show()

            # Now save whichever on had the largest signal
            # We want to sort the list backwards, because we want the largest possible radius
            # that also gives the best value. If the radius is smaller than the actual one, you
            # can end up with the same exact maximum value, since you are just cutting off extra
            # parts of the circle.
            refinedPeakPositions.append(tuple(maximumPositions[::-1][np.argmax(maximumValues[::-1])]))
            refinedRadii.append(possibleRadii[::-1][np.argmax(maximumValues[::-1])])

        else:
            # Position of the peaks in the convolution image (NOT the real image)
            upsampledPosition = np.int16(lorentzPositions[i])
            
            # We don't need to refine if we fit a peak, since that already did refine the
            # position
            if fitPeaks:
                refinementOffset = 0

            else:
                # This just looks long because my naming is a little verbose
                # + we also have to make sure we don't accidentally go off of the image
                localRegion = convArr[max(upsampledPosition[0]-localPadding, 0):min(upsampledPosition[0]+localPadding,convArr.shape[0]-1),max(upsampledPosition[1]-localPadding, 0):min(upsampledPosition[1]+localPadding, convArr.shape[1]-1)]
                # Find maximum intensity in small region around that position
                realLocalMax = np.unravel_index(np.argmax(localRegion.flatten()), localRegion.shape)
                # This is relative to the small region we just created, so we have to subtract off the bounds
                # eg. if this local max was found to be in the center of the local image, that would mean
                # that the original upsampled position was correct.
                refinementOffset = realLocalMax - np.repeat(localPadding, 2)

            # Now put everything together the coordinates of the circle in the original image
            refinedPeakPositions.append(tuple(upsampledPosition + refinementOffset))
            refinedRadii.append(initialRadius)


    # Now we (optionally) remove overlapping particles
    if not allowOverlap:
        # Calculate the contact matrix
        # negative contact padding because we don't want to remove particles that are
        # good but just close to each other
        adjMat = adjacencyMatrix(refinedPeakPositions, refinedRadii, contactPadding=-int(np.mean(refinedRadii)/10)) - np.eye(len(refinedPeakPositions))

        # Locate all of the particles that are overlapping
        overlappingParticles = np.array([i for i in range(len(refinedPeakPositions)) if np.sum(adjMat[i,:]) > 0])

        # Now find how good of detections each of these is, and order them from worst to best.
        particleScores = np.zeros(len(overlappingParticles))
        for i in range(len(particleScores)):
            # Need to convert to int16 since we might have negative values
            pMask = circularMask(singleChannelFrame.shape, refinedPeakPositions[i], refinedRadii[i])[:,:,0].astype(np.int16)

            # The next couple if statements are for only doing an outline, adding a halo, removing the center, etc.
            # If we only want the outline, we subtract another circular mask from the above one
            if outlineOnly:
                # Should be 2 if negative inside is true, 1 otherwise
                negativeInsideFactor = 1 + int(negativeInside)
                if outlineThickness < 1:
                    innerRadius = np.ceil((1 - outlineThickness) * refinedRadii[i])
                else:
                    innerRadius = refinedRadii[i] - outlineThickness
                    pMask -= negativeInsideFactor * circularMask(singleChannelFrame.shape, refinedPeakPositions[i], innerRadius)[:,:,0]

                if negativeHalo:
                    pMask = 2*pMask - (circularMask(singleChannelFrame.shape, refinedPeakPositions[i], refinedRadii[i]+2)[:,:,0].astype(np.int16) - circularMask(singleChannelFrame.shape, refinedPeakPositions[i], refinedRadii[i])[:,:,0].astype(np.int16))

            elif negativeHalo:
                pMask = 2*pMask - circularMask(singleChannelFrame.shape, refinedPeakPositions[i], refinedRadii[i]+2)[:,:,0].astype(np.int16)

            if np.sum(pMask) > 0:
                # I square-root the mask sum so that the particles at the edge don't
                # always win out against actual particles
                # TODO: This still doesn't give priority to particles on screen;
                # need to figure out a way to do that
                particleScores[i] = np.sum(singleChannelFrame * pMask) / np.sqrt(np.sum(pMask))
            else:
                particleScores[i] = 0 

        # Sort according to worst scores
        order = np.argsort(particleScores)[::-1]
        overlappingParticles = overlappingParticles[order]

        # Now we start removing the particles with the worst scores until we no longer have overlap
        index = 0
        while np.sum(adjMat) > 0:
            adjMat[overlappingParticles[index],:] = 0
            adjMat[:,overlappingParticles[index]] = 0
            refinedPeakPositions[overlappingParticles[index]] = None
            refinedRadii[overlappingParticles[index]] = None

            index += 1

           
        # Now remove all of the None values we put in
        refinedPeakPositions = [rpp for rpp in refinedPeakPositions if rpp is not None]
        refinedRadii = [rr for rr in refinedRadii if rr is not None]


    # Debug option
    if debug:
        fig = plt.figure(figsize=(12,4))

        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot_surface(np.arange(convArr.shape[1]),
                        np.vstack(np.arange(convArr.shape[0])), convArr, cmap=plt.cm.jet)
        ax1.set_title('Convolution')

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.plot_surface(np.arange(convArr.shape[1]),
                        np.vstack(np.arange(convArr.shape[0])), cleanedConvArr, cmap=plt.cm.jet)
        ax2.set_title('Cleaned Convolution')

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(singleChannelFrame)
        ax3.set_title('Detected circles')
        for i in range(len(refinedPeakPositions)):
            c = plt.Circle(refinedPeakPositions[i][::-1], refinedRadii[i], color='red', fill=False, linewidth=1)
            ax3.add_artist(c)

        fig.tight_layout()

    return np.array(refinedPeakPositions), np.array(refinedRadii)


def circularKernelFind(singleChannelFrame, radius, fftPadding, outlineOnly=False, outlineThickness=.05, negativeHalo=False, haloThickness=.03, negativeInside=False, paddingValue=None, debug=False):
    """
    Calculate the convolution of a circular mask with an image,
    identifying likely locations of circles within the image. Adapted from
    method described in [1].

    To perform the convolution, we take the product of the fft of each the
    kernel and image, and then ifft the result to transform back to real space.
    The only peculiarity is that there can be some issues with the edges of image
    (especially if any circle centers are near the edge/off-screen) so we
    have to take care to properly pad the image and the kernel.

    Note that this method is not intended to be used as the front end for
    circle detection for real data. The method `pepe.tracking.convCircle()`
    makes use of this method to track circles, while also offering many more
    features. It is recommended to use that function unless there is a very
    specific purpose that it does not serve.

    Parameters
    ----------
    singleChannelFrame : np.ndarray[H,W]
        A single-channel image (grayscale or just one channel) within which we
        are looking to identify circular objects.

    radius : float
        The radius of circles to detect in the image.

    fftPadding : int
        The amount of padding to add to each side of the image, to prevent the
        fft from having issues. A good choice is usually 1.5 times the radius,
        though, less padding can be used if circles are not expected to often
        be located near the edges.

    outlineOnly : bool
        Whether to look for only a particle outline (True), or to look for a filled-in circle (False).

        See `outlineThickness`.

    outlineThickness : float or int
        Thickness of the particle outline in the kernel; only relevant if `outlineOnly=True`.

        If float, will be taken as fraction of the radius; if integer, will be taken as number
        of pixels.

    negativeHalo : bool
        Whether to surround the particle mask with a region of negative value, which
        further promotes identifying distinct objects as grains.

    haloThickness : float or int
        Thickness of the negative halo in the kernel; only relevant if `negativeHalo=True`.

        If float, will be taken as fraction of the radius; if integer, will be taken as number
        of pixels.

    negativeInside : bool
        Whether to fill the inside of the particle mask with a negative value, which 
        further promotes identifying outlines of circles.

        Has no effect unless `outlineOnly=True`.

    paddingValue : [float, `'sigmoid'`, or `None`]
        What value to fill in for the extra padding that is added during the fft/ifft.

        `None` leaves the value at 0.

        `'sigmoid'` will create a smooth-ish transition to 0.

        Any float value will be inserted into every point.

    debug : bool
        Whether or not to plot various quantities of the calculation for inspection
- `S`: Soft particles; shore hardness 40
- `M`: Medium particles; shore hardness 50
- `H`: Hard particles; shore hardness 60, thickness
        at the end of the evaluation.

    Returns
    -------

    chiSqr : np.ndarray[H,W]
        The convolution of the kernel with the image. Larger values represent
        higher likelihood that there is a particle centered there.

    References
    ----------

    [1] Franklin, S. V., & Shattuck, M. D. (Eds.). (2016). Handbook
    of Granular Materials. Chapter 2: Experimental Techniques. p.64-68.
    CRC Press. https://doi.org/10.1201/b19291
    """

    if singleChannelFrame.ndim > 2:
        imageArr = np.mean(singleChannelFrame, axis=-1)
    else:
        imageArr = singleChannelFrame

    fftPadding = int(fftPadding)

    # Pad the proper image
    paddedImageArr = np.zeros((int(imageArr.shape[0] + 2*fftPadding), int(imageArr.shape[1] + 2*fftPadding)))
    paddedImageArr[fftPadding:-fftPadding,fftPadding:-fftPadding] = imageArr

    # Both dimensions will always have the same first crop index, but possibly a different second one
    cropBounds = [int(fftPadding + radius), int(-fftPadding + radius), int(-fftPadding + radius)]
    # If the fftpadding and the radius are exactly the same, we get some weird behavior
    # This happens because python can index negative values as that element from the end of
    # the array, but indexing -0 just gives 0, and so it gives you a (0,0) array.
    if cropBounds[1] >= 0:
        cropBounds[1] = None
        cropBounds[2] = None

    # We have a couple options for what values to put in the padding
    if paddingValue is None or paddingValue == 0:
        # Do nothing, and leave the value at 0
        pass

    elif paddingValue == 'sigmoid':
        # Sigmoid gives a smoothish decay from the image to 0 at the boundaries
        # Most of the constants here are just arbitrarily chosen to make things look good
        # Top
        paddedImageArr[:fftPadding] += np.multiply.outer(2/(1 + np.exp(np.linspace(0, 5, fftPadding))), paddedImageArr[fftPadding])[::-1,:] 
        # Bottom
        paddedImageArr[-fftPadding:] += np.multiply.outer(2/(1 + np.exp(np.linspace(0, 5, fftPadding))), paddedImageArr[-fftPadding-1])
        # Left
        paddedImageArr[:,:fftPadding] += np.multiply.outer(paddedImageArr[:,fftPadding], 2/(1 + np.exp(np.linspace(0, 5, fftPadding))))[:,::-1]
        # Right
        paddedImageArr[:,-fftPadding:] += np.multiply.outer(paddedImageArr[:,-fftPadding-1], 2/(1 + np.exp(np.linspace(0, 5, fftPadding))))

    else:
        # Just put a fixed value in
        paddedImageArr[:fftPadding] = paddingValue
        paddedImageArr[-fftPadding:] = paddingValue
        paddedImageArr[fftPadding:-fftPadding,:fftPadding] = paddingValue
        paddedImageArr[fftPadding:-fftPadding,-fftPadding:] = paddingValue

    center = np.array([radius,radius])
    # To be able to properly convolute the kernel with the image, they have to
    # be the same size, so just just put our kernel into an array the same size
    # as the image (though most entries will just be zero)
    kernelArr = circularMask(paddedImageArr.shape, center, radius)[:,:,0].astype(np.float64)
   
    # If we only want the outline, we subtract another circular mask from the above one
    if outlineOnly:
        # Should be 2 if negative inside is true, 1 otherwise
        negativeInsideFactor = 1 + int(negativeInside)
        if outlineThickness < 1:
            innerRadius = np.ceil((1 - outlineThickness) * radius)
        else:
            innerRadius = radius - outlineThickness
        
        kernelArr = kernelArr - negativeInsideFactor * circularMask(paddedImageArr.shape, center, innerRadius)[:,:,0].astype(np.float64)

        if negativeHalo:
            if haloThickness < 1:
                haloRadius = radius + np.ceil((1 - haloThickness) * radius)
            else:
                haloRadius = radius + haloThickness

            kernelArr = 2*kernelArr - (circularMask(paddedImageArr.shape, center, haloRadius)[:,:,0].astype(np.float64) - circularMask(paddedImageArr.shape, center, radius)[:,:,0].astype(np.float64))

    elif negativeHalo:
        if haloThickness < 1:
            haloRadius = np.ceil((1 + haloThickness) * radius)
        else:
            haloRadius = radius + haloThickness

        kernelArr = 2*kernelArr - circularMask(paddedImageArr.shape, center, haloRadius)[:,:,0].astype(np.float64)

    # First convolutional term
    convTerm1 = ifft2(fft2(paddedImageArr**2) * fft2(kernelArr))
    # Trim padding
    convTerm1 = convTerm1[cropBounds[0]:cropBounds[1],cropBounds[0]:cropBounds[2]]

    # Second term is pretty easy since we choose our weight function to be our
    # particle mask (and it just has 0s and 1s, so it is equal to it's square)
    convTerm2 = 2 * ifft2(fft2(paddedImageArr) * fft2(kernelArr))
    # Trim padding
    convTerm2 = convTerm2[cropBounds[0]:cropBounds[1],cropBounds[0]:cropBounds[2]]

    # For the normalization term, we want to sum all parts of the kernel that are
    # on screen (in the original image's frame) as we translate it across the image.
    originalImageMask = rectMask(paddedImageArr.shape, np.repeat(fftPadding, 2), np.array(imageArr.shape))[:,:,0]
    normalizationTerm = ifft2(fft2(originalImageMask) * fft2(kernelArr))
    # Trim padding
    normalizationTerm = np.real(normalizationTerm[cropBounds[0]:cropBounds[1],cropBounds[0]:cropBounds[2]])
    # Put everything together
    # Technically there is a +1 here, but that won't affect any minima, so we don't really care
    chiSqr = np.real((convTerm1 - convTerm2) / normalizationTerm)
    if np.min(chiSqr) < 0 and np.max(chiSqr) < 0:
        chiSqr = np.abs(chiSqr)

    if debug:
        fig, ax = plt.subplots(1, 4, figsize=(13,4))

        ax[0].imshow(kernelArr)
        ax[0].set_title('Kernel')

        ax[1].imshow(paddedImageArr)
        ax[1].set_title('Padded image')

        ax[2].imshow(normalizationTerm)
        ax[2].set_title('Normalization')

        ax[3].imshow(chiSqr)
        ax[3].set_title('Real-space convolution (Re)')

        fig.tight_layout()

    return chiSqr

