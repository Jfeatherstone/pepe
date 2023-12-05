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

from lmfit import minimize, Parameters, fit_report

def convCircle(singleChannelFrame, radius, radiusTolerance=None, offscreenParticles=False, kernelBlurKernel=3, outlineOnly=False, outlineThickness=.05, negativeHalo=False, haloThickness=.03, negativeInside=False, peakDownsample=10, minPeakPrevalence=.1, intensitySoftmax=1.2, intensitySoftmin=.1, invert=False, allowOverlap=True, fitPeaks=True, debug=False):
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

    allowOverlap : bool
        Whether or not to allow detections of particles that overlap. If set to
        False, then the particle with the stronger signature will be kept
        
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
                                 negativeInside=negativeInside, kernelBlurKernel=kernelBlurKernel, debug=debug)

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
    # (only if we are actually downsampling)
    refinedPeakPositions = []
    refinedRadii = []

    if peakDownsample > 1:
        # To do this, we use a non-linear optimization scheme
        for i in range(len(peakPositions)):

            costArr = []

            def refineObjectiveFunction(params):
                kernelArr = genCircularKernel(np.array([params["y"].value, params["x"].value]), params["r"].value,
                                              imageArr.shape, kernelBlurKernel, outlineOnly,
                                              outlineThickness, negativeHalo, haloThickness, negativeInside)

                # We are minimizing this function, so we need the cost to be negative
                # (unless we have to invert the image)
                costArr.append((np.sum(kernelArr * imageArr) / np.sum(kernelArr)) * (int(invert)*2 - 1))
                return (np.sum(kernelArr * imageArr) / np.sqrt(np.sum(kernelArr))) * (int(invert)*2 - 1)

            params = Parameters()

            params.add('y', value=lorentzPositions[i][0], min=lorentzPositions[i][0]-imageArr.shape[0]*.02, max=lorentzPositions[i][0]+imageArr.shape[0]*.02)
            params.add('x', value=lorentzPositions[i][1], min=lorentzPositions[i][1]-imageArr.shape[1]*.02, max=lorentzPositions[i][1]+imageArr.shape[1]*.02)
            # May or may not want to vary the radius
            # Small tolerances on the radius are so that min != max
            params.add('r', value=np.mean(possibleRadii), vary=(len(possibleRadii) > 1), max=np.max(possibleRadii)-.01, min=np.min(possibleRadii)+.01)

            result = minimize(refineObjectiveFunction, params, method='powell', max_nfev=10000, options={"ftol": 1e-5, "xtol": 1e-5})

            #print(fit_report(result))

            if result is not None:
                refinedPeakPositions.append([result.params["y"].value, result.params["x"].value])
                refinedRadii.append(result.params["r"].value)
                
                #if debug:
                #    plt.plot(costArr)
                #    plt.show()

            else:
                refinedPeakPositions.append(lorentzPositions[i])
                refinedRadii.append(np.mean(possibleRadii))

    else:
        refinedPeakPositions = lorentzPositions
        refinedRadii = np.repeat(np.mean(possibleRadii), len(refinedPeakPositions))


    # Now we (optionally) remove overlapping particles
    # TODO: Allow overlap to be a float value, which allows overlaps
    # that do not exceed a certain threshold (eg. that fraction of the radius).
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
            pMask = genCircularKernel(refinedPeakPositions[i], refinedRadii[i],
                                      singleChannelFrame.shape, kernelBlurKernel, outlineOnly,
                                      outlineThickness, negativeHalo, haloThickness, negativeInside).astype(np.int16)

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


def circularKernelFind(singleChannelFrame, radius, fftPadding, kernelBlurKernel=3, outlineOnly=False, outlineThickness=.05, negativeHalo=False, haloThickness=.03, negativeInside=False, paddingValue=None, debug=False):
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

    kernelBlurKernel : int
        The kernel size to use to blur the kernel that will be convolved with the image, in
        pixels. A value of 0 or None will use the original kernel.

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
    # See function below for more information on generating this kernel
    kernelArr = genCircularKernel(center, radius, paddedImageArr.shape, kernelBlurKernel, outlineOnly,
                                 outlineThickness, negativeHalo, haloThickness, negativeInside)

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


def genCircularKernel(center, radius, imageShape, kernelBlurKernel=3, outlineOnly=False, outlineThickness=.05, negativeHalo=False, haloThickness=.03, negativeInside=False,):

    kernelArr = circularMask(imageShape, center, radius)[:,:,0].astype(np.float64)

    # If we only want the outline, we subtract another circular mask from the above one
    if outlineOnly:
        # Should be 2 if negative inside is true, 1 otherwise
        negativeInsideFactor = 1 + int(negativeInside)
        if outlineThickness < 1:
            innerRadius = np.ceil((1 - outlineThickness) * radius)
        else:
            innerRadius = radius - outlineThickness
        
        kernelArr = kernelArr - negativeInsideFactor * circularMask(imageShape, center, innerRadius)[:,:,0].astype(np.float64)

        if negativeHalo:
            if haloThickness < 1:
                haloRadius = radius + np.ceil((1 - haloThickness) * radius)
            else:
                haloRadius = radius + haloThickness

            kernelArr = 2*kernelArr - (circularMask(imageShape, center, haloRadius)[:,:,0].astype(np.float64) - circularMask(imageShape, center, radius)[:,:,0].astype(np.float64))

    elif negativeHalo:
        if haloThickness < 1:
            haloRadius = np.ceil((1 + haloThickness) * radius)
        else:
            haloRadius = radius + haloThickness

        kernelArr = 2*kernelArr - circularMask(imageShape, center, haloRadius)[:,:,0].astype(np.float64)

    if kernelBlurKernel is not None and kernelBlurKernel > 0:
        kernelArr = cv2.blur(kernelArr, (kernelBlurKernel,kernelBlurKernel))

    return kernelArr
