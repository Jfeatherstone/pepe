import numpy as np
import cv2
import matplotlib.pyplot as plt

import numba
from scipy.fft import fft2, ifft2

from pepe.preprocess import circularMask
from pepe.topology import findPeaks2D


def convCircle(singleChannelFrame, radius, radiusTolerance=None, offscreenParticles=False, peakDownsample=10, minPeakPrevalence=.1, draw=False, intensitySoftmax=1.2):
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

    peakDownsample : int
        The factor by which to downsample the image when performing the initial peak
        detection. The final peak detection will always be performed at the original
        resolution, so this should not affect the accuracy of the detected circles,
        though it is possible that some particles in a densely packed system may
        not be detected.

        Set to `1` or `None` to perform no downsampling, though this is not recommended,
        as the peak finding algorithm is O(N) where N is the total number of grid points.

    minPeakPrevalence : float
        The minimum prevalence (or persistence) of peaks that will be considered as
        particles. Should be a float in the range `[0, 1.]`, representing the percent
        of the total intensity data range that must be spanned by a feature.

        See `pepe.topology.findPeaks2D()` for more information.

    draw : bool
        Whether or not to draw debug information on a plot.
        
    intensitySoftmax : float
        Value at which to cap the intensity values in the image, calculated as
        this value multipled by the mean of the image.

        Set to `None` to skip this preprocessing step.
   
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
        paddingFactor = 2.5
    else:
        paddingFactor = 1.5

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

    # This is the amount that the coordinates will have to be shifted
    # eventually. See documentation for circularKernelFind for more info.
    # If we are varying the radius, this isn't actually true, since this might
    # have to be changed eventually, which is why we call this the initial offset
    initialConvOffset = np.repeat(int(-initialRadius*(paddingFactor+1)), 2)

    if singleChannelFrame.ndim > 2:
        imageArr = np.mean(singleChannelFrame, axis=-1)
    else:
        imageArr = singleChannelFrame

    if intensitySoftmax is not None:
        softmax = np.mean(imageArr) * intensitySoftmax
        imageArr[imageArr >= softmax] = softmax

    # Calculate convolution
    convArr = circularKernelFind(imageArr, initialRadius, fftPadding=int(initialRadius*paddingFactor))

    # Downsample data by 5-10x and run peak detection
    downsampledConvArr = cv2.resize(cv2.blur(convArr, (peakDownsample,peakDownsample)), (0,0),
                                    fx=1/peakDownsample, fy=1/peakDownsample, interpolation=cv2.INTER_CUBIC)

    peakPositions, peakPrevalences = findPeaks2D(downsampledConvArr, minPeakPrevalence=minPeakPrevalence)

        
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

            upsampledPosition = np.array([peakPositions[i][0]*peakDownsample, peakPositions[i][1]*peakDownsample]) + initialConvOffset
            for j in range(len(possibleRadii)):
                # This is the point in the real image (not the conv arr)
                imagePadding = int(possibleRadii[j]*1.2)
                # Crop out the small area of the image that we are working with
                localImage = imageArr[max(upsampledPosition[0]-imagePadding, 0):min(upsampledPosition[0]+imagePadding,imageArr.shape[0]-1),max(upsampledPosition[1]-imagePadding, 0):min(upsampledPosition[1]+imagePadding, imageArr.shape[1]-1)]
                # Perform another convolution
                # Don't need as much padding for this one, since we are quite sure the particle is somewhere
                # near the center
                localConvPadding = int(possibleRadii[j]*.2)
                localConvArr = circularKernelFind(localImage, possibleRadii[j], fftPadding=localConvPadding)

                realLocalMax = np.unravel_index(np.argmax(localConvArr.flatten()), localConvArr.shape)
                # We want to convert these back to real-space coordinates here
                maximumPositions[j] = upsampledPosition + realLocalMax - np.repeat(imagePadding + localConvPadding + possibleRadii[j], 2)
                # I am not sure if there is supposed to be some factor of the radius here, but I
                # suspected there does have to be one.
                # I think the fft/ifft normalization should generally take care of it, but
                # as of now the algorithm will always go with the smallest radius possible,
                # which makes me suspect I am missing a factor TODO
                maximumValues[j] = localConvArr[realLocalMax]
            
            # Now save whichever on had the largest signal
            refinedPeakPositions.append(tuple(maximumPositions[np.argmax(maximumValues)]))
            refinedRadii.append(possibleRadii[np.argmax(maximumValues)])

        else:

            upsampledPosition = np.array([peakPositions[i][0]*peakDownsample, peakPositions[i][1]*peakDownsample])
            # This just looks long because my naming is a little verbose
            # + we also have to make sure we don't accidentally go off of the image
            # That should never happen because the fft padding is quite large, but can't hurt
            # to be extra careful
            localRegion = convArr[max(upsampledPosition[0]-localPadding, 0):min(upsampledPosition[0]+localPadding,convArr.shape[0]-1),max(upsampledPosition[1]-localPadding, 0):min(upsampledPosition[1]+localPadding, convArr.shape[1]-1)]
            # Find maximum intensity in small region around that position
            realLocalMax = np.unravel_index(np.argmax(localRegion.flatten()), localRegion.shape)
            # This is relative to the small region we just created, so we have to subtract off the bounds
            # eg. if this local max was found to be in the center of the local image, that would mean
            # that the original upsampled position was correct.
            refinementOffset = realLocalMax - np.repeat(localPadding, 2)

            # Now put everything together the coordinates of the circle in the original image
            refinedPeakPositions.append(tuple(upsampledPosition + refinementOffset + initialConvOffset))
            refinedRadii.append(initialRadius)


    # Debug option
    if draw:
        fig = plt.figure(figsize=(9,4))

        #ax0 = fig.add_subplot(1, 2, 1)
        #ax0.imshow(singleChannelFrame)
        #ax0.set_title('Original image')

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(peakDownsample*np.arange(downsampledConvArr.shape[1]),
                        peakDownsample*np.vstack(np.arange(downsampledConvArr.shape[0])), downsampledConvArr, cmap=plt.cm.jet)
        ax1.set_title('Convolution')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(singleChannelFrame)
        ax2.set_title('Detected circles')
        for i in range(len(refinedPeakPositions)):
            c = plt.Circle(refinedPeakPositions[i][::-1], refinedRadii[i], color='red', fill=False, linewidth=2)
            ax2.add_artist(c)

        fig.tight_layout()

    return refinedPeakPositions, refinedRadii


def circularKernelFind(singleChannelFrame, radius, fftPadding, debug=False, trimPadding=False):
    """
    Calculate the convolution of a circular mask with an image,
    identifying likely locations of circles within the image. Adapted from
    method described in pages 64-68 of [1].

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

    debug : bool
        Whether or not to plot various quantities of the calculation for inspection
        at the end of the evaluation.

    trimPadding : bool
        Whether to trim the added padding to the image (True) or leave it in the final
        result (False). If trying to identify circles near the outskirts of the image,
        it is recommended to not trim the padding off, since the minimum of the function
        may be located there. In this case, the grid point at which the intensity is maximum
        will be offset from the actual center of that feature by an amount:

        `(-radius - fftPadding, -radius - fftPadding)`.

        If the padding is removed, the offset will simply be:

        `(-radius, -radius)`.

        This offset arises from the fact that the convolutional method expects the center of
        the kernel to be located at `(0, 0)`, but for any real image, we cannot have negative
        coordinates. We center the kernel in the top left corner of an image, but the center
        of the circle has to be at `(radius, radius)` for the entire shape to show up.
    Returns
    -------

    References
    ----------

    [1] Franklin, S.V., Shattuck, M.D. Handbook of Granular Materials (2016) Chapter 2:
        Experimental Techniques
    """

    if singleChannelFrame.ndim > 2:
        imageArr = np.mean(singleChannelFrame, axis=-1)
    else:
        imageArr = singleChannelFrame
    ## First, we calculate conv(I^2, W)

    # Pad the proper image
    paddedImageArr = np.zeros((int(imageArr.shape[0] + 2*fftPadding), int(imageArr.shape[1] + 2*fftPadding)))
    # Note that the intensity is squared here, since we don't want to square the padding
    # later on
    #paddedImageArr[fftPadding:-fftPadding,fftPadding:-fftPadding] = imageArr**2
    paddedImageArr[fftPadding:-fftPadding,fftPadding:-fftPadding] = imageArr

    # Instead of just 0s in the padded area, we want to put in some real-ish values
    # We just use the tanh function to do this (though something like a sigmoid or (r)elu
    # would probably work fine too)
    #rowMat = (1 + np.tanh(3 * np.arange(-0.5, 0.5, paddedImageArr.shape[0])))/2
    #columnMat = (1 + np.tanh(3 * np.arange(-0.5, 0.5, paddedImageArr.shape[1])))/2 
    #Y, X = np.ogrid[:paddedImageArr.shape[0], :paddedImageArr.shape[1]]
    #paddingValues = .1/(1 + np.exp(3*(Y/paddedImageArr.shape[0] - .5)**2 + 3*(X/paddedImageArr.shape[1] - .5)**2))

    # Just some flat value
    #paddingValues = np.zeros_like(paddedImageArr) + .02

    # Now fill in the values
    #paddedImageArr[:fftPadding] = paddingValues[:fftPadding]
    #paddedImageArr[-fftPadding:] = paddingValues[-fftPadding:]
    #paddedImageArr[:,:fftPadding] = paddingValues[:,:fftPadding]
    #paddedImageArr[:,-fftPadding:] = paddingValues[:,-fftPadding:]

    # Top
    paddedImageArr[:fftPadding] += np.multiply.outer(2/(1 + np.exp(np.arange(0, 5, 5/fftPadding))), paddedImageArr[fftPadding])[::-1,:] 
    #paddedImageArr[:fftPadding] += np.multiply.outer(4/(1 + np.exp(np.arange(0, 3, 3/fftPadding))), np.repeat(np.mean(imageArr[0]), paddedImageArr.shape[1]))[::-1,:]
    # Bottom
    paddedImageArr[-fftPadding:] += np.multiply.outer(2/(1 + np.exp(np.arange(0, 5, 5/fftPadding))), paddedImageArr[-fftPadding-1])

    # Left
    paddedImageArr[:,:fftPadding] += np.multiply.outer(paddedImageArr[:,fftPadding], 2/(1 + np.exp(np.arange(0, 5, 5/fftPadding))))[:,::-1]
    # Right
    paddedImageArr[:,-fftPadding:] += np.multiply.outer(paddedImageArr[:,-fftPadding-1], 2/(1 + np.exp(np.arange(0, 5, 5/fftPadding))))


    #center = np.array([paddedImageArr.shape[0]/2, paddedImageArr.shape[1]/2])
    center = np.array([radius,radius])
    # To be able to properly convolute the kernel with the image, they have to
    # be the same size, so just just put our kernel into an array the same size
    # as the image (though most entries will just be zero)
    kernelArr = circularMask(paddedImageArr.shape, center, radius)[:,:,0].astype(np.float64)

    # First convolutional term (and clip out the padding)
    #convTerm1 = ifft2(fft2(paddedImageArr) * fft2(kernelArr))
    convTerm1 = ifft2(fft2(paddedImageArr**2) * fft2(kernelArr))

    if trimPadding:
        convTerm1 = convTerm1[fftPadding:-fftPadding,fftPadding:-fftPadding]

    # Second term is pretty easy since we choose our weight function to be our
    # particle mask
    # This value is explicitly:
    #convTerm2 = ifft2(fft2(kernelArr) * fft2(kernelArr**2))
    # But for our case, this is just the kernel itself (since it only has values of 0 and 1)
    convTerm2 = kernelArr
    #convTerm2 = 2 * ifft2(fft2(paddedImageArr) * fft2(kernelArr))

    if trimPadding:
        convTerm2 = convTerm2[fftPadding:-fftPadding,fftPadding:-fftPadding]

    # This is technically <W I_p^2>, but W = I_p which only have 0s or 1s
    normalizationTerm = np.sum(kernelArr)#[fftPadding:-fftPadding,fftPadding:-fftPadding])

    if trimPadding:
        normalizationTerm = np.sum(kernelAr[fftPadding:-fftPadding,fftPadding:-fftPadding])

    # Put everything together
    # Technically there is a +1 here, but that won't affect any minima, so we don't really care
    chiSqr = np.abs((convTerm1 - convTerm2) / normalizationTerm)

    if debug:
        fig, ax = plt.subplots(1, 3, figsize=(10,4))

        ax[0].imshow(kernelArr)
        ax[0].set_title('Kernel')

        ax[1].imshow(paddedImageArr)
        ax[1].set_title('Padded image')

        ax[2].imshow(chiSqr)
        ax[2].set_title('Real-space convolution (Norm)')

        fig.tight_layout()
        plt.show()


    return chiSqr

