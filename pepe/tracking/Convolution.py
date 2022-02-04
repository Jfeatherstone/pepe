import numpy as np

import matplotlib.pyplot as plt

import numba

from scipy.fft import fft2, ifft2

from pepe.preprocess import circularMask

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
    paddedImageArr = np.zeros((imageArr.shape[0] + 2*fftPadding, imageArr.shape[1] + 2*fftPadding))
    # Note that the intensity is squared here, since we don't want to square the padding
    # later on
    paddedImageArr[fftPadding:-fftPadding,fftPadding:-fftPadding] = imageArr**2

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
    convTerm1 = ifft2(fft2(paddedImageArr) * fft2(kernelArr))

    if trimPadding:
        convTerm1 = convTerm1[fftPadding:-fftPadding,fftPadding:-fftPadding]

    # Second term is pretty easy since we choose our weight function to be our
    # particle mask
    # This value is explicitly:
    #convTerm2 = ifft2(fft2(kernelArr) * fft2(kernelArr**2))
    # But for our case, this is just the kernel itself (since it only has values of 0 and 1)
    convTerm2 = kernelArr
    
    if trimPadding:
        convTerm2 = convTerm2[fftPadding:-fftPadding,fftPadding:-fftPadding]

    # This is technically <W I_p^2>, but W = I_p which only have 0s or 1s
    normalizationTerm = np.sum(kernelArr)#[fftPadding:-fftPadding,fftPadding:-fftPadding])

    if trimPadding:
        normalizationTerm = np.sum(kernelAr[fftPadding:-fftPadding,fftPadding:-fftPadding])

    # Put everything together
    # Technically there is a +1 here, but that won't affect any minima, so we don't really care
    chiSqr = (convTerm1 - convTerm2) / normalizationTerm

    if debug:
        fig, ax = plt.subplots(1, 3, figsize=(10,4))

        ax[0].imshow(kernelArr)
        ax[0].set_title('Kernel')

        ax[1].imshow(paddedImageArr)
        ax[1].set_title('Padded image')

        ax[2].imshow(np.abs(chiSqr))
        ax[2].set_title('Real-space convolution (Norm)')

        fig.tight_layout()
        plt.show()


    return chiSqr


def kernelFind(image, kernel, fftPadding=100, debug=False):
    """

    """

    # Make sure we aren't passed multiple channels
    # If we do, we just average over all of the channels, making
    # the image grayscale (sorta)
    if image.ndim > 2:
        imageArr = np.mean(image, axis=-1)
    else:
        imageArr = image

    if kernel.ndim > 2:
        kernelArr = np.mean(kernel, axis=-1)
    else:
        kernelArr = kernel

    # Normalize
    imageArr /= np.max(imageArr)
    kernelArr /= np.max(kernelArr)

    # Pad the proper image
    paddedImageArr = np.zeros((imageArr.shape[0] + 2*fftPadding, imageArr.shape[1] + 2*fftPadding))
    paddedImageArr[fftPadding:-fftPadding,fftPadding:-fftPadding] = imageArr

    # Instead of just 0s in the padded area, we want to put in some real-ish values
    # We just use the tanh function to do this (though something like a sigmoid or (r)elu
    # would probably work fine too)
    #rowMat = (1 + np.tanh(3 * np.arange(-0.5, 0.5, paddedImageArr.shape[0])))/2
    #columnMat = (1 + np.tanh(3 * np.arange(-0.5, 0.5, paddedImageArr.shape[1])))/2
  
    Y, X = np.ogrid[:paddedImageArr.shape[0], :paddedImageArr.shape[1]]

    paddingValues = 1/(1 + np.exp(3*(Y/paddedImageArr.shape[0] - .5)**2 + 3*(X/paddedImageArr.shape[1] - .5)**2))

    #rowMat = 1/(1 + np.exp(np.arange(0, 1, paddedImageArr.shape[0])))
    #columnMat = 1/(1 + np.exp(np.arange(0, 1, paddedImageArr.shape[1])))

    #paddingValues = np.add.outer(rowMat, columnMat)

    # Now fill in the values
    paddedImageArr[:fftPadding] = paddingValues[:fftPadding]
    paddedImageArr[-fftPadding:] = paddingValues[-fftPadding:]
    paddedImageArr[:,:fftPadding] = paddingValues[:,:fftPadding]
    paddedImageArr[:,-fftPadding:] = paddingValues[:,-fftPadding:]

    # To be able to properly convolute the kernel with the image, they have to
    # be the same size, so just just put our kernel into an array the same size
    # as the image (though most entries will just be zero)
    resizedKernelArr = np.zeros_like(paddedImageArr)
    resizedKernelArr[0:kernelArr.shape[0],0:kernelArr.shape[1]] = kernelArr

    # Now we calculate the convolution of the image and kernel, which is just
    # the product of their fourier transforms
    convolutionFFTArr = fft2(paddedImageArr) * fft2(resizedKernelArr)

    # Now ifft back to real space, and then cut off the padding that we
    # introduced
    convolutionArr = ifft2(convolutionFFTArr)
    clippedConvolutionArr = convolutionArr[fftPadding:-fftPadding,fftPadding:-fftPadding]

    if debug:
        fig, ax = plt.subplots(1, 3, figsize=(10,4))

        ax[0].imshow(paddingValues)
        ax[0].set_title('FFT padding')

        ax[1].imshow(paddedImageArr)
        ax[1].set_title('Padded image')

        ax[2].imshow(np.real(clippedConvolutionArr))
        ax[2].set_title('Real-space convolution (Re)')

        fig.tight_layout()
        plt.show()

    return clippedConvolutionArr
