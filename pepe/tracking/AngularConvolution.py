"""
Particle orientation (rotation) tracking via convolution.
"""
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

def angularConvolution(image, kernelImg, dTheta=.01, angleBounds=(0, 2*np.pi)):
    """
    Perform an angular convolution, calculating the effective similarity between two
    images as one of them is rotated.

    Parameters
    ----------

    image : np.ndarray[H,W] or PIL.Image
       The image for which the rotation should be identified in reference
       to the kernel.

       Should be a single channel image; if a multichannel image is passed,
       the convolution will be performed on the image with the channels
       averaged.

    kernelImg : np.ndarray[H,W] or PIL.Image
        The kernel that will be rotated to be matched up with `image`.

       Should be a single channel image; if a multichannel image is passed,
       the convolution will be performed on the image with the channels
       averaged.

    dTheta : float
        The angle stepsize between subsequent rotations, in radians.

        This will only increase the precision up to a certain limit, determined
        by the resolution of the images.

    angleBounds : tuple(float)
        The minimum (first element) and maximum (second element) angle to rotate
        the kernel by during the convolution, in radians.

    Returns
    -------

    angleArr : np.ndarray
        The angles for which the convolution was calculated over, in radians.

    convArr : np.ndarray
        The similarity between the images at each rotation angle.

    """
    # If multiple channels are passed, we grayscale the image
    if len(np.array(image).shape) == 3:
        grayImage = np.mean(image, axis=-1)
    else:
        grayImage = np.array(image)

    if len(np.array(kernelImg).shape) == 3:
        grayKernel = np.mean(kernelImg, axis=-1)
    else:
        grayKernel = np.array(kernelImg)
    
    # Make sure our images are the same size (and make them so if not)
    paddedHeight = max(grayImage.shape[0], grayKernel.shape[0])
    paddedWidth = max(grayImage.shape[1], grayKernel.shape[1])

    paddedImage = np.zeros((paddedHeight, paddedWidth))
    paddedKernel = np.zeros((paddedHeight, paddedWidth))

    imageSizeDiff = np.array([(paddedHeight - grayImage.shape[0])/4, (paddedWidth - grayImage.shape[1])/4], dtype=np.int64)
    kernelSizeDiff = np.array([(paddedHeight - grayKernel.shape[0])/4, (paddedWidth - grayKernel.shape[1])/4], dtype=np.int64)

    # Add in the negatives for the end
    imagePadding = [imageSizeDiff[0], -imageSizeDiff[0], imageSizeDiff[1], -imageSizeDiff[1]]
    kernelPadding = [kernelSizeDiff[0], -kernelSizeDiff[0], kernelSizeDiff[1], -kernelSizeDiff[1]]

    # Change zeros to be None
    imagePadding = [pad if pad > 0 else None for pad in imagePadding]
    kernelPadding = [pad if pad > 0 else None for pad in kernelPadding]

    paddedImage[imagePadding[0]:imagePadding[1], imagePadding[2]:imagePadding[3]] = grayImage
    paddedKernel[kernelPadding[0]:kernelPadding[1], kernelPadding[2]:kernelPadding[3]] = grayKernel


    steps = abs(int((angleBounds[0] - angleBounds[1])/dTheta))
    # Pillow wants angles in degrees, so we convert here (but all of the parameter values
    # should be given in radians)
    angleArr = np.linspace(angleBounds[0]* 180/np.pi, angleBounds[1] * 180/np.pi, steps)
    # We should always make sure that 0 is in our array
    if 0 not in angleArr:
        angleArr = np.append(angleArr, 0)
        angleArr = np.sort(angleArr)
    
    convArr = np.zeros(len(angleArr))
   
    pilKernel = Image.fromarray(paddedKernel)

    for i in range(len(angleArr)):
        convArr[i] = np.sum((paddedImage - np.array(pilKernel.rotate(angleArr[i])))**2)

    # Normalize a bit
    convArr -= np.max(convArr)
    convArr = np.abs(convArr)
        
    return angleArr * np.pi/180, convArr
