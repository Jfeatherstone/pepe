import numpy as np

import cv2

from pepe.analysis import gSquared

def g2HoughCircle(singleChannelFrame, radius, cannyEdgeThreshold=70, accumulatorThreshold=20, radiusTolerance=15, minSeparation=None):
    """

    """

    # Check if we are given a range of radii or just a single value
    minRadius = None
    maxRadius = None

    if hasattr(radius, '__iter__'):
        if len(radius) == 2:
           minRadius = radius[0]
           maxRadius = radius[1]
    else:
        # Use the tolerance
        minRadius = radius - radiusTolerance
        maxRadius = radius + radiusTolerance

    if minRadius is None or maxRadius is None:
        raise Exception('Exception parsing expected radius!')

    # Calculate the gradient squared of the image
    gSqr = gSquared(singleChannelFrame)

    # We don't expect too much overlap, so this should be a reasonable
    # separation distance
    if minSeparation is None:
        minSeparation = 2*minRadius - radiusTolerance

    detectedCircles = cv2.HoughCircles(np.array(gSqr, dtype=np.uint8), 
                    cv2.HOUGH_GRADIENT,
                    dp=1, # Accumulator (no idea what it does, but 1 is good)
                    minDist=minSeparation, # Minimum distance between circles
                    param1=cannyEdgeThreshold, # Threshold for canny edge detection
                    param2=accumulatorThreshold, # Accumulator threshold
                    minRadius=minRadius,
                    maxRadius=maxRadius)

    if detectedCircles is None:
        return None, None

    centers = np.array(list(zip(detectedCircles[0,:,1], detectedCircles[0,:,0])))
    radii = detectedCircles[0,:,2]
    
    return centers, radii

