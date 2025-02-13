import numpy as np

import cv2
import os

def loadVideo(path, start=None, end=None, skip=None):
    """
    Load in images from a video file.

    This yields each item as a generator to reduce the amount of
    memory required, which is especially helpful for high resolution
    images.

    If you want every frame loaded at once, you can use
    `returnList=True`.

    Parameters
    ----------
    path : str
        Path to a video file.

    start : int, optional
        The index of the first image that will be returned.

    end : int, optional
        The index of the last image that will be returned

    skip : int, optional
        The interval to skip through frames.

        For example, `skip=2` would mean the only frames
        that are returned are 0, 2, 4, 6, ...

    Returns
    -------
    images : generator for numpy.ndarray(uint8) 
        Frames of the video.
    """

    # We read each frame using opencv
    cam = cv2.VideoCapture(path)

    if start is not None:
        cam.set(cv2.CAP_PROP_POS_FRAMES, start-1) 

    # To keep track of how many images we have, in case some n
    # is provided
    i = start if start is not None else 0

    skipInterval = skip if skip is not None else 1
    # Slight subtlety that if the start value
    # isn't evenely divisible by the skip interval, we
    # need to adjust that as the proper remainder
    skipRemainder = 0 if start is None else start % skipInterval

    while(True):
        if end is not None and i >= end:
            break

        # Ret will be false if we've come to the end of the video
        ret, frame = cam.read()

        # Stop if we ran out of frames
        if not ret:
            break

        # Have to skip index values according to skip
        if i % skipInterval != skipRemainder:
            i += 1
            continue

        yield frame.astype(np.uint8)
        i += 1


def getNumFrames(path, start=None, end=None, skip=None):
    """
    Returns the number of images to be loaded.

    Since the images are loaded via a generator, we cannot
    efficiently read how many total images there are directory
    from the generator object.

    Parameters
    ----------
    path : str
        Path to a video file.

    start : int, optional
        The index of the first image that will be returned.

    end : int, optional
        The index of the last image that will be returned

    skip : int, optional
        The interval to skip through frames.

        For example, `skip=2` would mean the only frames
        that are returned are 0, 2, 4, 6, ...
    """

    cam = cv2.VideoCapture(path)
    length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    # First figure out the length based on start and end
    if (start is None and end is None):
        length = length
    else:
        length = min(length, (end if end is not None else length) - (start if start is not None else 0))

    # Now based on skip interval
    if skip is not None:
        length = int(np.ceil(length / skip))

    return length
