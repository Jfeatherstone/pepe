import numpy as np

import cv2
import os

def loadVideo(path, start=None, end=None):
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

    start : int or None
        The index of the first image that will be returned.

    end : int or None
        The index of the last image that will be returned

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

    while(True):
        if end is not None and i >= end:
            break

        # Ret will be false if we've come to the end of the video
        ret, frame = cam.read()

        if ret:
            yield frame.astype(np.uint8)
            i += 1
        else:
            break


def getNumFrames(path, start=None, end=None):
    """
    Returns the number of images to be loaded.

    Since the images are loaded via a generator, we cannot
    efficiently read how many total images there are directory
    from the generator object.

    Parameters
    ----------
    path : str
        Path to a video file.

    start : int or None
        The index of the first image that will be returned.

    end : int or None
        The index of the last image that will be returned
    """

    cam = cv2.VideoCapture(path)
    length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    return length if (start is None and end is None) else min(length, (end if end is not None else length) - (start if start is not None else 0))
