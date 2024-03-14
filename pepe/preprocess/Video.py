import numpy as np

import cv2
import os

def loadVideo(path, start=None, end=None, returnList=False):
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

    returnList : bool
        Whether create a generator and return each frame
        one by one (False) or load every image at once
        and return a list (True).

    Returns
    -------
    images : generator for numpy.ndarray(uint8) or list(numpy.ndarray(uint8))
        Frames of the video, either as a generator or a list.
    """
    # If we want a list, we can just recursively call
    # the function but save the frames afterwards.
    if returnList:
        frameList = []
        for img in loadVideo(path, start, end, False):
            frameList.append(img)

        return frameList

    # We read each frame using opencv
    cam = cv2.VideoCapture(path)
   
    # To keep track of how many images we have, in case some n
    # is provided
    i = start if start is not None else 0

    while(True):
        if end is not None and i > end:
            break

        # Ret will be false if we've come to the end of the video
        ret, frame = cam.read()

        if ret:
            yield frame.astype(np.uint8)
            i += 1
        else:
            break
