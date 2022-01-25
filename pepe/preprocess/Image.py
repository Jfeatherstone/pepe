import numpy as np

from PIL import Image
import cv2

def checkImageType(frame):
    """
    Make sure that the image is a proper image, and not a path

    Parameters
    ----------

    frame : str or numpy.ndarray
        Either a path to an image, or an image array

    Returns
    -------

    numpy.ndarray : The image array

    """
    if isinstance(frame, str):
        # I don't want to overwrite the image itself, so create a new var for that
        newFrame = np.array(Image.open(frame), dtype=np.uint8)
    else:
        newFrame = frame

    return newFrame

