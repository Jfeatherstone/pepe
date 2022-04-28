
import numpy as np

def derivative(xArr, dx=1):
    """
    Use five-point central difference stencil to calculate the discrete derivative
    of a set of data.
    """
    dArr = np.zeros(len(xArr))

    # First and last points have to use a directional stencil
    dArr[0] = (xArr[0] - xArr[1]) / dx
    dArr[-1] = (xArr[-2] - xArr[-1]) / dx

    dArr[1] = (xArr[0] - xArr[2]) / (2*dx)
    dArr[-2] = (xArr[-3] - xArr[-1]) / (2*dx)

    # Now everything in between
    dArr[2:-2] = (xArr[:-4] - 8*xArr[1:-3] + 8*xArr[3:-1] - xArr[4:]) / (12*dx)

    return dArr
