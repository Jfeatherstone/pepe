"""
Common functional forms to fit to.
"""
import numpy as np

def cone(p, center, halfAngle, offset, roundness):
    """
    A mix between a cone and a gaussian.

    Parameters
    ----------

    p : [float, float]
        The coordinates [y,x] of the point to compute the height on the cone.

    center : [float, float]
        The coordinates [y,x] of the center of the cone.

    halfAngle : float
        The half angle (in radians) of the cone.

    offset : float
        The z coordinate of the tip of the cone.

    roundness : float
        The factor of roundness for the tip of the cone; 0 is a perfectly
        sharp cone, and values greater than 0 denote increasing roundness.

    Returns
    -------

    z : float
        The z position at the point `p` on the cone.
    """
    d = np.sqrt((center[0] - p[0])**2 + (center[1] - p[1])**2)

    # Otherwise we will have /0 error
    if d == 0:
        return offset

    return offset - d/np.tan(halfAngle) * np.exp(-(roundness)**2/d)


def lorentzian(p, center, width, amp, offset):
    r"""
    The 2D Lorentzian function, evaluated at a single point.

    Parameters
    ----------

    p : [float, float]
        The coordinates [y,x] of the point to compute the height on the Lorentzian.

    center : [float, float]
        The coordinates [y,x] of the center of the Lorentzian.
    
    width : float
        The width of the Lorentzian.

    amp : float
        The amplitude of the Lorentzian.

    offset : float
        The z coordinate of the tip of the Lorentzian.

    Returns
    -------

    z : float
        The z position at the point `p` on the Lorentzian.
    """
    return amp * 0.5 * width / ( ((center[0] - p[0])**2 + (center[1] - p[1])**2) + 0.25 * width**2) + offset
