"""
Methods to visualize the positions of particles.
"""
import numpy as np
import matplotlib.pyplot as plt

from pepe.visualize import genColors, checkColorType

def visCircles(centers, radii, ax=None, color=None, sameColors=False, annotations=None, setBounds=False, linewidth=1):
    """
    Draw circles on an axis.

    Parameters
    ----------

    centers : iterable[N,2] of float
        List of particle centers [y,x] (in pixels).

    radii : float or iterable[N] of float
        List of particle radii (in pixels).

    ax : matplotlib axis or None
        Axis to draw the circles on. If `None`, a new
        axis will be created.

    color : iterable of color type (str, hex, etc.) or single color type
        Color of the perimeter of the circles. Can be 
        an iterable of colors allowing for different colors
        for each circle. Can also be a list of values, which will
        be used to create a color map.

    sameColors : bool
        If `color=None`, then colors will be chosen automatically;
        these can either be all the same color (`sameColors=True`)
        or all distinct colors (`sameColors=False`).

        Only relevant if `color=None`.

    annotations : iterable
        Annotations/labels for each point to be drawn next
        to the corresponding circle. Can be any type that
        can be converted to a string use `str(*)`.

    setBounds : bool
        Whether to set the boundaries of the axes based on the
        positions of the circles (`setBounds=True`) or to leave
        them as default (`setBounds=False`).

        For example, if drawing circles on top of an image,
        you should use `setBounds=False`, since the image will
        take care of making sure everything is shown, but if you
        are just drawing circles on an empty canvas (for example,
        if `ax=none` and you aren't drawing anything afterwards)
        then you should use `setBounds=True`.

    lineWidth : float
        The width of the perimeter of the circles.

    Returns
    -------
    ax : matplotlib axis
        
    """
    # Cut out nan values
    # It is assumed that both of the center and the radius will be nan
    if hasattr(radii, '__iter__'):
        goodIndices = np.array([0 if np.isnan(r) else 1 for r in radii])
        npRadii = np.array([r for r in radii if not np.isnan(r)])
    else:
        goodIndices = np.ones(len(centers), dtype=np.uint8)
        npRadii = np.repeat(radii, len(centers))
    
    npCenters = np.array([c for c in centers if not np.isnan(c[0])])

    if ax is None:
        fig, ax = plt.subplots()

    if color is None:
        if sameColors:
            singleColor = genColors(1)[0]
            circleColorsList = [singleColor for _ in range(len(npCenters))]
        else:
            circleColorsList = genColors(len(npCenters))
    # We can't just check if color has the attribute '__iter__'
    # since a string will have that, and a string could just be
    # a single color
    elif not type(color) in [list, np.ndarray]:
        # If a single color is given, we want to repeat that
        circleColorsList = [color for _ in range(len(npCenters))]
    else:
        circleColorsList = color

    for i in range(len(npCenters)):
        c = plt.Circle(npCenters[i,::-1], npRadii[i], color=checkColorType(circleColorsList[i]), fill=False, linewidth=linewidth)
        ax.add_artist(c)

    if annotations is not None:
        if len(annotations) != len(centers):
            print('Warning: Invalid number of annotations provided! Ignoring...')
            pass
        else:
            for i in range(len(npCenters)):
                # The annotation array might be mismatched if we had any nan circles
                # so we have to account for that
                ax.text(npCenters[i,1], npCenters[i,0], annotations[i + int(np.sum(1 - goodIndices[:i]))], ha='center', va='center', color=circleColorsList[i])

    if setBounds:
        ax.set_xlim([np.min(npCenters[:,1])-1.5*np.max(npRadii), np.max(npCenters[:,1])+1.5*np.max(npRadii)])
        ax.set_ylim([np.min(npCenters[:,0])-1.5*np.max(npRadii), np.max(npCenters[:,0])+1.5*np.max(npRadii)])
        ax.set_aspect('equal')

    return ax
