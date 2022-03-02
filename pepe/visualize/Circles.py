import numpy as np
import matplotlib.pyplot as plt

from pepe.visualize import genRandomColors, genRandomDistancedColors

def visCircles(centers, radii, ax=None, colors=None, annotations=None, sameColors=False, setBounds=False):
    """
    Draw circles on an axis.

    Parameters
    ----------

    centers : np.ndarray[N,2] or list[N,2]
        List of particle centers [y,x] (in pixels).

    radii : np.ndarray[N] or list[N]
        List of particle radii (in pixels).
    """
    npCenters = np.array(centers)

    if ax is None:
        fig, ax = plt.subplots()

    if colors is None:
        if sameColors:
            singleColor = genRandomDistancedColors(1)[0]
            circleColorsList = [singleColor for _ in range(len(centers))]
        else:
            circleColorsList = genRandomDistancedColors(len(centers))

    elif not type(colors) is list:
        # If a single color is given, we want to repeat that
        circleColorsList = [colors for _ in range(len(centers))]

    else:
        circleColorsList = colors

    for i in range(len(centers)):
        c = plt.Circle(npCenters[i,::-1], radii[i], color=circleColorsList[i], fill=False, linewidth=1)
        ax.add_artist(c)

    if annotations is not None:
        if len(annotations) != len(centers):
            print('Warning: Invalid number of annotations provided! Ignoring...')
            pass
        else:
            for i in range(len(centers)):
                ax.text(npCenters[i,1], npCenters[i,0], annotations[i], ha='center', va='center', color=circleColorsList[i])

    if setBounds:
        ax.set_xlim([np.min(npCenters[:,1])-1.5*np.max(radii), np.max(npCenters[:,1])+1.5*np.max(radii)])
        ax.set_ylim([np.min(npCenters[:,0])-1.5*np.max(radii), np.max(npCenters[:,0])+1.5*np.max(radii)])
        ax.set_aspect('equal')

    return ax
