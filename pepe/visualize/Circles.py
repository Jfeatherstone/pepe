"""
Methods to visualize the positions of particles.
"""
import numpy as np
import matplotlib.pyplot as plt

from pepe.visualize import genColors

def visCircles(centers, radii, ax=None, colors=None, annotations=None, sameColors=False, setBounds=False, linewidth=1):
    """
    Draw circles on an axis.

    Parameters
    ----------

    centers : np.ndarray[N,2] or list[N,2]
        List of particle centers [y,x] (in pixels).

    radii : np.ndarray[N] or list[N]
        List of particle radii (in pixels).
    """
    # Cut out nan values
    goodIndices = np.array([0 if np.isnan(r) else 1 for r in radii])
    npRadii = np.array([r for r in radii if not np.isnan(r)])
    npCenters = np.array([c for c in centers if not np.isnan(c[0])])

    if ax is None:
        fig, ax = plt.subplots()

    if colors is None:
        if sameColors:
            singleColor = genColors(1)[0]
            circleColorsList = [singleColor for _ in range(len(npCenters))]
        else:
            circleColorsList = genColors(len(npCenters))

    elif not type(colors) is list:
        # If a single color is given, we want to repeat that
        circleColorsList = [colors for _ in range(len(npCenters))]

    else:
        circleColorsList = colors

    for i in range(len(npCenters)):
        c = plt.Circle(npCenters[i,::-1], npRadii[i], color=circleColorsList[i], fill=False, linewidth=linewidth)
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
