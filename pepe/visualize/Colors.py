"""
Methods to generate good color schemes for visualization.
"""
import numpy as np
import matplotlib.colors as mcolors
import colour

from sklearn.neighbors import KDTree

# Some manually selected colors that are good
# Taken from xkcd survey: https://blog.xkcd.com/2010/05/03/color-survey-results/
MANUAL_COLORS = ['#7E1E9C', # Purple
                '#15B01A', # Green
                '#0343DF', # Blue
                '#FF88C0', # Pink
                '#653700', # Brown
                '#E50000', # Red
                '#029386', # Teal
                '#F97306', # Orange
                '#033500', # Dark green
                '#00035B', # Dark blue
                '#650021', # Maroon
                '#BF77F6', # Light purple
                '#929591', # Gray
                '#6E750E', # Olive
                '#00FFFF', # Cyan
                ]


def checkColorType(c):
    """
    Will check whether `c` is a valid
    color to be used for plotting, visualization, etc.
    (eg. valid hex codes, colors defined by matplotlib, etc.)
    
    If the color is valid, it will be returned; if it is not
    valid, a color will be generated using colour.Color(pick_for=c)
    such that this method called with the same argument will return
    the same color.

    Parameters
    ----------

    c : color type (str, hex, etc.) or object
        Either a valid color, or an object for
        which a color should be chosen for.

    Returns
    -------

    color : valid color
    """
    if mcolors.is_color_like(c):
        return c

    return str(colour.Color(pick_for=c))


def rgbToHex(rgbC):
    return '%02x%02x%02x' % rgbC

def hexToRgb(hexC):
    return [int(hexC[i:i+2], 16) for i in (0, 2, 4)]

def genColors(size, seed=21):
    """
    Generate a set of colors for visualization by a hierarchy of methods:

    1. First, select from MANUAL_COLORS defined above.
    2. Use genRandomDistancedColors()
    3. Use genRandomColors() 
    """
    if size < len(MANUAL_COLORS):
        return [MANUAL_COLORS[i] for i in range(size)]
    
    else:
        # genRandomDistanceColors will already supplement with full random
        # colors if size is too big, so no need to deal with it here.
        return MANUAL_COLORS + genRandomDistancedColors(size - len(MANUAL_COLORS), seed)


def genRandomColors(size, seed=21):
    """
    Generate colors by randomly selecting rgb values.

    Parameters
    ----------

    size : int
        The number of colors to generate.

    seed : int
        The seed to use for random generation.


    Returns
    -------
    
    colors : list[size]
        A list of hex codes for colors.
    """
    np.random.seed(seed)

    randomColors = [f"#{rgbToHex(tuple(np.random.choice(range(256), size=3).flatten()))}" for i in range(size)]

    return randomColors

def genRandomDistancedColors(size, seed=21):
    """
    Generate colors by sampling from the list of CSS named colors,
    where the first color will be randomly selected, and each subsequent
    one will be chosen to maximize it's distance from previously selected
    colors in LAB color space (using a kd-tree).

    If the number of requested colors is greater than the number of named
    colors, the remaining ones will be randomly generated.

    Maybe a bit overkill...

    Parameters
    ----------

    size : int
        The number of colors to generate.

    seed : int
        The seed to use for random generation.


    Returns
    -------
    
    colors : list[size]
        A list of hex codes for colors.
    """
    allCSSColors = list(mcolors.CSS4_COLORS.values())

    if size > len(allCSSColors):
        allCSSColors += genRandomColors(size - len(allCSSColors), seed)
        return allCSSColors

    np.random.seed(seed)

    allCSSColorsLAB = np.array([rgbToLab(hexToRgb(c[1:])) for c in allCSSColors])

    # Randomly generate the starting one
    selection = np.random.randint(len(allCSSColors))
    addedIndices = [selection]
    colors = [allCSSColors[selection]]

    colorsLAB = np.zeros((size, 3))    
    colorsLAB[0] = allCSSColorsLAB[selection]

    kdTree = KDTree(allCSSColorsLAB, leaf_size=10)
    # For each other color, we want to select the furthest possible color from our current set
    for i in range(1, size):

        dist, ind = kdTree.query(colorsLAB[:i], k=len(allCSSColorsLAB))
        dist = np.array([dist[j,:][np.argsort(ind[j])] for j in range(len(dist))])
        fullCost = np.sum(dist, axis=0)
        selection = np.argsort(fullCost)[::-1]
        selection = [s for s in selection if not s in addedIndices]

        colors.append(allCSSColors[selection[0]])
        addedIndices.append(selection[0])
        colorsLAB[i] = allCSSColorsLAB[selection[0]]

    return colors


def maxDifferenceOrderColors(colors):
    """
    Sort a set of colors such that the difference between consecutive
    colors is maximized.

    A fun little application of the (inverse of the) travelling salesman problem :)
    """
    pass


# This method was taken from:
# https://gist.github.com/manojpandey/f5ece715132c572c80421febebaf66ae
# TODO: Clean it up
def rgbToLab(inputColor):

    num = 0
    RGB = [0, 0, 0]

    for value in inputColor:
        value = float(value) / 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        RGB[num] = value * 100
        num = num + 1

    XYZ = [0, 0, 0, ]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)

    # Observer= 2°, Illuminant= D65
    XYZ[0] = float(XYZ[0]) / 95.047         # ref_X =  95.047
    XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883

    num = 0
    for value in XYZ:

        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)

        XYZ[num] = value
        num = num + 1

    Lab = [0, 0, 0]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    Lab[0] = round(L, 4)
    Lab[1] = round(a, 4)
    Lab[2] = round(b, 4)

    return Lab
