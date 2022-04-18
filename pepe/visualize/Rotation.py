"""
Methods to visualize rotational motion of a particle.
"""
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from pepe.preprocess import checkImageType
from pepe.visualize import genColors, visCircles

def visRotation(imageFilePaths, centerArr, radiusArr, angleArr, outputFolderPath='./', offset=(0,0), barWidth=2.5, outputFileName='particle_orientation.gif'):
    """
    Create a gif of the tracked (position and orientation) particles
    overlaid on the original images.
    """

    # Where we will store each drawn image/figure, to be made into
    # a gif afterwards
    imageArr = [None]*len(imageFilePaths)

    colorArr = genColors(len(centerArr))

    for i in range(len(imageFilePaths)):
        
        fig, ax = plt.subplots()

        image = checkImageType(imageFilePaths[i])
        ax.imshow(image)

        # Draw the circles
        visCircles([centerArr[p][i] + np.array(offset) for p in range(len(centerArr))], [radiusArr[p][i] for p in range(len(radiusArr))], ax=ax, colors=colorArr)

        # Now we want to draw a bar in the center of each circle to
        # visualize the rotation
        for p in range(len(centerArr)):
            # Left end of the line
            point1 = centerArr[p][i] + np.array(offset) - np.array([np.sin(angleArr[p][i]), np.cos(angleArr[p][i])]) * radiusArr[p][i] / 2
            # Right end of the line
            point2 = centerArr[p][i] + np.array(offset) + np.array([np.sin(angleArr[p][i]), np.cos(angleArr[p][i])]) * radiusArr[p][i] / 2

            # Now actually draw the line
            plt.plot([point1[1], point2[1]], [point1[0], point2[0]], linewidth=barWidth, c=colorArr[p])


        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        imageArr[i] = Image.frombytes('RGB', canvas.get_width_height(), 
                                    canvas.tostring_rgb())

        plt.close(fig)


    imageArr[0].save(outputFolderPath + outputFileName, save_all=True,
                                append_images=imageArr[1:], duration=25, optimize=True, loop=True)

    return

