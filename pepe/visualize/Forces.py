import numpy as np
import matplotlib.pyplot as plt

from pepe.visualize import visCircles, genRandomDistancedColors


def visForces(forceArr, alphaArr, betaArr, fps=None):

    fig, ax = plt.subplots(1, 3, figsize=(12,3))
    
    if len(forceArr) == 0:
        return fig, ax

    tArr = np.arange(len(forceArr[0]))
   
    if fps is None:
        fps = 1

    for i in range(len(tArr)):
        ax[0].plot(tArr/fps, forceArr[i])
        ax[1].plot(tArr/fps, alphaArr[i], 'o')
        ax[2].plot(tArr/fps, betaArr[i])

    ax[0].set_ylabel('Force [N]')
    ax[1].set_ylabel('Alpha [rad]')
    ax[2].set_ylabel('Beta [rad]')

    for i in range(3):
        if fps == 1:
            ax[i].set_xlabel('Time [step]')
        else:
            ax[i].set_xlabel('Time [s]')

    fig.tight_layout()

    return fig, ax 


def visContacts(center, radius, betaArr, alphaArr=None, forceArr=None, ax=None, setBounds=False, circleColor=None, forceColors=None, drawCircle=False):

    npCenter = np.array(center)

    if ax is None:
        fig, ax = plt.subplots()

    if forceColors is None:
        colors = genRandomDistancedColors(len(betaArr), 1000)
    else:
        colors = forceColors
    
    if drawCircle:
        visCircles([npCenter], [radius], ax, setBounds=setBounds, colors=circleColor)

    if len(betaArr) == 0:
        return ax

    if forceArr is None:
        weighting = np.zeros(len(betaArr)) + radius/4
    else:
        weighting = np.array(forceArr) * radius/4

    if alphaArr is None:
        for i in range(len(betaArr)):
            contactPoint = npCenter + .95*radius * np.array([np.cos(betaArr[i]), np.sin(betaArr[i])])
            ax.scatter([contactPoint[1]], [contactPoint[0]], c=colors[i])
            #c = plt.Circle(contactPoint[::-1], weighting[i], color='red', fill=False, linewidth=1)
            #ax.add_artist(c)
    else:
        for i in range(len(betaArr)):
            if not np.isnan(alphaArr[i]):
                contactPoint = npCenter + radius * np.array([np.cos(betaArr[i]), np.sin(betaArr[i])])
                point1 = contactPoint + weighting[i] * np.array([np.cos(betaArr[i] + alphaArr[i]), np.sin(betaArr[i] + alphaArr[i])])
                point2 = contactPoint - weighting[i] * np.array([np.cos(betaArr[i] + alphaArr[i]), np.sin(betaArr[i] + alphaArr[i])])
                ax.plot([point1[1], point2[1]], [point1[0], point2[0]], linewidth=5, c=colors[i])

    return ax


def fullVisContacts(outputDir, centerArr, radiusArr, betaArr, alphaArr=None, forceArr=None, forceColors=None, circleColors=None, startIndex=0, imageSize=(1024, 1280), fps=25):

    if forceColors is None:
        forceColorArr = [genRandomColors(len(b), int(time.perf_counter()*1e6) % 1024) for b in betaArr]
    else:
        forceColorArr = forceColors

    if len(betaArr) == 0:
        return False

    # The list comprehension is to make sure that we index a particle that actually has forces acting
    # on it.
    tSteps = len([b for b in betaArr if len(b) > 0][0])
    # To save the image of each plot so we can create a gif at the end
    images = [None for i in range(tSteps)]

    for i in range(tSteps):
        clear_output(wait=True)
        fig, ax = plt.subplots()
        
        visCircles([centerArr[p][i] for p in range(len(centerArr))], [radiusArr[p][i] for p in range(len(radiusArr))], ax=ax)

        for particleIndex in range(len(betaArr)):
            visContacts(centerArr[particleIndex][i], radiusArr[particleIndex][i],
                        betaArr[particleIndex][:,i], ax=ax, forceColors=forceColors[particleIndex])#, alphaArr=alphaArr[particleIndex][:,i])
            
        ax.set_xlim([0, imageSize[1]])
        ax.set_ylim([0, imageSize[0]])
        ax.set_aspect('equal')
        ax.set_title(f'Frame {startIndex + i}')
        
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        images[i] = Image.frombytes('RGB', canvas.get_width_height(), 
                     canvas.tostring_rgb())
        plt.close(fig)

    images[0].save(outputDir + 'contact_tracking.gif', save_all=True, append_images=images[1:], duration=fps, optimize=True, loop=True)
