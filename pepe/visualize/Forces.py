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
        weighting = np.array(forceArr) / np.max(forceArr) * radius / 4

    if alphaArr is None:
        for i in range(len(betaArr)):
            contactPoint = npCenter + .95*radius * np.array([np.cos(betaArr[i]), np.sin(betaArr[i])])
            ax.scatter([contactPoint[1]], [contactPoint[0]], c=colors[i])
            #c = plt.Circle(contactPoint[::-1], weighting[i], color='red', fill=False, linewidth=1)
            #ax.add_artist(c)
    else:
        for i in range(len(betaArr)):
            contactPoint = npCenter + radius * np.array([np.cos(betaArr[i]), np.sin(betaArr[i])])
            point1 = contactPoint + weighting[i] * np.array([np.cos(betaArr[i] + alphaArr[i]), np.sin(betaArr[i] + alphaArr[i])])
            point2 = contactPoint - weighting[i] * np.array([np.cos(betaArr[i] + alphaArr[i]), np.sin(betaArr[i] + alphaArr[i])])
            ax.plot([point1[1], point2[1]], [point1[0], point2[0]], linewidth=5, c=colors[i])

    return ax
