"""
Methods to visualize forces acting on a particle.
"""
import numpy as np
import matplotlib.pyplot as plt

from pepe.visualize import visCircles, genRandomDistancedColors, genColors


def visForces(forceArr, alphaArr, betaArr, centerArr=None, angleArr=None, fps=None):
    """
    Visualize all of the forces acting on a single particle, by plotting
    the magnitudes, alphas, and betas (and optionally center position).

    Requires that `forceArr`, `alphaArr`, `betaArr`, and `centerArr` have
    indices of `forceArr[forceIndex][timeIndex]`, as would be given by
    rectangularizing via `pepe.utils.rectangularizeForceArrays()` and 
    indexing a single value in the first dimension.

    Examples
    --------

    ```
    # Solve for some forces
    forceArr, alphaArr, betaArr, centerArr, radiusArr = forceOptimize(...)

    # Particle index
    pI = 0

    fig, ax = visForces(forceArr[pI], alphaArr[pI], betaArr[pI], centerArr[pI])
    plt.show()
    ```

    Parameters
    ----------

    forceArr : np.ndarray[F,T]
        Array containing magnitudes of F forces for T timesteps.

    alphaArr : np.ndarray[F,T]
        Array containing alpha angles of F forces for T timesteps.

    betaArr : np.ndarray[F,T]
        Array containing beta angles of F forces for T timesteps.

    centerArr : np.ndarray[T,2] or None
        Array containing center position in form [y,x] of the particle for T timesteps.

    angleArr : np.ndarray[T] or None
        Array containing angles in radians of the particle for T timesteps.

    fps : float or None
        The number of frames per second of the capture video, used to convert the x-axis
        units from frame number to proper seconds.

    Returns
    -------

    fig : plt.figure()
        The figure object the quantities are plotted on.

    ax : plt.axis()
        The list of 3 (or 4, if centers are provided) axes that the quantities are plotted on.
    """

    fig, ax = plt.subplots(1, 3 + int(centerArr is not None) + int(angleArr is not None), figsize=(3.6*(3+int(centerArr is not None)+int(angleArr is not None)),4))
    
    if len(forceArr) == 0:
        return fig, ax

    tArr = np.arange(len(forceArr[0]))
   
    if fps is None:
        fps = 1
   
    if centerArr is not None:
        ax[-1 - int(angleArr is not None)].plot(tArr/fps, centerArr[:,1], label='X')
        ax[-1 - int(angleArr is not None)].plot(tArr/fps, centerArr[:,0], label='Y')
        ax[-1 - int(angleArr is not None)].set_ylabel('Position [px]')
        ax[-1 - int(angleArr is not None)].legend()

    if angleArr is not None:
        # The weird ax indexing is to make sure it works regardless of
        # whether centerArr is passed or not
        ax[-1].plot(tArr/fps, angleArr)
        ax[-1].set_ylabel('Angle [rad]')

    for i in range(len(forceArr)):
        ax[0].plot(tArr/fps, forceArr[i])
        ax[1].plot(tArr/fps, alphaArr[i], 'o')
        ax[2].plot(tArr/fps, betaArr[i], 'o')

    ax[0].set_ylabel('Force [N]')
    ax[1].set_ylabel('Alpha [rad]')
    ax[2].set_ylabel('Beta [rad]')

    for i in range(3 + int(centerArr is not None) + int(angleArr is not None)):
        if fps == 1:
            ax[i].set_xlabel('Time [frame]')
        else:
            ax[i].set_xlabel('Time [s]')

    fig.tight_layout()

    return fig, ax 


def visContacts(center, radius, betaArr, alphaArr=None, forceArr=None, ax=None, setBounds=False, circleColor=None, forceColors=None, drawCircle=False):
    """
    Visualize the contacts for a system of particles, indicating either positions
    of contacts, or positions and contact angles (if `alphaArr` is provided).

    Returns
    -------

    ax : plt.axis()
    """

    npCenter = np.array(center)

    if ax is None:
        fig, ax = plt.subplots()

    if forceColors is None:
        colors = genColors(len(betaArr), 1000)
    elif type(forceColors) is str:
        colors = [forceColors for i in range(len(betaArr))]
    else:
        colors = forceColors
    
    if drawCircle:
        visCircles([npCenter], [radius], ax, setBounds=setBounds, colors=colors[0])

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
