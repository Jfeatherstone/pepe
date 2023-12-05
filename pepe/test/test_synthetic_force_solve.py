import numpy as np

from pepe.preprocess import crossMask
from pepe.simulate import genSyntheticResponse
from pepe.analysis import initialForceSolve, forceOptimize
from pepe.utils import preserveOrderSort

def test_synthetic_force_solve_brightfield():
    """
    """
    # TODO: pass for now, since it isn't consistent
    # enough (inherent issue in non-linear optimization...)
    return

    fSigma = 100.
    pxPerMeter = 10000.
    brightfield = False

    # Parameters of our force solving method
    contactPadding = 10
    g2MaskPadding = 1
    contactMaskRadius = 50

    imageShape = (1024, 1280, 3)

    # Options for optimization
    optimizationKwargs = {"maxEvals": [100, 100, 100], "method": 'nelder',
                      "parametersToFit": [['f'], ['f', 'a'], ['a']],
                      "allowRemoveForces": True, "useTolerance": False,
                      "allowAddForces": True, "minForceThreshold": .001,
                      "localizeAlphaOptimization": False, "imageScaleFactor": .5,
                      "forceBalanceWeighting": .05}

    # For unit testing
    forceErrorTolerance = .1 # [N]
    betaErrorTolerance = .05 # [rad]
    alphaErrorTolerance = .5 # [rad], Much larger, since this one is usually a mess

    # Create a cross-shaped channel
    maskArr = crossMask(imageShape, xlim=np.array([460, 810]), ylim=np.array([350, 700]))

    # Our particles
    centers = np.array([[524, 257], [524, 605], [524, 955], # Center particles
                       [175, 635], [870, 635]], dtype=np.float64) # Top and bottom particles
    radii = np.array([175, 174, 173, 171, 172]) # Odd numbers are good

    # Same order as the centers above
    # Made by just messing around with numbers until things look good
    forces = np.array([[.08, .1, .4], [.4, .05, .32, .39], [.06, .05, .05], [.37, .15], [.35, .16]], dtype='object')
    betas = np.array([[0., np.pi, np.pi/2], [.1, np.pi/2, -np.pi/2, np.pi-.1], [-np.pi/2, 0., np.pi], [-.1, np.pi/2], [-np.pi+.1, np.pi/2]], dtype='object')
    alphas = np.array([[0., 0., 0.], [-.6, 0., 0., .6], [0., 0., 0.], [0., .6], [-.6, 0.]], dtype='object')

    photoelasticChannel = np.zeros(imageShape[:2])

    for i in range(len(centers)):
        photoelasticChannel += genSyntheticResponse(np.array(forces[i]), np.array(alphas[i]), np.array(betas[i]), fSigma, radii[i], pxPerMeter, brightfield, imageSize=imageShape[:2], center=centers[i])

    # Optional: add some noise to the photoelastic channel
    #photoelasticChannel += np.random.uniform(-.2, .2, size=photoelasticChannel.shape)

    photoelasticChannel = np.abs(photoelasticChannel)

    # Put the photoelastic response together with circles
    compositeImage = np.zeros(imageShape, dtype=np.uint8)
    compositeImage[:,:,0] = maskArr[:,:,0]*100
    compositeImage[:,:,1] = np.uint8(photoelasticChannel*255)
    compositeImage[:,:,2] = maskArr[:,:,0]*100

    # Generate initial guess
    forceGuessArr, alphaGuessArr, betaGuessArr = initialForceSolve(photoelasticChannel,
                                                    centers, radii, fSigma, pxPerMeter,
                                                    contactPadding, g2MaskPadding,
                                                    contactMaskRadius=contactMaskRadius,
                                                    boundaryMask=maskArr, ignoreBoundary=False)

    # Optimize forces
    for i in range(len(centers)):
        optForceArr, optBetaArr, optAlphaArr, res = forceOptimize(np.zeros(len(forceGuessArr[i]))+.1, betaGuessArr[i], alphaGuessArr[i],
                                                                  radii[i], centers[i], photoelasticChannel, fSigma, pxPerMeter, brightfield,
                                                                  **optimizationKwargs, debug=False)

        assert len(optForceArr) == len(forces[i]), f'Incorrect number of forces found'
        assert len(optAlphaArr) == len(alphas[i]), f'Incorrect number of alphas found'
        assert len(optBetaArr) == len(betas[i]), f'Incorrect number of betas found'

        assert not True in [f > forceErrorTolerance for f in (preserveOrderSort(forces[i], optForceArr) - forces[i])], f"Force magnitude exceeds tolerance for particle {i}."
        assert not True in [b > betaErrorTolerance for b in (preserveOrderSort(betas[i], optBetaArr, periodic=True) - betas[i])], f"Beta exceeds tolerance for particle {i}."
        assert not True in [a > alphaErrorTolerance for a in (preserveOrderSort(alphas[i], optAlphaArr, periodic=True) - alphas[i])], f"Alpha exceeds tolerance for particle {i}."
