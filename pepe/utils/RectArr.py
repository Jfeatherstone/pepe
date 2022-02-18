import numpy as np

def rectangularizeForceArrays(forceArr, alphaArr, betaArr, centerArr, radiusArr):
    """
    Take a given (triangular) set of forces, and make them consistent and rectangular
    arrays that can be easily sliced,
    """
    rectForceArr = []
    rectBetaArr = []
    rectAlphaArr = []

    # Scalar
    maxNumParticles = np.max([len(betaArr[i]) for i in range(len(betaArr))])
    numTimesteps = len(forceArr)
    # First, make the centers array look nice, which we then use to identify
    # particles
    rectCenterArr = np.zeros((maxNumParticles, numTimesteps, 2))
    rectRadiusArr = np.zeros((maxNumParticles, numTimesteps))
    
    # We have to initialize the first element so that we can then use the
    # preserveOrderSort function to make sure the identities stay
    # consistent
    rectCenterArr[:,0] = list(centerArr[0]) + [[np.nan, np.nan]]*(maxNumParticles - len(centerArr[0]))
    rectRadiusArr[:,0] = list(radiusArr[0]) + [np.nan]*(maxNumParticles - len(centerArr[0]))
    
    particleOrder = np.zeros((numTimesteps, maxNumParticles), dtype=np.int16)
    particleOrder[0] = np.arange(len(particleOrder[0]))
    particleExists = np.zeros((numTimesteps, maxNumParticles), dtype=np.int16)
    
    for i in range(1, numTimesteps):
        particleOrder[i] = preserveOrderArgsort(rectCenterArr[:,i-1][np.isnan(rectCenterArr[:,i-1])[:,0] == False], centerArr[i], padMissingValues=True)
        rectCenterArr[:,i] = [centerArr[i][particleOrder[i,j]] if particleOrder[i,j] is not None else [np.nan, np.nan] for j in range(len(particleOrder[i]))]
        rectRadiusArr[:,i] = [radiusArr[i][particleOrder[i,j]] if particleOrder[i,j] is not None else np.nan for j in range(len(particleOrder[i]))]
        
    # We now have linked the particles from frame to frame, and can
    # rectangularize the other quantities on a particle-by-particle basis
    for i in range(maxNumParticles):
        # The maximum number of forces this particle ever has
        maxNumForces = np.max([len(forceArr[j][particleOrder[j,i]]) for j in range(numTimesteps) if particleOrder[j,i] is not None])
        
        singleParticleForceArr = np.zeros((maxNumForces, numTimesteps))
        singleParticleBetaArr = np.zeros((maxNumForces, numTimesteps))
        singleParticleAlphaArr = np.zeros((maxNumForces, numTimesteps))

        singleParticleForceArr[:,0] = list(forceArr[0][particleOrder[0,i]]) + [np.nan]*(maxNumForces - len(forceArr[0][particleOrder[0,i]]))
        singleParticleBetaArr[:,0] = list(betaArr[0][particleOrder[0,i]]) + [np.nan]*(maxNumForces - len(betaArr[0][particleOrder[0,i]]))
        singleParticleAlphaArr[:,0] = list(alphaArr[0][particleOrder[0,i]]) + [np.nan]*(maxNumForces - len(alphaArr[0][particleOrder[0,i]]))
        
        for j in range(1, numTimesteps):
            particleIndex = particleOrder[j,i]
            order = preserveOrderArgsort(singleParticleBetaArr[:,j-1][np.isnan(singleParticleBetaArr[:,j-1]) == False], betaArr[j][particleIndex], padMissingValues=True)
            order = list(order) + [None]*(maxNumForces - len(order))
            singleParticleForceArr[:,j] = [forceArr[j][particleIndex][order[k]] if order[k] is not None else np.nan for k in range(len(order))]
            singleParticleBetaArr[:,j] = [betaArr[j][particleIndex][order[k]] if order[k] is not None else np.nan for k in range(len(order))]
            singleParticleAlphaArr[:,j] = [alphaArr[j][particleIndex][order[k]] if order[k] is not None else np.nan for k in range(len(order))]

        print(singleParticleBetaArr[:,-1])

        rectForceArr.append(singleParticleForceArr)
        rectAlphaArr.append(singleParticleAlphaArr)
        rectBetaArr.append(singleParticleBetaArr)
        
    return rectForceArr, rectAlphaArr, rectBetaArr, rectCenterArr, rectRadiusArr

