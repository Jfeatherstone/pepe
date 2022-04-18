"""
Coordinate transformation from discrete cartesian space to discrete polar space.
"""
import numpy as np

# I honestly don't think this will ever be used, but can't hurt to
# keep it around
def cartesianToPolar(mat, center=None, fillValue=np.nan):
    """
    Converts a discretized matrix in cartesian coordinates to polar
    coordinates.
    """
    
    if center is None:
        centerIndex = np.array([(mat.shape[0] - 1)/2, (mat.shape[1] - 1)/2])
        
    # First, we have to calculate the radii and angles    
    yArr = np.float64(np.arange(mat.shape[0])) - centerIndex[0]
    xArr = np.float64(np.arange(mat.shape[1])) - centerIndex[1]
    
    radiusArr = np.add.outer(yArr**2, xArr**2)
    angleArr = np.arctan2.outer(yArr, xArr)
    
    # Now we have to discretize these values, since they will become our new axes
    
    # Calculate what the smallest change in radius is
    dR = np.sort(np.unique(radiusArr.flatten()))[:2]
    dR = dR[1] - dR[0]
    print(dR)
    
    radiusIndices = np.int64(radiusArr / dR)
    
    # Calculate what the smallest change in angle we could possible have is
    # For this, we assume the farthest point from the center is in the bottom right
    # corner
    dTheta = abs(angleArr[-1,-1] - angleArr[-1,-2])
    print(dTheta)
    # Now make sure we are on the domain [0, 2pi] instead of [-pi, pi]
    angleArr += np.pi
    angleIndices = np.int64(angleArr / dTheta)

    minRI = np.min(radiusIndices.flatten())
    minThetaI = np.min(angleIndices.flatten())
    
    # Build the array, which may not be the same size as the previous array
    polarMat = np.zeros((np.max(radiusIndices.flatten()) - minRI + 1,
                         np.max(angleIndices.flatten()) - minThetaI + 1))
    
    polarMat[:,:] = fillValue
    
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            polarMat[radiusIndices[i,j] - minRI, angleIndices[i,j] - minThetaI] = mat[i,j]
            
    return polarMat, (dR, dTheta)
