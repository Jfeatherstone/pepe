import numpy as np
from scipy.stats import linregress

def determineLinearRegions(data, minLength=.1, minR2=.96, maxSlopeInterceptDiff=.75):
    """
    Determine regions of a plot that are approximately linear by performing
    linear least-squares on a rolling window.
    
    Parameters
    ----------
    
    data : array_like
        Data within which linear regions are to be identified
        
    minLength : int or float
        The minimum length of a linear segment, either as an
        integer number of indices, or as a float fraction of the
        overall data length.
        
    minR2 : float
        The minimum r-squared value for a region to be
        considered linear.
        
    maxSlopeInterceptDiff : float
        The float percentage difference allowed between slopes
        and intercepts of adjacent slices for them to be
        considered the same region.
        
    Returns
    -------
    
    regionIndices : np.ndarray[N,2]
        The start and end indices for the N detected regions.
        
    slopes : np.ndarray[N]
        The slope of each region.
        
    intercepts : np.ndarray[N]
        The intercept of each region.
    """
    if minLength < 1:
        minLinSteps = int(len(data)*minLength)
    else:
        minLinSteps = int(minLength)
        
    inLinearRegion = False
    linearRegions = []
    slopes = []
    intercepts = []
    
    # Perform least squares on a rolling window
    i = 0
    while i < len(data) - minLinSteps:
        xArr = np.arange(i, i+minLinSteps)
        slope, intercept, r2, p_value, std_err = linregress(xArr, data[i:i+minLinSteps])
        
        if np.abs(r2) > minR2:
            
            if inLinearRegion:
                # Calculate how different new slope is from old one
                if np.abs((np.mean(slopes[-1]) - slope) / np.mean(slopes[-1])) < maxSlopeInterceptDiff and np.abs((np.mean(intercepts[-1]) - intercept) / np.mean(intercepts[-1])) < maxSlopeInterceptDiff:
                    # This is still the same linear region, so we extend the bounds
                    linearRegions[-1][1] = i+minLinSteps
                    
                    # And average in the slopes and intercepts
                    slopes[-1] += [slope]
                    intercepts[-1] += [intercept]
                else:
                    # Otherwise, we have a new linear region, which we start
                    # at the end of the other one
                    i = linearRegions[-1][1]
                    inLinearRegion = False
                    continue

            else:
                # New linear region
                linearRegions.append([i, i+minLinSteps])
                slopes.append([slope])
                intercepts.append([intercept])
                
            inLinearRegion = True
        
        else:
            inLinearRegion = False
            
        i += 1
            
    slopes = np.array([np.mean(s) for s in slopes])
    intercepts = np.array([np.mean(inter) for inter in intercepts])
    return np.array(linearRegions), slopes, intercepts
