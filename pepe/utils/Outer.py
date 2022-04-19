"""
Outer products defined in a `numba`-friendly way.
"""
import numpy as np

import numba

@numba.njit(parallel=True, fastmath=True, cache=True)
def outerSubtract(x, y):
    """
    Perform outer subtraction of two arrays, x - y.

    Created because `numpy.subtract.outer` is not compatable with numba.

    Parameters
    ----------
    x : np.ndarray[N]
        First array.

    y : np.ndarray[M]
        Second array.

    Returns
    -------
    outer : np.ndarray[N,M]
        Matrix where the element [i,j] is x[i]-y[j].
    """
    outer = np.zeros((len(x), len(y)))
    for i in range(outer.shape[0]):
        for j in range(outer.shape[1]):
            outer[i,j] = x[i] - y[j]

    return outer

