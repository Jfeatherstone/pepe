import inspect
import numpy as np

from pepe.auto import forceSolve, forceSolveArgDTypes

def test_forceSolve_DTypeCoverage():
    """
    Make sure that all of the dtypes for the arguments in
    pepe.auto.forceSolve are defined for the purpose of
    reading in values from a settings file.
    """
    args, counts= np.unique(list(inspect.signature(forceSolve).parameters.keys()) + list(forceSolveArgDTypes.keys()), return_counts=True) 
    missingArgs = args[counts == 1]

    assert len(missingArgs) == 0, f"Missing data types for args: {missingArgs}"

