import inspect
import numpy as np

from pepe.auto import forceSolve, forceSolveArgDTypes, excludedArgs

def test_forceSolve_DTypeCoverage():
    """
    Make sure that all of the dtypes for the arguments in
    pepe.auto.forceSolve are defined for the purpose of
    reading in values from a settings file.
    """
    args, counts= np.unique(list(inspect.signature(forceSolve).parameters.keys()) + list(forceSolveArgDTypes.keys()), return_counts=True) 
    missingArgs = [a for a in args[counts == 1] if a not in excludedArgs]
    
    assert len(missingArgs) == 0, f"Missing data types for args: {missingArgs}"

