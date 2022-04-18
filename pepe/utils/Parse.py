"""
String-to-list parsing used in reading settings files.
"""
import numpy as np
import ast

def parseList(string, dtype=object):
    """
    Takes in a string representation of a list
    and builds a proper list.
    """
    # Most of the heavy lifting is done by ast 
    builtList = ast.literal_eval(string)

    # Now we can use numpy to fix the type
    builtArr = np.array(builtList, dtype=dtype)

    # Now convert back to a list
    return list(builtList)
