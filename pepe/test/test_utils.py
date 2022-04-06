import numpy as np

from pepe.utils import *

"""
Parse.py
"""

def test_ParseStrList():
    assert parseList("['a', 'b', 'c']", str) == ['a', 'b', 'c']

def test_ParseIntList():
    assert parseList("[1, 2, 3, 4]", int) == [1, 2, 3, 4]

def test_ParseNestedStrList(): 
    assert parseList("[['a', 'b'], ['b', 'c'], ['c']]") == [['a', 'b'], ['b', 'c'], ['c']]

def test_ParseNestedIntList():
    assert parseList("[[1, 2], [3], [4, 5]]") == [[1, 2], [3], [4, 5]]

"""
Sort.py
"""

def test_POASOneDim():
    # len(new) == len(old)
    new = [2.1, 1.1, 4.1, 5.1, 3.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new) == [1, 4, 0, 2, 3]

def test_POASOneDimPad():
    # len(new) < len(old), pad
    new = [2.1, 1.1, 4.1, 5.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=True) == [1, None, 0, 2, 3]

def test_POASOneDimPad_2():
    # len(new) > len(old), pad
    new = [2.1, 1.1, 4.1, 5.1]
    old = [2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=True) == [0, 2, 3, 1]

def test_POASOneDimNoPad():
    # len(new) < len(old), no pad
    new = [2.1, 1.1, 4.1, 5.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=False) == [1, 0, 2, 3]

def test_POASOneDimNoPad_2():
    # len(new) > len(old), no pad
    new = [2.1, 1.1, 4.1, 5.1]
    old = [2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=False) == [0, 2, 3, 1]

def test_POASOneDimPadDistance():
    # len(new) < len(old), pad, distance
    new = [2.1, 1.1, 4.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=True, maxDistance=.5) == [1, None, 0, 2, None]

def test_POASTwoDim():
    # len(new) == len(old)
    new = [[1, 2], [3, 3], [5, 4]]
    old = [[.9, 2], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new) == [0, 2, 1]

def test_POASTwoDimPad():
    # len(new) < len(old), pad
    new = [[3, 3], [5, 4]]
    old = [[.9, 2], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=True) == [None, 1, 0]

def test_POASTwoDimNoPad():
    # len(new) < len(old), no pad
    new = [[3, 3], [5, 4]]
    old = [[.9, 2], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=False) == [1, 0]

def test_POASTwoDimPadDistance():
    # len(new) > len(old), pad, distance
    new = [[3, 3], [5, 4], [7, 7], [9, 10], [11, 11]]
    old = [[.9, 2], [4.9, 4.1], [2.9, 2.9], [10.9, 10.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=True, maxDistance=.5) == [None, 1, 0, 4, 2, 3]

def test_POASTwoDimNanFill():
    # len(new) < len(old), pad, distance, fill
    new = [[3, 3], [5, 4], [9, 9]]
    old = [[.9, 2], [np.nan, np.nan], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=True, maxDistance=.5, fillNanSpots=True) == [None, 2, 1, 0]

def test_POASTwoDimNanNoFill():
    # len(new) < len(old), pad, distance, no fill
    new = [[3, 3], [5, 4], [9, 9]]
    old = [[.9, 2], [np.nan, np.nan], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=True, maxDistance=.5, fillNanSpots=False) == [None, None, 1, 0, 2]
