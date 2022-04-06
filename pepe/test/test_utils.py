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
    new = [2.1, 1.1, 4.1, 5.1, 3.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new) == [1, 4, 0, 2, 3]

def test_POASOneDimPad():
    new = [2.1, 1.1, 4.1, 5.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=True) == [1, 0, 2, 3, None]

def test_POASOneDimNoPad():
    new = [2.1, 1.1, 4.1, 5.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=False) == [1, 0, 2, 3]

def test_POASOneDimPadDistance():
    new = [2.1, 1.1, 4.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=True, maxDistance=.5) == [1, None, 0, 2, None]

def test_POASTwoDim():
    new = [[1, 2], [3, 3], [5, 4]]
    old = [[.9, 2], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new) == [0, 2, 1]

def test_POASTwoDimPad():
    new = [[3, 3], [5, 4]]
    old = [[.9, 2], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=True) == [None, 1, 0]

def test_POASTwoDimNoPad():
    new = [[3, 3], [5, 4]]
    old = [[.9, 2], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=False) == [1, 0]


