import numpy as np

from pepe.utils import *

"""
Parse.py
"""

def test_parseList_StrList():
    assert parseList("['a', 'b', 'c']", str) == ['a', 'b', 'c']

def test_parseList_IntList():
    assert parseList("[1, 2, 3, 4]", int) == [1, 2, 3, 4]

def test_parseList_NestedStrList(): 
    assert parseList("[['a', 'b'], ['b', 'c'], ['c']]") == [['a', 'b'], ['b', 'c'], ['c']]

def test_parseList_NestedIntList():
    assert parseList("[[1, 2], [3], [4, 5]]") == [[1, 2], [3], [4, 5]]

"""
Sort.py
"""

def test_preserveOrderArgsort_OneDim():
    # len(new) == len(old)
    new = [2.1, 1.1, 4.1, 5.1, 3.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new) == [1, 4, 0, 2, 3]

def test_preserveOrderArgsort_OneDimPad():
    # len(new) < len(old), pad
    new = [2.1, 1.1, 4.1, 5.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=True) == [1, None, 0, 2, 3]

def test_preserveOrderArgsort_OneDimPad_2():
    # len(new) > len(old), pad
    new = [2.1, 1.1, 4.1, 5.1]
    old = [2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=True) == [0, 2, 3, 1]

def test_preserveOrderArgsort_OneDimNoPad():
    # len(new) < len(old), no pad
    new = [2.1, 1.1, 4.1, 5.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=False) == [1, 0, 2, 3]

def test_preserveOrderArgsort_OneDimNoPad_2():
    # len(new) > len(old), no pad
    new = [2.1, 1.1, 4.1, 5.1]
    old = [2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=False) == [0, 2, 3, 1]

def test_preserveOrderArgsort_OneDimPadDistance():
    # len(new) < len(old), pad, distance
    new = [2.1, 1.1, 4.1]
    old = [1, 3, 2, 4, 5]
    assert preserveOrderArgsort(old, new, padMissingValues=True, maxDistance=.5) == [1, None, 0, 2, None]

def test_preserveOrderArgsort_TwoDim():
    # len(new) == len(old)
    new = [[1, 2], [3, 3], [5, 4]]
    old = [[.9, 2], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new) == [0, 2, 1]

def test_preserveOrderArgsort_TwoDimPad():
    # len(new) < len(old), pad
    new = [[3, 3], [5, 4]]
    old = [[.9, 2], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=True) == [None, 1, 0]

def test_preserveOrderArgsort_TwoDimNoPad():
    # len(new) < len(old), no pad
    new = [[3, 3], [5, 4]]
    old = [[.9, 2], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=False) == [1, 0]

def test_preserveOrderArgsort_TwoDimPadDistance():
    # len(new) > len(old), pad, distance
    new = [[3, 3], [5, 4], [7, 7], [9, 10], [11, 11]]
    old = [[.9, 2], [4.9, 4.1], [2.9, 2.9], [10.9, 10.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=True, maxDistance=.5) == [None, 1, 0, 4, 2, 3]

def test_preserveOrderArgsort_TwoDimNanFill():
    # len(new) < len(old), pad, distance, fill
    new = [[3, 3], [5, 4], [9, 9]]
    old = [[.9, 2], [np.nan, np.nan], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=True, maxDistance=.5, fillNanSpots=True) == [None, 2, 1, 0]

def test_preserveOrderArgsort_TwoDimNanNoFill():
    # len(new) < len(old), pad, distance, no fill
    new = [[3, 3], [5, 4], [9, 9]]
    old = [[.9, 2], [np.nan, np.nan], [4.9, 4.1], [2.9, 2.9]]
    assert preserveOrderArgsort(old, new, padMissingValues=True, maxDistance=.5, fillNanSpots=False) == [None, None, 1, 0, 2]


"""
Outer.py
"""

def test_outerSubtract_2x2():
    a = np.array([1, 2])
    b = np.array([2, 3])
    assert np.sum(outerSubtract(a, b) - np.array([[-1, -2], [0, -1]])) == 0

def test_outerSubtract_3x3():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    assert np.sum(outerSubtract(a, b) - np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])) == 0


"""
RectArr.py
"""

def test_rectArr_FakeData():
    cArr = [[[2, 1], [9, 15]], # a, b
           [[10, 15], [3, 2]], # b, a
           [[4, 2], [10, 14]]] # a, b

    rArr = [[5, 5], [6, 5], [5, 6]]

    fArr = [[[.1, .5], [.3]],
           [[.4, .1], [.45, .05]],
           [[.43], [.46, .2]]]

    aArr = [[[.02, .01], [.2]],
           [[.25, .01], [.05, .04]],
           [[.03], [.26, .02]]]

    bArr = [[[0, 3], [1.5]],
           [[1.52, 0], [2.99, 0.02]],
           [[2.95], [1.55, .02]]]

    rfArr, raArr, rbArr, rcArr, rrArr = rectangularizeForceArrays(fArr, aArr, bArr, cArr, rArr)

    assert len(rcArr) == 2
    assert np.sum(rcArr[0] - np.array([[2, 1], [3, 2], [4, 2]])) == 0
    assert np.sum(rcArr[1] - np.array([[9, 15], [10, 15], [10, 14]])) == 0

    assert np.sum(rrArr[0] - np.array([5, 5, 5])) == 0
    assert np.sum(rrArr[1] - np.array([5, 6, 6])) == 0
