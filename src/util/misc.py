'''
Created on 4 Jul 2014

@author: bryanfeeney
'''
import sys
import numpy as np

def printStderr(msg):
    sys.stdout.flush()
    sys.stderr.write(msg + '\n')
    sys.stderr.flush()
    

def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate

def constantArray(shape, defaultValue):
    '''
    Return an np array with the given shape (any tuple will do) set to the given
    value at every dimension.
    '''
    result = np.ndarray(shape=shape, defaultValue)
    result.fill(defaultValue)
    return result


def converged (boundIters, boundValues, bvIdx, epsilon, minIters = 100):
    '''
    Returns true if we've converged. To measure convergence we consider the angle between
    the last two bound measurements, and a horizontal plane, and if this angle (measured
    in degrees) is less than epsilon, then we've converged
    
    Params:
    boundIters  - the iterations at which the bound was measured each time.
    boundValues - the list of bound value measurements, as a numpy array
    bvIdx       - points to the position in which the *next* bound value will be stored
    epsilon     - the minimum angle, in degrees, that the bound vales must attain
    minBvIdx    - how many iterations do we allow to proceed before we start checking for
                  convergence
    
    Return:
    True if converged, false otherwise
    '''
    if (bvIdx < 2) or (boundIters[bvIdx - 1] < minIters):
        return False
    
    opposite = boundValues[bvIdx - 1] - boundValues[bvIdx - 2]
    adjacent = boundIters[bvIdx - 1] - boundIters[bvIdx - 2]
    angle = np.degrees (np.arctan(opposite / adjacent))
    
    return angle < epsilon
    

def clamp (array1, array2, array3, length):
    '''
    Clamp the arrays to the given length
    '''
    inputs = [array1, array2, array3]
    outputs = []
    for inArr in inputs:
        outArr = np.empty(shape=(length,), dtype=inArr.dtype)
        outArr[:] = inArr[:length]
        outputs.append (outArr)
    
    return outputs[0], outputs[1], outputs[2]