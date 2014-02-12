'''
Created on 1 Dec 2013

@author: bryanfeeney
'''

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport log
from libc.float cimport FLT_MIN, DBL_MIN


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def selfSoftDot_f8(double[:,:] mat):
    '''
    Considers the given matrix to be a collection of stacked row-vectors. 
    Returns the sum of the dot products of each row-vector and its 
    soft-max form.
    '''
    cdef:
        double total = 0.0
        double lse = 0.0
        int rows = mat.shape[0]
        int cols = mat.shape[1]
        int row = 0
        int col = 0
    
    with nogil:
        for row in range(rows):
            expSum = 0.0
            for col in range(cols):
                expSum += exp(mat[row,col])
        
            for col in range(cols):
                total += exp(mat[row,col]) * mat[row,col] / expSum
    
    return total

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def selfSoftDot_f4(float[:,:] mat):
    '''
    Considers the given matrix to be a collection of stacked row-vectors. 
    Returns the sum of the dot products of each row-vector and its 
    soft-max form.
    '''
    cdef:
        float total = 0.0
        float lse = 0.0
        int rows = mat.shape[0]
        int cols = mat.shape[1]
        int row = 0
        int col = 0
    
    with nogil:
        for row in range(rows):
            expSum = 0.0
            for col in range(cols):
                expSum += exp(mat[row,col])
        
            for col in range(cols):
                total += exp(mat[row,col]) * mat[row,col] / expSum
    
    return total


