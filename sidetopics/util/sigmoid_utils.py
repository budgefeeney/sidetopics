'''
Created on 1 Dec 2013

@author: bryanfeeney
'''

import os
import numpy as np
import sidetopics.util.sig_fast as compiled
import sys

WarnIfSlow=True

def rowwise_softmax (matrix, out=None):
    '''
    Assumes each row of the given matrix is an unnormalized distribution and
    uses the softmax metric to normalize it. This additionally uses some
    scaling to ensure that we never overflow.
    '''
    if out is None:
        out = np.ndarray(shape=matrix.shape, dtype=matrix.dtype)
    
    row_maxes = matrix.max(axis=1) # Underflow makes sense i.e. Pr(K=k) = 0. Overflow doesn't, i.e Pr(K=k) = \infty
    np.exp(matrix - row_maxes[:, np.newaxis], out=out)
    out /= out.sum(axis=1)[:,np.newaxis]
    return out

def colwise_softmax (matrix, out=None):
    '''
    Assumes each row of the given matrix is an unnormalized distribution and
    uses the softmax metric to normalize it. This additionally uses some
    scaling to ensure that we never overflow.
    '''
    if out is None:
        out = np.ndarray(shape=matrix.shape, dtype=matrix.dtype)

    col_maxes = matrix.max(axis=0) # Underflow makes sense i.e. Pr(K=k) = 0. Overflow doesn't, i.e Pr(K=k) = \infty
    np.exp(matrix - col_maxes[np.newaxis, :], out=out)
    out /= out.sum(axis=0)[np.newaxis, :]
    return out

def lse(matrix):
    '''
    The log-sum-exp function. For each _row_ in the matrix, calculate the
    log of the sum of the exponent of the values, i.e.
    
    np.log(np.sum(np.exp(matrix), axis=1))
    
    The result is a vector of log-sum-exp values. 
    
    Note that unlike rowwise_softmax, this DOES NOT uses any scaling
    tricks to avoid overflow.
    '''
    return np.log(np.sum(np.exp(matrix), axis=1))

def selfSoftDot(matrix):
    '''
    Considers the given matrix to be a collection of stacked row-vectors. 
    Returns the sum of the dot products of each row-vector and its 
    soft-max form.
    
    This words on DENSE matrices only, and it appears in this module simply
    for convenience.
    
    Uses fast, memory-efficient operations for matrices of single
    and double-precision numbers, uses fast-ish numpy code as a
    fallback, but at the cost of creating a copy of of the matrix.
    '''
    if matrix.dtype == np.float64:
        return compiled.selfSoftDot_f8(matrix)
    elif matrix.dtype == np.float32:
        return compiled.selfSoftDot_f4(matrix)
    
    if WarnIfSlow:
        sys.stderr.write("WARNING: Slow code path triggered (selfSoftDot)")
    return np.sum(matrix * rowwise_softmax(matrix))

def scaledSelfSoftDot(matrix, scale):
    '''
    Considers the given matrix to be a collection of stacked row-vectors. 
    Returns the weighted sum of the dot products of each row-vector and its 
    soft-max form, where the weights are given by the scale parameter.
    
    This words on DENSE matrices only, and it appears in this module simply
    for convenience.
    
    Uses fast, memory-efficient operations for matrices of single
    and double-precision numbers, uses fast-ish numpy code as a
    fallback, but at the cost of creating a copy of of the matrix.
    '''
    if matrix.dtype == np.float64:
        if scale.dtype == np.int64:
            return compiled.selfSoftDot_f8_i8(matrix, scale)
        else:
            return compiled.selfSoftDot_f8_f8(matrix, scale.astype(np.float64))
    elif matrix.dtype == np.float32:
        if scale.dtype == np.int64:
            return compiled.selfSoftDot_f4_i8(matrix, scale)
        else:
            scale_f4 = scale if scale.dtype == np.float32 else scale.astype(np.float32)
            return compiled.selfSoftDot_f4_f4(matrix, scale_f4)
    
    if WarnIfSlow:
        sys.stderr.write("WARNING: Slow code path triggered (scaledSelfSoftDot)")
    return np.sum(matrix * rowwise_softmax(matrix))

