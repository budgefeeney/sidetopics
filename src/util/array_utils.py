#!/usr/bin/python
# -*- coding: utf-8 -*- 
'''
A collection of functions that perform various scaling and normalization 
operations on numpy arrays

Created on 2 Sep 2013

@author: bryanfeeney
'''
import numpy as np

def normalizerows_ip (matrix):
    '''
    Normalizes a matrix IN-PLACE.
    '''
    row_sums = matrix.sum(axis=1)
    matrix   /= row_sums[:, np.newaxis]
    return matrix

def rowwise_softmax (matrix):
    '''
    Assumes each row of the given matrix is an unnormalized distribution and
    uses the softmax metric to normalize it. This additionally uses some
    scaling to ensure that we never overflow.
    '''
    # TODO Just how compute intense is this method call?
    
    row_maxes = matrix.max(axis=1) # Underflow makes sense i.e. Pr(K=k) = 0. Overflow doesn't, i.e Pr(K=k) = \infty
    result    = np.exp(matrix - row_maxes[:, np.newaxis])
    result   /= result.sum(axis=1)[:,np.newaxis]
    return result