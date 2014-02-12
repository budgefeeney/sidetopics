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