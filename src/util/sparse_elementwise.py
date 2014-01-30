'''
Created on 1 Dec 2013

@author: bryanfeeney
'''
from distutils.core import setup

import numpy as np
import pyximport; 
pyximport.install(setup_args={"include_dirs": np.get_include(), "libraries":[('m', dict())]}, reload_support=True)
import util.se_fast as compiled
from numba import autojit
from math import log
import sys
from util.overflow_safe import safe_log

def entropyOfDot (topics, vocab):
    '''
    Given a DxK matrix of topic assignments for each of the D document,
    and a KxT matrix of per-topic word distributions for each of the T
    words, let us define the DxTxK tensor of per word,topic,document 
    probabilities, such that Z_dtk = topics_dt * vocab_kt
    
    The entropy of this distribution is 
    
    H[Z] = -sum_d sum_k sum_t Z_dtk * log(Z_dtk)
    
    This calculates and returns that entropy.
    '''
    if topics.dtype == np.float64:
        return compiled.entropyOfDot_f8(topics, vocab)
    elif topics.dtype == np.float32:
        return compiled.entropyOfDot_f4(topics, vocab)
    else:
        return entropyOfDot_py (topics, vocab)
    
@autojit
def entropyOfDot_py (topics, vocab):
    '''
    Given a DxK matrix of topic assignments for each of the D document,
    and a KxT matrix of per-topic word distributions for each of the T
    words, let us define the DxTxK tensor of per word,topic,document 
    probabilities, such that Z_dtk = topics_dt * vocab_kt
    
    The entropy of this distribution is 
    
    H[Z] = -sum_d sum_k sum_t Z_dtk * log(Z_dtk)
    
    This calculates and returns that entropy.
    '''
    (D,K) = topics.shape
    (_,T) = vocab.shape
    
    result = 0
    
    for d in range(D):
        denom = 0.0
        for k in range(K):
            denom += topics[d,k]
        
        for k in range(K):
            for t in range(T):
                result -= (topics[d,k] / denom) * vocab[k,t] * log ((topics[d,k] / denom) * vocab[d,k])
    
    return result
    


def sparseScalarQuotientOfDot (A, B, C, out=None):
    '''
    Returns A / np.dot(B, C), however it does so keeping in  mind 
    the sparsity of A, calculating values only where required.
     
    Params
    A         - a sparse CSR matrix
    B         - a dense matrix
    C         - a dense matrix
    out       - if specified, must be a sparse CSR matrix with identical
                non-zero pattern to A (i.e. same indices and indptr)
    
    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    if out is None:
        out = A.copy()
        
    if A.dtype == np.float64:
        compiled.sparseScalarQuotientOfDot_f8(A.data, A.indices, A.indptr, B, C, out.data)
    elif A.dtype == np.float32:
        compiled.sparseScalarQuotientOfDot_f4(A.data, A.indices, A.indptr, B, C, out.data)
    else:
        _sparseScalarQuotientOfDot_py(A,B,C, out)
    return out


def sparseScalarProductOfDot (A, B, C, out=None):
    '''
    Returns A * np.dot(B, C), however it does so keeping in  mind 
    the sparsity of A, calculating values only where required.
     
    Params
    A         - a sparse CSR matrix
    B         - a dense matrix
    C         - a dense matrix
    out       - if specified, must be a sparse CSR matrix with identical
                non-zero pattern to A (i.e. same indices and indptr)
    
    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    if out is None:
        out = A.copy()
    if A.dtype == np.float64:
        compiled.sparseScalarQuotientOfDot_f8(A.data, A.indices, A.indptr, B, C, out.data)
    elif A.dtype == np.float32:
        compiled.sparseScalarQuotientOfDot_f4(A.data, A.indices, A.indptr, B, C, out.data)
    else:
        _sparseScalarQuotientOfDot_py(A,B,C, out)
    return out

def _sparseScalarProductOfDot_py(A,B,C, out=None):
    '''
    Calculates A * B.dot(C) where A is a sparse matrix
    
    Retains sparsity in the result, unlike the built-in operator
    
    Note the type of the return-value is the same as the type of
    the sparse matrix A. If this has an integral type, this will
    only provide integer-based multiplication.
    '''
    if out is None:
        out = A.copy()
    if out is not A:
        out.data[:] = A.data
    
    out.data *= B.dot(C)[csr_indices(out.indptr, out.indices)]
    
    return out


def sparseScalarProductOfSafeLnDot (A, B, C, out=None):
    '''
    Returns A * np.log(np.dot(B, C)), however it does so keeping in
    mind the sparsity of A, calculating values only where required.
    Moreover if any product of the dot is zero, it's replaced with
    the minimum non-zero value allowed by the datatype, to avoid NaNs
     
    Params
    A         - a sparse CSR matrix
    B         - a dense matrix
    C         - a dense matrix
    out       - if specified, must be a sparse CSR matrix with identical
                non-zero pattern to A (i.e. same indices and indptr)
    
    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    if out is None:
        out = A.copy()
    if A.dtype == np.float64:
        compiled.sparseScalarProductOfSafeLnDot_f8(A.data, A.indices, A.indptr, B, C, out.data)
    elif A.dtype == np.float32:
        compiled.sparseScalarProductOfSafeLnDot_f4(A.data, A.indices, A.indptr, B, C, out.data)
    else:
        _sparseScalarProductOfSafeLnDot_py(A,B,C, out)
    return out


def _sparseScalarProductOfSafeLnDot_py(A,B,C, out=None):
    '''
    Calculates A * B.dot(C) where A is a sparse matrix
    
    Retains sparsity in the result, unlike the built-in operator
    
    Note the type of the return-value is the same as the type of
    the sparse matrix A. If this has an integral type, this will
    only provide integer-based multiplication.
    '''
    if out is None:
        out = A.copy()
    out.data[:] = A.data
    
    rhs = B.dot(C)
    rhs[rhs < sys.float_info.min] = sys.float_info.min
    out.data *= safe_log(rhs)[csr_indices(out.indptr, out.indices)]
    
    return out


def sparseScalarProductOf(A,B, out=None):
    '''
    Calculates A * B where A is a sparse matrix
    
    Retains sparsity in the result, unlike the built-in operator
    
    Note the type of the return-value is the same as the type of
    the sparse matrix A. If this has an integral type, this will
    only provide integer-based multiplication.
    '''
    if out is None:
        out = A.copy()
    if not out is A:
        out.data[:] = A.data
    out.data *= B[csr_indices(out.indptr, out.indices)]
    
    return out


def _sparseScalarQuotientOfDot_py(A,B,C, out=None):
    '''
    Calculates A / B.dot(C) where A is a sparse matrix
    
    Retains sparsity in the result, unlike the built-in operator
    
    Note the type of the return-value is the same as the type of
    the sparse matrix A. If this has an integral type, this will
    only provide integer-based division.
    '''
    if out is None:
        out = A.copy()
    if not out is A:
        out.data[:] = A.data
    
    out.data /= B.dot(C)[csr_indices(out.indptr, out.indices)]
    
    return out


@autojit
def csr_indices(ptr, ind):
    '''
    Returns the indices of a CSR matrix, given its indptr and indices arrays.
    '''
    rowCount = len(ptr) - 1 
    
    rows = [0] * len(ind)
    totalElemCount = 0

    for r in range(rowCount):
        elemCount = ptr[r+1] - ptr[r]
        if elemCount > 0:
            rows[totalElemCount : totalElemCount + elemCount] = [r] * elemCount
        totalElemCount += elemCount

    return [rows, ind.tolist()]
