'''
Created on 1 Dec 2013

@author: bryanfeeney
'''
import os

import numpy as np
import scipy.sparse as ssp
import util.se_fast as compiled
from math import log
import sys
from util.overflow_safe import safe_log
from util.sigmoid_utils import rowwise_softmax

WarnIfSlow=True


def scaledSumOfLnOnePlusExp(weights, matrix):
    '''
    Calculates sum(weights[row] * log(1 + exp(matrix[row,col]))
    for all rows and columns.
    
    Avoids under and overflow via approx
    
    Temporarily placed here for convenience
    '''
    assert not np.isfortran(matrix), "Matrix is not stored in row-major form"

    if matrix.dtype == np.float64:
        return compiled.scaledSumOfLnOnePlusExp_f8(weights, matrix)
    elif matrix.dtype == np.float32:
        return compiled.scaledSumOfLnOnePlusExp_f4(weights, matrix)
    else:
        raise ValueError ("No implementation for dtype=" + (str(matrix.dtype)))

def lse(matrix):
    '''
    The log-sum-exp function. For each _row_ in the matrix, calculated the
    log of the sum of the exponent of the values, i.e.
    
    log(sum exp(X[d,:]) for d in range(X.shape[0]))
    
    albeit a bit more efficiently than that code would suggest.
    '''
    assert not np.isfortran(matrix), "Matrix is not stored in row-major form"
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
    assert not np.isfortran(matrix), "Matrix is not stored in row-major form"
    if matrix.dtype == np.float64:
        return compiled.selfSoftDot_f8(matrix)
    elif matrix.dtype == np.float32:
        return compiled.selfSoftDot_f4(matrix)
    
    if WarnIfSlow:
        sys.stderr.write("WARNING: Slow code path triggered (selfSoftDot)")
    return np.sum(matrix * rowwise_softmax(matrix))

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
    assert not np.isfortran(topics), "Topics matrix is not stored in row-major form"
    assert not np.isfortran(vocab),  "Vocab matrix is not stored in row-major form"

    if topics.dtype == np.float64:
        return compiled.entropyOfDot_f8(topics, vocab)
    elif topics.dtype == np.float32:
        return compiled.entropyOfDot_f4(topics, vocab)
    else:
        return entropyOfDot_py (topics, vocab)
    

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
    if WarnIfSlow:
        sys.stderr.write("WARNING: Slow code path triggered (entropyOfDot_py)")
    
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



def sparseScalarQuotientOfDot (A, B, C, out=None, start=None, end=None):
    '''
    Returns A / np.dot(B, C), however it does so keeping in  mind 
    the sparsity of A, calculating values only where required.
     
    Params
    A         - a sparse CSR matrix
    B         - a dense matrix
    C         - a dense matrix
    out       - if specified, must be a sparse CSR matrix with identical
                non-zero pattern to A (i.e. same indices and indptr)
    start     - where to start indexing A
    end       - where to stop indexing A
    
    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    assert ssp.isspmatrix_csr(A), "A matrix is not a CSR matrix"
    assert not np.isfortran(B), "B matrix is not stored in row-major form"
    assert not np.isfortran(C), "C matrix is not stored in row-major form"

    if start is None:
        start = 0
    if end is None:
        end = A.shape[0]
    if out is None:
        out = A[start:end,:].copy()
        
    if A.dtype == np.float64:
        compiled.sparseScalarQuotientOfDot_f8(A.data, A.indices, A.indptr, B, C, out.data, start, end)
    elif A.dtype == np.float32:
        compiled.sparseScalarQuotientOfDot_f4(A.data, A.indices, A.indptr, B, C, out.data, start, end)
    else:
        raise ValueError ("No native Cython implementation for dtype " + str(A.dtype))
        _sparseScalarQuotientOfDot_py(A,B,C, out)
    return out

def sparseScalarQuotientOfNormedDot (A, B, C, d, out=None):
    '''
    Returns A / np.dot(B, C/D), however it does so keeping in  mind
    the sparsity of A, calculating values only where required.

    Params
    A         - a sparse CSR matrix
    B         - a dense matrix
    C         - a dense matrix
    d         - a dense vector whose dimensionality matches the column-count of C
    out       - if specified, must be a sparse CSR matrix with identical
                non-zero pattern to A (i.e. same indices and indptr)

    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    assert ssp.isspmatrix_csr(A), "A matrix is not a CSR matrix"
    assert not np.isfortran(B), "B matrix is not stored in row-major form"
    assert not np.isfortran(C), "C matrix is not stored in row-major form"

    if out is None:
        out = A.copy()

    if A.dtype == np.float64:
        compiled.sparseScalarQuotientOfNormedDot_f8(A.data, A.indices, A.indptr, B, C, d, out.data)
    elif A.dtype == np.float32:
        compiled.sparseScalarQuotientOfNormedDot_f4(A.data, A.indices, A.indptr, B, C, d, out.data)
    else:
        raise ValueError ("No implementation for the datatype " + str(A.dtype))
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
    assert ssp.isspmatrix_csr(A), "A matrix is not a CSR matrix"
    assert not np.isfortran(B), "B matrix is not stored in row-major form"
    assert not np.isfortran(C), "C matrix is not stored in row-major form"

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
    if WarnIfSlow:
        sys.stderr.write("WARNING: Slow code path triggered (_sparseScalarProductOfDot_py)")
        
    if out is None:
        out = A.copy()
    if out is not A:
        out.data[:] = A.data
    
    out.data *= B.dot(C)[csr_indices(out.indptr, out.indices)]
    
    return out


def sparseScalarProductOfSafeLnDot (A, B, C, out=None, start=None, end=None):
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
    start     - where to start indexing A
    end       - where to stop indexing A
    
    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    assert ssp.isspmatrix_csr(A), "A matrix is not a CSR matrix"
    assert not np.isfortran(B), "B matrix is not stored in row-major form"
    assert not np.isfortran(C), "C matrix is not stored in row-major form"

    if start is None:
        start = 0
    if end is None:
        end = A.shape[0]
    if out is None:
        out = A[start:end,:].copy()

    if (start, end) == (0, A.shape[0]) and A.dtype == np.float64:
        out = A.copy()
        oldWay = np.sum(compiled.sparseScalarProductOfSafeLnDot_f8_full(
             A.data, A.indices, A.indptr, B, C, out.data))
        out = A.copy()
        newWay = np.sum(compiled.sparseScalarProductOfSafeLnDot_f8(
             A.data, A.indices, A.indptr, B, C, out.data, 0, A.shape[0]))

        print ("The old way was " + str(oldWay) + " and the new way was " + str(newWay))
    elif A.dtype == np.float64:
        compiled.sparseScalarProductOfSafeLnDot_f8(A.data, A.indices, A.indptr, B, C, out.data, start, end)
    elif A.dtype == np.float32:
        compiled.sparseScalarProductOfSafeLnDot_f4(A.data, A.indices, A.indptr, B, C, out.data, start, end)
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
    if WarnIfSlow:
        sys.stderr.write("WARNING: Slow code path triggered (_sparseScalarProductOfSafeLnDot_py)")
        
    if not (A.dtype == B.dtype and B.dtype == C.dtype and (out is None or C.dtype == out.dtype)):
        raise ValueError ("Inconsistent dtypes in the three matrices and possibly the out-param")
        
    if out is None:
        out = A.copy()
    else:
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
    assert ssp.isspmatrix_csr(A), "A matrix is not a CSR matrix"
    assert not np.isfortran(B), "B matrix is not stored in row-major form"

    if WarnIfSlow:
        sys.stderr.write("WARNING: Slow code path triggered (sparseScalarProductOf)")
        
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
    if WarnIfSlow:
        sys.stderr.write("WARNING: Slow code path triggered (_sparseScalarQuotientOfDot_py)")
    
    if not (A.dtype == B.dtype and B.dtype == C.dtype and (out is None or C.dtype == out.dtype)):
        raise ValueError ("Inconsistent dtypes in the three matrices and possibly the out-param")
    
    if out is None:
        out = A.copy()
    if not out is A:
        out.data[:] = A.data
    
    out.data /= B.dot(C)[csr_indices(out.indptr, out.indices)]
    
    return out


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
