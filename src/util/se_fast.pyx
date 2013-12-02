'''
Created on 1 Dec 2013

@author: bryanfeeney
'''

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport log 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def entropyOfDot_f8 (double[:,:] topics, double[:,:] vocab):
    '''
    Given a DxK matrix of topic assignments for each of the D document,
    and a KxT matrix of per-topic word distributions for each of the T
    words, let us define the DxTxK tensor of per word,topic,document 
    probabilities, such that Z_dtk = topics_dt * vocab_kt
    
    The entropy of this distribution is 
    
    H[Z] = -sum_d sum_k sum_t Z_dtk * log(Z_dtk)
    
    This calculates and returns that entropy.
    '''
    cdef:
        double result = 0.0
        
        int D = topics.shape[0]
        int K = topics.shape[1]
        int T = vocab.shape[1]
        
        int d = 0
        int k = 0
        int t = 0
        
        double z_dtk = 0.0
        double denom = 0.0
    
    with nogil:
        while d < D:
            # Calculate the probability normalisers
            denom = 0.0
            k = 0
            while k < K:
                denom += topics[d,k]
                k += 1
            
            # Use the normalised probabilites to estimate the entropy
            k = 0
            while k < K:
                t = 0
                while t < T:
                    z_dtk   = (topics[d,k] / denom) * vocab[k,t]
                    result -= z_dtk * log (z_dtk)
                    t += 1
                k += 1
            d += 1
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def entropyOfDot_f4 (float[:,:] topics, float[:,:] vocab):
    '''
    Given a DxK matrix of topic assignments for each of the D document,
    and a KxT matrix of per-topic word distributions for each of the T
    words, let us define the DxTxK tensor of per word,topic,document 
    probabilities, such that Z_dtk = topics_dt * vocab_kt
    
    The entropy of this distribution is 
    
    H[Z] = -sum_d sum_k sum_t Z_dtk * log(Z_dtk)
    
    This calculates and returns that entropy.
    '''
    cdef:
        float result = 0.0
        
        int D = topics.shape[0]
        int K = topics.shape[1]
        int T = vocab.shape[1]
        
        int d = 0
        int k = 0
        int t = 0
        
        float z_dtk = 0.0
        float denom = 0.0
    
    with nogil:
        while d < D:
            # Calculate the probability normalisers
            denom = 0.0
            k = 0
            while k < K:
                denom += topics[d,k]
                k += 1
            
            # Use the normalised probabilites to estimate the entropy
            k = 0
            while k < K:
                t = 0
                while t < T:
                    z_dtk   = (topics[d,k] / denom) * vocab[k,t]
                    result -= z_dtk * log (z_dtk)
                    t += 1
                k += 1
            d += 1
    return result

    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparseScalarQuotientOfDot_f8(double[:] A_data, int[:] A_indices, int[:] A_ptr, double[:,:] B, double[:,:] C, double[:] out_data):
    '''
    Returns A / np.dot(B, C), however it does so keeping in  mind 
    the sparsity of A, calculating values only where required.
     
    Params
    A_data    - the values buffer of the sparse CSR matrix A
    A_indices - the indices buffer of the sparse CSR matrix A
    A_ptr     - the index pointer buffer of the sparse CSR matrix A
    B         - a dense matrix
    C         - a dense matrix
    out_data  - the values buffer into which the result will be placed.
    
    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    cdef int rowCount = len(A_ptr) - 1 
    cdef int elemCount = 0, e = 0
    cdef int row = 0, col = 0, i = 0
    with nogil:
        while row < rowCount:
            elemCount = A_ptr[row+1] - A_ptr[row]
            e = 0
            while e < elemCount:
                col = A_indices[i]
                out_data[i] = A_data[i] / dotProduct_f8(row,col,B,C)
                i += 1
                e += 1
            row += 1
    
    return out_data

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparseScalarQuotientOfDot_f4(float[:] A_data, int[:] A_indices, int[:] A_ptr, float[:,:] B, float[:,:] C, float[:] out_data):
    '''
    Returns A / np.dot(B, C), however it does so keeping in  mind 
    the sparsity of A, calculating values only where required.
     
    Params
    A_data    - the values buffer of the sparse CSR matrix A
    A_indices - the indices buffer of the sparse CSR matrix A
    A_ptr     - the index pointer buffer of the sparse CSR matrix A
    B         - a dense matrix
    C         - a dense matrix
    out_data  - the values buffer into which the result will be placed.
    
    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    cdef int rowCount = len(A_ptr) - 1 
    cdef int elemCount = 0, e = 0
    cdef int row = 0, col = 0, i = 0
    with nogil:
        while row < rowCount:
            elemCount = A_ptr[row+1] - A_ptr[row]
            e = 0
            while e < elemCount:
                col = A_indices[i]
                out_data[i] = A_data[i] / dotProduct_f4(row,col,B,C)
                i += 1
                e += 1
            row += 1
    
    return out_data


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparseScalarProductOfDot_f8(double[:] A_data, int[:] A_indices, int[:] A_ptr, double[:,:] B, double[:,:] C, double[:] out_data):
    '''
    Returns A * np.dot(B, C), however it does so keeping in  mind 
    the sparsity of A, calculating values only where required.
     
    Params
    A_data    - the values buffer of the sparse CSR matrix A
    A_indices - the indices buffer of the sparse CSR matrix A
    A_ptr     - the index pointer buffer of the sparse CSR matrix A
    B         - a dense matrix
    C         - a dense matrix
    out_data  - the values buffer into which the result will be placed.
    
    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    cdef int rowCount = len(A_ptr) - 1 
    cdef int elemCount = 0, e = 0
    cdef int row = 0, col = 0, i = 0
    with nogil:
        while row < rowCount:
            elemCount = A_ptr[row+1] - A_ptr[row]
            e = 0
            while e < elemCount:
                col = A_indices[i]
                out_data[i] = A_data[i] * dotProduct_f8(row,col,B,C)
                i += 1
                e += 1
            row += 1
    
    return out_data

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparseScalarProductOfDot_f4(float[:] A_data, int[:] A_indices, int[:] A_ptr, float[:,:] B, float[:,:] C, float[:] out_data):
    '''
    Returns A * np.dot(B, C), however it does so keeping in  mind 
    the sparsity of A, calculating values only where required.
     
    Params
    A_data    - the values buffer of the sparse CSR matrix A
    A_indices - the indices buffer of the sparse CSR matrix A
    A_ptr     - the index pointer buffer of the sparse CSR matrix A
    B         - a dense matrix
    C         - a dense matrix
    out_data  - the values buffer into which the result will be placed.
    
    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    cdef int rowCount = len(A_ptr) - 1 
    cdef int elemCount = 0, e = 0
    cdef int row = 0, col = 0, i = 0
    with nogil:
        while row < rowCount:
            elemCount = A_ptr[row+1] - A_ptr[row]
            e = 0
            while e < elemCount:
                col = A_indices[i]
                out_data[i] = A_data[i] * dotProduct_f4(row,col,B,C)
                i += 1
                e += 1
            row += 1
    
    return out_data

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparseScalarProductOfLnDot_f8(double[:] A_data, int[:] A_indices, int[:] A_ptr, double[:,:] B, double[:,:] C, double[:] out_data):
    '''
    Returns A * np.log(np.dot(B, C)), however it does so keeping in 
    mind the sparsity of A, calculate values only when required.
     
    Params
    A_data    - the values buffer of the sparse CSR matrix A
    A_indices - the indices buffer of the sparse CSR matrix A
    A_ptr     - the index pointer buffer of the sparse CSR matrix A
    B         - a dense matrix
    C         - a dense matrix
    out_data  - the values buffer into which the result will be placed.
    
    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    cdef int rowCount = len(A_ptr) - 1 
    cdef int elemCount = 0, e = 0
    cdef int row = 0, col = 0, i = 0
    with nogil:
        while row < rowCount:
            elemCount = A_ptr[row+1] - A_ptr[row]
            e = 0
            while e < elemCount:
                col = A_indices[i]
                out_data[i] = A_data[i] * log(dotProduct_f8(row,col,B,C))
                i += 1
                e += 1
            row += 1
    
    return out_data

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparseScalarProductOfLnDot_f4(float[:] A_data, int[:] A_indices, int[:] A_ptr, float[:,:] B, float[:,:] C, float[:] out_data):
    '''
    Returns A * np.log(np.dot(B, C)), however it does so keeping in 
    mind the sparsity of A, calculate values only when required.
     
    Params
    A_data    - the values buffer of the sparse CSR matrix A
    A_indices - the indices buffer of the sparse CSR matrix A
    A_ptr     - the index pointer buffer of the sparse CSR matrix A
    B         - a dense matrix
    C         - a dense matrix
    out_data  - the values buffer into which the result will be placed.
    
    Returns
    out_data, though note that this is the same parameter passed in and overwitten.
    '''
    cdef int rowCount = len(A_ptr) - 1 
    cdef int elemCount = 0, e = 0
    cdef int row = 0, col = 0, i = 0
    with nogil:
        while row < rowCount:
            elemCount = A_ptr[row+1] - A_ptr[row]
            e = 0
            while e < elemCount:
                col = A_indices[i]
                out_data[i] = A_data[i] * log(dotProduct_f4(row,col,B,C))
                i += 1
                e += 1
            row += 1
    
    return out_data

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dotProduct_f8 (int r, int c, double[:,:] B, double[:,:] C) nogil:
    '''
    The dot product of the r-th row of B and the c-th column of C.
    Done directly with a for-loop, no BLAS, SSE or anything. Still
    pretty fast though
    '''

    cdef double result = 0
    cdef int innerDim = B.shape[1]
    
    cdef int i = 0
    while i < innerDim:
        result += B[r,i] * C[i,c]
        i += 1

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dotProduct_f4 (int r, int c, float[:,:] B, float[:,:] C) nogil:
    '''
    The dot product of the r-th row of B and the c-th column of C.
    Done directly with a for-loop, no BLAS, SSE or anything. Still
    pretty fast though
    '''
    cdef float result = 0
    cdef int innerDim = B.shape[1]
    
    cdef int i = 0
    while i < innerDim:
        result += B[r,i] * C[i,c]
        i += 1

    return result


