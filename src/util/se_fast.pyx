'''
Created on 1 Dec 2013

@author: bryanfeeney
'''

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport log, exp
from libc.float cimport FLT_MIN, DBL_MIN


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def scaledSumOfLnOnePlusExp_f8(double[:] weights, double[:,:] matrix):
    '''
    Calculates sum(weights[row] * log(1 + exp(matrix[row,col]))
    for all rows and columns.
    
    Avoids under and overflow via approx
    
    Temporarily placed here for convenience
    '''
    cdef:
        double sum = 0.0
        int row = 0
        int rowCount = matrix.shape[0]
        int col = 0
        int colCount = matrix.shape[1]
        double value = 0.0
        
    with nogil:
        while row < rowCount:
            col = 0
            while col < colCount:
                value = matrix[row,col]
                sum += weights[row] * _safe_log_one_plus_exp_f8(value)
                col += 1
            row += 1
    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def scaledSumOfLnOnePlusExp_f4(float[:] weights, float[:,:] matrix):
    '''
    Calculates sum(weights[row] * log(1 + exp(matrix[row,col]))
    for all rows and columns.
    
    Avoids under and overflow via approx
    
    Temporarily placed here for convenience
    '''
    cdef:
        float sum = 0.0
        float value = 0.0
        int row = 0
        int rowCount = matrix.shape[0]
        int col = 0
        int colCount = matrix.shape[1]
        
    with nogil:
        while row < rowCount:
            col = 0
            while col < colCount:
                value = matrix[row,col]
                sum += weights[row] * _safe_log_one_plus_exp_f4(value)
                col += 1
            row += 1
    return sum

def safe_log_one_plus_exp_f4(float x):
    cdef:
        float result = 0.0
    with nogil:
        result = _safe_log_one_plus_exp_f4(x)
    
    return result

cdef float _safe_log_one_plus_exp_f4(float x) nogil:
    '''
    Returns log(1+exp(x))
    
    Works around three classes of precision issue:
     * Overflow
     * Underflow
     * Catastrophic cancellation (1+e^x == 1 for all x less some value)
    
    Note: There may be an advantage (thanks to branch prediction) in
    presorting the inputs when presenting a sequence of values
    '''
    cdef:
        float LOWER = -11
        float UPPER =  13.8
    
    if x <= LOWER:
        return exp(x) # 1 dominates in 1+exp(x), exceeding precision, so a dumb approach would give the same answer for all x. This gives an answer that's "close enough" to the answer we'd get with infinite precision
    elif x <= -LOWER:
        return log(1 + exp(x)) # within machine precision, so just evaluate.
    elif x <= UPPER:
        return x + exp(-x) # avoid overflow by evaluating log e^x + log (1 + e^-x) == log (e^x + 1). Use LOWER trick above to avoid domination by the "1" term
    else:
        return x # difference between 1+e^UPPER and e^UPPER is neglible

def safe_log_one_plus_exp_f8(double x):
    cdef:
        double result = 0.0
    with nogil:
        result = _safe_log_one_plus_exp_f8(x)
    
    return result
    
cdef double _safe_log_one_plus_exp_f8(double x) nogil:
    '''
    Returns log(1+exp(x))
    
    Works around three classes of precision issue:
     * Overflow
     * Underflow
     * Catastrophic cancellation (1+e^x == 1 for all x less some value)
    
    Note: There may be an advantage (thanks to branch prediction) in
    presorting the inputs when presenting a sequence of values
    '''
    cdef:
        double LOWER = -17
        double UPPER =  33
    
    if x <= LOWER:
        return exp(x) # 1 dominates in 1+exp(x), exceeding precision, so a dumb approach would give the same answer for all x. This gives an answer that's "close enough" to the answer we'd get with infinite precision
    elif x <= -LOWER:
        return log(1 + exp(x)) # within machine precision, so just evaluate.
    elif x <= UPPER:
        return x + exp(-x) # avoid overflow by evaluating log e^x + log (1 + e^-x) == log (e^x + 1). Use LOWER trick above to avoid domination by the "1" term
    else:
        return x # difference between 1+e^UPPER and e^UPPER is neglible

    

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
def sparseScalarQuotientOfDot_f8(double[:] A_data, int[:] A_indices, int[:] A_ptr, double[:,:] B, double[:,:] C, double[:] out_data, int start, int end):
    '''
    Returns A / np.dot(B, C), however it does so keeping in  mind 
    the sparsity of A, calculating values only where required.

    The start and end index into A only. It is assumed that B has (end-start)
    rows.
     
    Params
    A_data    - the values buffer of the sparse CSR matrix A
    A_indices - the indices buffer of the sparse CSR matrix A
    A_ptr     - the index pointer buffer of the sparse CSR matrix A
    B         - a dense matrix with (end-start) rows
    C         - a dense matrix
    start     - where to start processing in A
    end       - where to stop processing in A
    out_data  - the values buffer into which the result will be placed.
    
    Returns
    out_data, though note that this is the same parameter passed in and overwritten.
    '''
    cdef:
        int A_row = 0
        int B_row = 0
        int col = 0
        int elemCount =0
        int e = 0
        int i = 0

    with nogil:
        while A_row < end:
            elemCount = A_ptr[A_row+1] - A_ptr[A_row]
            e = 0
            if A_row >= start:
                while e < elemCount:
                    col = A_indices[i]
                    out_data[i] = A_data[i] / dotProduct_f8(B_row,col,B,C)
                    i += 1
                    e += 1
                B_row += 1
            A_row += 1
    
    return out_data

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparseScalarQuotientOfDot_f4(float[:] A_data, int[:] A_indices, int[:] A_ptr, float[:,:] B, float[:,:] C, float[:] out_data, int start, int end):
    '''
    Returns A / np.dot(B, C), however it does so keeping in  mind
    the sparsity of A, calculating values only where required.

    The start and end index into A only. It is assumed that B has (end-start)
    rows.

    Params
    A_data    - the values buffer of the sparse CSR matrix A
    A_indices - the indices buffer of the sparse CSR matrix A
    A_ptr     - the index pointer buffer of the sparse CSR matrix A
    B         - a dense matrix with (end-start) rows
    C         - a dense matrix
    start     - where to start processing in A
    end       - where to stop processing in A
    out_data  - the values buffer into which the result will be placed.

    Returns
    out_data, though note that this is the same parameter passed in and overwritten.
    '''
    cdef:
        int A_row = 0
        int B_row = 0
        int col = 0
        int elemCount =0
        int e = 0
        int i = 0

    with nogil:
        while A_row < end:
            elemCount = A_ptr[A_row+1] - A_ptr[A_row]
            e = 0
            if A_row >= start:
                while e < elemCount:
                    col = A_indices[i]
                    out_data[i] = A_data[i] / dotProduct_f4(B_row,col,B,C)
                    i += 1
                    e += 1
                B_row += 1
            A_row += 1
    
    return out_data


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparseScalarQuotientOfNormedDot_f8(double[:] A_data, int[:] A_indices, int[:] A_ptr, double[:,:] B, double[:,:] C, double[:] d, double[:] out_data):
    '''
    Returns A / np.dot(B, C/D), however it does so keeping in  mind
    the sparsity of A, calculating values only where required.

    Params
    A         - a sparse CSR matrix
    B         - a dense matrix
    C         - a dense matrix
    D         - a dense vector whose dimensionality matches the column-count of C
    out       - if specified, must be a sparse CSR matrix with identical
                non-zero pattern to A (i.e. same indices and indptr)

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
                out_data[i] = A_data[i] / dotProductNormed_f8(row,col,B,C,d)
                i += 1
                e += 1
            row += 1

    return out_data

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparseScalarQuotientOfNormedDot_f4(float[:] A_data, int[:] A_indices, int[:] A_ptr, float[:,:] B, float[:,:] C, float[:] d, float[:] out_data):
    '''
    Returns A / np.dot(B, C/D), however it does so keeping in  mind
    the sparsity of A, calculating values only where required.

    Params
    A         - a sparse CSR matrix
    B         - a dense matrix
    C         - a dense matrix
    D         - a dense vector whose dimensionality matches the column-count of C
    out       - if specified, must be a sparse CSR matrix with identical
                non-zero pattern to A (i.e. same indices and indptr)

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
                out_data[i] = A_data[i] / dotProductNormed_f4(row,col,B,C,d)
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
def sparseScalarProductOfSafeLnDot_f8(double[:] A_data, int[:] A_indices, int[:] A_ptr, double[:,:] B, double[:,:] C, double[:] out_data, int start, int end):
    '''
    Returns A * np.log(np.dot(B, C)), however it does so keeping in 
    mind the sparsity of A, calculating values only when required.
    Moreover if any product of the dot is zero, it's replaced with
    the minimum non-zero value allowed by the datatype, to avoid NaNs
     
    Params
    A_data    - the values buffer of the sparse CSR matrix A
    A_indices - the indices buffer of the sparse CSR matrix A
    A_ptr     - the index pointer buffer of the sparse CSR matrix A
    B         - a dense matrix
    C         - a dense matrix
    out_data  - the values buffer into which the result will be placed.
    start     - which row (inclusive) of A to start with
    end       - which row (exclusive) of A to end with
    
    Returns
    out_data, though note that this is the same parameter passed in and overwritten.
    '''
    cdef:
        int A_row = 0
        int B_row = 0
        int col = 0
        int elemCount =0
        int e = 0
        int i = 0
        double dotProd = 0.0
        double logOfMin = log (DBL_MIN)
    
    with nogil:
        while A_row < end:
            elemCount = A_ptr[A_row+1] - A_ptr[A_row]
            e = 0
            if A_row >= start:
                while e < elemCount:
                    col         = A_indices[i]
                    dotProd     = dotProduct_f8(B_row,col,B,C)
                    out_data[i] = A_data[i] * (log(dotProd) if dotProd > DBL_MIN else logOfMin)
                    i += 1
                    e += 1
                B_row += 1
            A_row += 1
    
    return out_data

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparseScalarProductOfSafeLnDot_f8_full(double[:] A_data, int[:] A_indices, int[:] A_ptr, double[:,:] B, double[:,:] C, double[:] out_data):
    cdef int rowCount = len(A_ptr) - 1
    cdef int elemCount = 0, e = 0
    cdef int row = 0, col = 0, i = 0
    cdef double dotProd = 0.0
    cdef double logOfMin = log (DBL_MIN)

    with nogil:
        while row < rowCount:
            elemCount = A_ptr[row+1] - A_ptr[row]
            e = 0
            while e < elemCount:
                col         = A_indices[i]
                dotProd     = dotProduct_f8(row,col,B,C)
                out_data[i] = A_data[i] * (log(dotProd) if dotProd > DBL_MIN else logOfMin)
                i += 1
                e += 1
            row += 1

    return out_data



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sparseScalarProductOfSafeLnDot_f4(float[:] A_data, int[:] A_indices, int[:] A_ptr, float[:,:] B, float[:,:] C, float[:] out_data, int start, int end):
    '''
    Returns A * np.log(np.dot(B, C)), however it does so keeping in 
    mind the sparsity of A, calculate values only when required.
    Moreover if any product of the dot is zero, it's replaced with
    the minimum non-zero value allowed by the datatype, to avoid NaNs
     
    Params
    A_data    - the values buffer of the sparse CSR matrix A
    A_indices - the indices buffer of the sparse CSR matrix A
    A_ptr     - the index pointer buffer of the sparse CSR matrix A
    B         - a dense matrix
    C         - a dense matrix
    out_data  - the values buffer into which the result will be placed.
    start     - which row (inclusive) of A to start with
    end       - which row (exclusive) of A to end with
    
    Returns
    out_data, though note that this is the same parameter passed in and overwritten.
    '''
    cdef:
        int A_row = 0
        int B_row = 0
        int col = 0
        int elemCount =0
        int e = 0
        int i = 0
        float dotProd = 0.0
        float logOfMin = log (FLT_MIN)

    with nogil:
        while A_row < end:
            elemCount = A_ptr[A_row+1] - A_ptr[A_row]
            e = 0
            if A_row >= start:
                while e < elemCount:
                    col         = A_indices[i]
                    dotProd     = dotProduct_f4(B_row,col,B,C)
                    out_data[i] = A_data[i] * (log(dotProd) if dotProd > FLT_MIN else logOfMin)
                    i += 1
                    e += 1
                B_row += 1
            A_row += 1
    
    return out_data

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dotProduct_f8 (int r, int c, double[:,:] B, double[:,:] C) nogil:
    '''
    The dot product of the r-th row of B and the c-th column of C.
    Done directly with a for-loop, no BLAS, SSE or anything. Still
    pretty fast though - just as quick as a numpy dot
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
cdef float dotProduct_f4 (int r, int c, float[:,:] B, float[:,:] C) nogil:
    '''
    The dot product of the r-th row of B and the c-th column of C.
    Done directly with a for-loop, no BLAS, SSE or anything. Still
    pretty fast though - just as quick as a numpy dot
    '''
    cdef float result = 0
    cdef int innerDim = B.shape[1]
    
    cdef int i = 0
    while i < innerDim:
        result += B[r,i] * C[i,c]
        i += 1

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dotProductNormed_f8 (int r, int c, double[:,:] B, double[:,:] C, double[:] d) nogil:
    '''
    Returns the dot product of B[r,:] and C[:,c] / d[:]
    Done directly with a for-loop, no BLAS, SSE or anything. Still
    pretty fast though - just as quick as a numpy dot
    '''
    cdef double result = 0
    cdef int innerDim = B.shape[1]

    cdef int i = 0
    while i < innerDim:
        result += B[r,i] * C[i,c] / d[i]
        i += 1

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float dotProductNormed_f4 (int r, int c, float[:,:] B, float[:,:] C, float[:] d) nogil:
    '''
    Returns the dot product of B[r,:] and C[:,c] / d[:]
    Done directly with a for-loop, no BLAS, SSE or anything. Still
    pretty fast though - just as quick as a numpy dot
    '''
    cdef float result = 0
    cdef int innerDim = B.shape[1]

    cdef int i = 0
    while i < innerDim:
        result += B[r,i] * C[i,c] / d[i]
        i += 1

    return result