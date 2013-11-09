'''
Contains Cython versions of the vec_transpose methods for matrices and operators
which have been shown to improve better than native Python and numba.

These are exposed via pure Python methods in the vectrans.py module, which is
what you should import and use to call these methods.

@author: bryanfeeney
'''


cimport cython
import numpy as np
cimport numpy as np
import scipy.sparse as ssp

@cython.cdivision(True)
@cython.boundscheck(False)
def sparse_vec_transpose_f8r(double[:] data, int[:] indices, int[:] indptr, int oldRows, int oldCols, int p):
    '''
    Applies the vec-transpose operator to the given sparse matrix, which is stored in
    C / row-wise format, with 64-bit doubles.
    
    Returns a new matrix after the vec-transpose operator has been applied.
    
    The vec-transpose operating on matrix A with scale p reshapes in fortran/column order each 
    column in that matrix into a matrix with p rows. It then returns a matrix of these sub-matrices
    stacked from top to bottom corresponding to columns read from left to right.
    '''
    cdef int length = len(data)
    cdef int[:] rows = np.ndarray((length,), dtype=np.int32)
    cdef int[:] cols = np.ndarray((length,), dtype=np.int32)
    
    cdef int newRows = oldCols * p
    cdef int newCols = oldRows / p
    
    cdef int oldRow = 0
    cdef int oldCol = 0
    
    cdef int dataPtr = 0
    while dataPtr < length:
        while indptr[oldRow + 1] <= dataPtr:
            oldRow += 1
        oldCol = indices[dataPtr]
        
        rows[dataPtr] = oldCol * p + oldRow % p
        cols[dataPtr] = oldRow / p
        
        dataPtr += 1
    
    cdef tuple shape = (newRows, newCols)
    return ssp.coo_matrix ((np.array(data), (np.array(rows), np.array(cols))), shape = shape).tocsr()


def sparse_vec_transpose_f4r(float[:] data, int[:] indices, int[:] indptr, int oldRows, int oldCols, int p):
    '''
    Applies the vec-transpose operator to the given sparse matrix, which is stored in
    C / row-wise format, with 32-bit doubles.
    
    Returns a new matrix after the vec-transpose operator has been applied.
    
    The vec-transpose operating on matrix A with scale p reshapes in fortran/column order each 
    column in that matrix into a matrix with p rows. It then returns a matrix of these sub-matrices
    stacked from top to bottom corresponding to columns read from left to right.
    '''
    cdef int length = len(data)
    cdef int[:] rows = np.ndarray((length,), dtype=np.int32)
    cdef int[:] cols = np.ndarray((length,), dtype=np.int32)
    
    cdef int newRows = oldCols * p
    cdef int newCols = oldRows / p
    
    cdef int oldRow = 0
    cdef int oldCol = 0
    
    cdef int dataPtr = 0
    while dataPtr < length:
        while indptr[oldRow + 1] <= dataPtr:
            oldRow += 1
        oldCol = indices[dataPtr]
        
        rows[dataPtr] = oldCol * p + oldRow % p
        cols[dataPtr] = oldRow / p
        
        dataPtr += 1
    
    cdef tuple shape = (newRows, newCols)
    return ssp.coo_matrix ((np.array(data), (np.array(rows), np.array(cols))), shape = shape).tocsr()


@cython.cdivision(True)
@cython.boundscheck(False)
def vec_transpose_f8r (double[:,:] inArray, int p):
    '''
    Applies the vec-transpose operator to the given dense matrix, which is stored in
    C / row-wise format, with 64-bit doubles.
    
    Returns a new matrix after the vec-transpose operator has been applied.
    
    The vec-transpose operating on matrix A with scale p reshapes in fortran/column order each 
    column in that matrix into a matrix with p rows. It then returns a matrix of these sub-matrices
    stacked from top to bottom corresponding to columns read from left to right.
    '''
    if inArray.strides[1] != 8: # Basically, if it's not in row-order
        return None
    
    cdef int oldRows = inArray.shape[0]
    cdef int oldCols = inArray.shape[1]
    cdef int newRows = oldCols * p
    cdef int newCols = oldRows / p
    
    cdef double[:,:] out = np.ndarray((newRows, newCols), dtype=np.float64)
    
    cdef int newRow
    cdef int newCol
    cdef int oldRow = 0
    cdef int oldCol = 0
    with nogil:
        while oldRow < oldRows:
            oldCol = 0
            while oldCol < oldCols:
                newRow = oldCol * p
                newCol = oldRow / p
                out[newRow, newCol] = inArray[oldRow, oldCol]
                
                oldCol += 1
            oldRow += 1
        
    return out

@cython.cdivision(True)
@cython.boundscheck(False)
def vec_transpose_f4r (float[:,:] inArray, int p):
    '''
    Applies the vec-transpose operator to the given dense matrix, which is stored in
    C / row-wise format, with 32-bit doubles.
    
    Returns a new matrix after the vec-transpose operator has been applied.
    
    The vec-transpose operating on matrix A with scale p reshapes in fortran/column order each 
    column in that matrix into a matrix with p rows. It then returns a matrix of these sub-matrices
    stacked from top to bottom corresponding to columns read from left to right.
    '''
    
    if inArray.strides[1] != 4: # Basically, if it's not in row-order
        return None
    
    cdef int oldRows = inArray.shape[0]
    cdef int oldCols = inArray.shape[1]
    cdef int newRows = oldCols * p
    cdef int newCols = oldRows / p
    
    cdef double[:,:] out = np.ndarray((newRows, newCols), dtype=np.float64)
    
    cdef int newRow
    cdef int newCol
    cdef int oldRow = 0
    cdef int oldCol = 0
    with nogil:
        while oldRow < oldRows:
            oldCol = 0
            while oldCol < oldCols:
                newRow = oldCol * p
                newCol = oldRow / p
                out[newRow, newCol] = inArray[oldRow, oldCol]
                
                oldCol += 1
            oldRow += 1
        
    return out

