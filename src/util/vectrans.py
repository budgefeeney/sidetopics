'''
Optimised functions to apply the vec-transpose operator, and create a
vec-transpose matrix. Also includes an implementation of the vec()
function.

The vec-transpose operator, given a matrix A and scale p, does a 
fortran-order reshape of each column such that the resulting matrix
contains p rows. The ultimate result is the matrix created by stacking
these matrices vertically. Vec-Transpose operators crop up with taking
the derivative of terms that appear as the left operand in a 
kronecker product.

The vec-tranpose matrix is the matrix T such that
  T vec(X) = vec(X.T)

Crucially we can construct a T such that a kronecker product A (x) B
can be rewritten as T (B (x) A) T' at which point we can use the 
vec-tranpose operator described above to take the derivative of B. 

So both of these crop up with dealing with the derivatives of terms
in Kronecker products.

Note that some of the implementations here are written in Cython for
speed: so Cython is a pre-requisite.

Created on 9 Nov 2013

@author: bryanfeeney
'''

import numpy as np
import scipy.sparse as ssp

from numba import autojit
from math import floor

import pyximport; 
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
import util.vt_fast as compiled

def vec(A):
    '''
    The vec operator for a matrix: returns a vector with all the columns of the
    matrix stacked on atop the other. For fortran/column ordered matrices this
    is essentially a no-op
    '''
    return np.reshape(A, (-1,1), order='F')

def vec_transpose_csr(A, p):
    '''
    Applies the vec-transpose operator to the given sparse matrix, which is stored in
    C / row-wise format
    
    Returns a new matrix after the vec-transpose operator has been applied.
    
    The vec-transpose operating on matrix A with scale p reshapes in fortran/column order each 
    column in that matrix into a matrix with p rows. It then returns a matrix of these sub-matrices
    stacked from top to bottom corresponding to columns read from left to right.
    '''
    (oldRows, oldCols) = A.shape
    if A.dtype == np.float64:
        return compiled.sparse_vec_transpose_f8r (A.data, A.indices, A.indptr, oldRows, oldCols, p)
    elif A.dtype == np.float32:
        return compiled.sparse_vec_transpose_f4r (A.data, A.indices, A.indptr, oldRows, oldCols, p)
    elif A.dtype == np.int32:
        return compiled.sparse_vec_transpose_i4r (A.data, A.indices, A.indptr, oldRows, oldCols, p)
    else: # Fall back on pure python (albeit with JIT compilation)
        return _vec_transpose_csr_jit(A.data, A.indices, A.indptr, A.shape, p)
    

@autojit
def _vec_transpose_csr_jit(data, indices, indptr, shape, p):
    '''
    Applies the vec-transpose operator to the given sparse matrix, which is stored in
    C / row-wise format. This is a pure Python implementation that uses Numba for some
    speed improvements. The public method will defer to compiled Cython code where the
    datatype is a 32-bit or 64-bit float, and so should be called accordingly.
    
    Returns a new matrix after the vec-transpose operator has been applied.
    
    The vec-transpose operating on matrix A with scale p reshapes in fortran/column order each 
    column in that matrix into a matrix with p rows. It then returns a matrix of these sub-matrices
    stacked from top to bottom corresponding to columns read from left to right.
    '''
    length = len(data)
    rows = np.ndarray((length,), dtype=np.int32)
    cols = np.ndarray((length,), dtype=np.int32)
    
    newRows, newCols = shape[1] * p, shape[0] / p
    oldRow, oldCol = 0, 0
    
    dataPtr = 0
    
    while dataPtr < length:
        while indptr[oldRow + 1] <= dataPtr:
            oldRow += 1
        oldCol = indices[dataPtr]
        
        rows[dataPtr] = oldCol * p + oldRow % p
        cols[dataPtr] = floor(oldRow / p)
        
        dataPtr += 1
    
    return ssp.coo_matrix ((data, (rows, cols)), shape = (newRows, newCols)).tocsr()

def vec_transpose(A, p):
    '''
    Applies the vec-transpose operator to the given dense matrix, which is stored in
    C / row-wise format
    
    Returns a new matrix after the vec-transpose operator has been applied.
    
    The vec-transpose operating on matrix A with scale p reshapes in fortran/column order each 
    column in that matrix into a matrix with p rows. It then returns a matrix of these sub-matrices
    stacked from top to bottom corresponding to columns read from left to right.
    '''
    if A.dtype == np.float64:
        return np.array (compiled.vec_transpose_f8r (A, p))
    elif A.dtype == np.float32:
        return np.array(compiled.vec_transpose_f4r (A, p))
    
    # Fall back to pure Python 
    (oldRows, oldCols) = A.shape
    newRows, newCols = oldCols * p, oldRows / p
    out = np.ndarray((newRows, newCols), dtype=A.dtype)
    
    for oldRow in range(oldRows):
        for oldCol in range(oldCols):
            newRow = oldCol * p + oldRow % p
            newCol = oldRow / p
            out[newRow, newCol] = A[oldRow, oldCol]
        
    return out


@autojit
def sp_vec_trans_matrix(shape):
    '''
    Generates a vec transpose matrix quickly, and returns it as a sparse
    matrix.
    
    The vec-transpose matrix T of a given matrix A is the matrix that 
    implements the equality
    
    T.dot(vec(A)) = vec(A.T)
    
    Params
        shape - the shape of the matrix A
    
    Returns
        A sparse transform matrix
    '''
    
    (rows, cols) = shape
    size         = rows * cols
    indptr  = np.arange(0, size + 1, dtype=np.int32)
    data    = np.ndarray((size,), dtype=np.float32)
    indices = np.ndarray((size,), dtype=np.int32)
    
    for r in range(size):
        indices[r] = floor(r / cols)  + (r % cols) * rows
    data.fill(1)
    
    return ssp.csr_matrix((data, indices, indptr), shape=(size, size))


    