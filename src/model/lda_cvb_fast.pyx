'''
Contains a number of functions required to implement LDA/VB/CV0
in a fast manner. See the lda_cvb module for more information on
the algorithm.

This module contains the core inference loop (which is algorithmically
similar to a Gibbs sampler, and so impossible to vectorize) as well
as methods to convert a bag of word-counts to a jagged 2-dim array
of word-observations per document, a representation required by most
LDA implementations.

As is typically the case, there are multiple implementations for multiple
different datatypes. However in general the newModelState() and 
newQueryState() methods in lda_cvb will coerce the input matrix W into
a matrix of int32 observations, regardless of the specified dtype

Compilation Notes
=================
This code expects a compiler with OpenMP support, and will multi-thread
certain operations where it offers benefit. On GCC this means you must 
link to the "gomp" library. You will also need to link to the standard C
math library.

To Dos
=================
TODO for the fastButInaccurate options, consider multithreading across
the D documents, ignoring the serial dependence on z_dnk, and then
recalculate the counts from z_dnk afterwards.
'''

cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, srand, malloc, free
from libc.math cimport log, exp
from libc.float cimport DBL_MAX, DBL_MIN, FLT_MAX, FLT_MIN
import scipy.linalg as la
import scipy.special as fns
#from openmp cimport omp_set_num_threads

cdef extern from "fastexp.h" nogil:
    double exp_approx "EXP" (double)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def toWordList_i32 (int[:] w_ptr, int[:] w_indices, int[:] w_data, int[:] docLens):
    '''
    For D documents, containing a vocabulary of T words, represented in a DxT matrix
    of word counts, return a list of word-observations, in a jagged DxN_d matrix, where
    N_d is the number of word observations in the d-th document. So a document with
    "The cat is the problem" would be converted into "cat is problem the the", this in
    turn would be converted to numeric word identifiers, and the order of those identifiers
    then permuted.
    
    For the sake of simplicity, we don't actually return a jagged array, instead we return
    an array of shape DxMaxN where MaxN = max_d N_d, and alongside this we return a vector
    of document lengths.
    
    As the number of word observations in a document will always be greater than the number
    of distinct words employed, this will increase the amount of memory required. This is
    partially offset by employing a dense representation, so we're storing 4 bytes per
    observation instead of 8-bytes per word-count.
    '''
    cdef:
        # The resulting matrix and its size
        int[:,:] result
        int D = len(w_ptr) - 1
        int maxN = np.max(docLens)
        # Loop indices
        int d
        int col = 0
        int colIdx = 0
        int nonZeroColCount = 0
        int observationCount = 0
        # Global (flat) index across all elements in the input
        int i = 0
        # Index across all observations (reset on every row)
        int obsIdx = 0
        # For swap operation
        int lhs
        int rhs
        int tmp
        int nd
        int n
    
    result = np.zeros((D,maxN), dtype=np.int32)
    srand(0xBADB055)
    
    with nogil:
        i = 0
        for d in range(D):
            obsIdx = 0
            nonZeroColCount = w_ptr[d+1] - w_ptr[d]
            for colIdx in range(nonZeroColCount):
                col = w_indices[i]
                observationCount = w_data[i]
                for obs in range(observationCount):
                    result[d,obsIdx] = col # column is the word
                    obsIdx += 1
                i += 1
        
        for d in range(D):
            nd = docLens[d]
            for n in range(nd):
                lhs = rand() % nd
                rhs = rand() % nd
                tmp = result[d,lhs]
                result[d,lhs] = result[d,rhs]
                result[d,rhs] = tmp
    
    return np.array(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def toWordList_f32 (int[:] w_ptr, int[:] w_indices, float[:] w_data, int[:] docLens):
    '''
    For D documents, containing a vocabulary of T words, represented in a DxT matrix
    of word counts, return a list of word-observations, in a jagged DxN_d matrix, where
    N_d is the number of word observations in the d-th document. So a document with
    "The cat is the problem" would be converted into "cat is problem the the", this in
    turn would be converted to numeric word identifiers, and the order of those identifiers
    then permuted.
    
    For the sake of simplicity, we don't actually return a jagged array, instead we return
    an array of shape DxMaxN where MaxN = max_d N_d, and alongside this we return a vector
    of document lengths.
    
    As the number of word observations in a document will always be greater than the number
    of distinct words employed, this will increase the amount of memory required. This is
    partially offset by employing a dense representation, so we're storing 4 bytes per
    observation instead of 8-bytes per word-count.
    '''
    cdef:
        # The resulting matrix and its size
        int[:,:] result
        int D = len(w_ptr) - 1
        int maxN = np.max(docLens)
        # Loop indices
        int d
        int col = 0
        int colIdx = 0
        int nonZeroColCount = 0
        int observationCount = 0
        # Global (flat) index across all elements in the input
        int i = 0
        # Index across all observations (reset on every row)
        int obsIdx = 0
        # For swap operation
        int lhs
        int rhs
        int tmp
        int nd
        int n
    
    result = np.zeros((D,maxN), dtype=np.int32)
    srand(0xBADB055)
    
    with nogil:
        i = 0
        for d in range(D):
            obsIdx = 0
            nonZeroColCount = w_ptr[d+1] - w_ptr[d]
            for colIdx in range(nonZeroColCount):
                col = w_indices[i]
                observationCount = <int> w_data[i]
                for obs in range(observationCount):
                    result[d,obsIdx] = col # column is the word
                    obsIdx += 1
                i += 1
        
        for d in range(D):
            nd = docLens[d]
            for n in range(nd):
                lhs = rand() % nd
                rhs = rand() % nd
                tmp = result[d,lhs]
                result[d,lhs] = result[d,rhs]
                result[d,rhs] = tmp
    
    return np.array(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def toWordList_f64 (int[:] w_ptr, int[:] w_indices, double[:] w_data, int[:] docLens):
    '''
    For D documents, containing a vocabulary of T words, represented in a DxT matrix
    of word counts, return a list of word-observations, in a jagged DxN_d matrix, where
    N_d is the number of word observations in the d-th document. So a document with
    "The cat is the problem" would be converted into "cat is problem the the", this in
    turn would be converted to numeric word identifiers, and the order of those identifiers
    then permuted.
    
    For the sake of simplicity, we don't actually return a jagged array, instead we return
    an array of shape DxMaxN where MaxN = max_d N_d, and alongside this we return a vector
    of document lengths.
    
    As the number of word observations in a document will always be greater than the number
    of distinct words employed, this will increase the amount of memory required. This is
    partially offset by employing a dense representation, so we're storing 4 bytes per
    observation instead of 8-bytes per word-count.
    '''
    cdef:
        # The resulting matrix and its size
        int[:,:] result
        int D = len(w_ptr) - 1
        int maxN = np.max(docLens)
        # Loop indices
        int d
        int col = 0
        int colIdx = 0
        int nonZeroColCount = 0
        int observationCount = 0
        # Global (flat) index across all elements in the input
        int i = 0
        # Index across all observations (reset on every row)
        int obsIdx = 0
        # For swap operation
        int lhs
        int rhs
        int tmp
        int nd
        int n
    
    result = np.zeros((D,maxN), dtype=np.int32)
    srand(0xBADB055)
    
    with nogil:
        i = 0
        for d in range(D):
            obsIdx = 0
            nonZeroColCount = w_ptr[d+1] - w_ptr[d]
            for colIdx in range(nonZeroColCount):
                col = w_indices[i]
                observationCount = <int> w_data[i]
                for obs in range(observationCount):
                    result[d,obsIdx] = col # column is the word
                    obsIdx += 1
                i += 1
        
        for d in range(D):
            nd = docLens[d]
            for n in range(nd):
                lhs = rand() % nd
                rhs = rand() % nd
                tmp = result[d,lhs]
                result[d,lhs] = result[d,rhs]
                result[d,rhs] = tmp
    
    return np.array(result)


def calculateCounts (W_list, docLens, z_dnk, T):
    '''
    Given the topic assignments, creates the counts of topic assignments per document,
    word observations per topic, and total words per topic.
    
    This can't be trivially vectorized (even with einsum) due to the fact that W_list
    is a jagged array stored in a square array.
    '''
    if z_dnk.dtype == np.float32:
        return calculateCounts_f32 (W_list, docLens, z_dnk, T)
    elif z_dnk.dtype == np.float64:
        return calculateCounts_f64 (W_list, docLens, z_dnk, T)
    else:
        raise ValueError ("No implementation defined for data-type " + str(z_dnk.dtype))



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculateCounts_f32 (int[:,:] W_list, int[:] docLens, float[:,:,:] z_dnk, int T):
    '''
    Given the topic assignments, creates the counts of topic assignments per document,
    word observations per topic, and total words per topic.

    This can't be trivially vectorized (even with einsum) due to the fact that W_list
    is a jagged array stored in a square array.
    '''
    cdef:
        float[:,:] n_dk
        float[:,:] n_kt
        float[:]   n_k
        float[:,:] v_dk
        float[:,:] v_kt
        float[:]   v_k
        int D
        int K
        int maxN
        int d,n,k
        int t
        float val

    D, maxN = W_list.shape[0], W_list.shape[1]
    K = z_dnk.shape[2]

    n_dk = np.zeros((D,K), dtype=np.float32)
    n_kt = np.zeros((K,T), dtype=np.float32)
    n_k  = np.zeros((K,),  dtype=np.float32)

    v_dk = np.zeros((D,K), dtype=np.float32)
    v_kt = np.zeros((K,T), dtype=np.float32)
    v_k  = np.zeros((K,),  dtype=np.float32)

    # Completely zero out the bits of Z that don't refer to any actual term
    # Recall Z is meant to be a jagged 2-dim array, but we store it in a matrix
    # for ease of access.
    with nogil:
        for d in range(D):
            for n in range(docLens[d], maxN):
                z_dnk[d,n,:] = 0 # Would be interesting to look into memset


    # Lastly manually loop through to get the counts of individual word
    # occurrences per assigned topi
    with nogil:
        for k in range(K):
            for d in range(D):
                for n in range(docLens[d]):
                    t = W_list[d,n]
                    val        = z_dnk[d,n,k]
                    n_dk[d,k] += val
                    n_kt[k,t] += val
                    n_k[k]    += val

                    val       *= (1 - val)
                    v_dk[d,k] += val
                    v_kt[k,t] += val
                    v_k[k]    += val

    return n_dk, n_kt, n_k, v_dk, v_kt, v_k

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculateCounts_f64 (int[:,:] W_list, int[:] docLens, double[:,:,:] z_dnk, int T):
    '''
    Given the topic assignments, creates the counts of topic assignments per document,
    word observations per topic, and total words per topic.

    This can't be trivially vectorized (even with einsum) due to the fact that W_list
    is a jagged array stored in a square array.
    '''
    cdef:
        double[:,:] n_dk
        double[:,:] n_kt
        double[:]   n_k
        double[:,:] v_dk
        double[:,:] v_kt
        double[:]   v_k
        int D
        int K
        int maxN
        int d,n,k
        int t
        double val

    D, maxN = W_list.shape[0], W_list.shape[1]
    K = z_dnk.shape[2]

    n_dk = np.zeros((D,K), dtype=np.float64)
    n_kt = np.zeros((K,T), dtype=np.float64)
    n_k  = np.zeros((K,),  dtype=np.float64)

    v_dk = np.zeros((D,K), dtype=np.float64)
    v_kt = np.zeros((K,T), dtype=np.float64)
    v_k  = np.zeros((K,),  dtype=np.float64)

    # Completely zero out the bits of Z that don't refer to any actual term
    # Recall Z is meant to be a jagged 2-dim array, but we store it in a matrix
    # for ease of access.
    with nogil:
        for d in range(D):
            for n in range(docLens[d], maxN):
                z_dnk[d,n,:] = 0 # Would be interesting to look into memset


    # Lastly manually loop through to get the counts of individual word
    # occurrences per assigned topi
    with nogil:
        for k in range(K):
            for d in range(D):
                for n in range(docLens[d]):
                    t = W_list[d,n]
                    val        = z_dnk[d,n,k]
                    n_dk[d,k] += val
                    n_kt[k,t] += val
                    n_k[k]    += val

                    val *= (1 - val)
                    v_dk[d,k] += val
                    v_kt[k,t] += val
                    v_k[k]    += val

    return n_dk, n_kt, n_k, v_dk, v_kt, v_k

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_f32(int iterations, int D_query, int D_train, int K, int T, \
                 int[:,:] W_list, int[:] docLens, \
                 float[:,:] q_n_dk, float[:,:] q_n_kt, float[:] q_n_k, \
                 float[:,:] q_v_dk, float[:,:] q_v_kt, float[:] q_v_k, \
                 float[:,:,:] z_dnk,\
                 float topicPriorDbl, float vocabPriorDbl):
    '''
    Performs the given number of iterations as part of the training
    procedure. There are two corpora, the model corpus of files, whose
    topic assignments are fixed and who constitute the prior (in this case
    the prior represented by real instead of pseudo-counts) and the query
    corpus for which we estimate topic assignments. We have the query-specific
    matrix of tokens generated by topic k for corpus d, then we have corpus
    wide matrices (though still prefixed by "q_") of word-counts per topic
    (the vocabularly distributions) and topic counts over all (the topic
    prior)
    
    Params:
    iterations - the number of iterations to perform
    D_query    - the number of query documents
    D_train    - the number of training documents
    K          - the number of topics
    T          - the number of possible words
    W_list     - a jagged DxN_d array of word-observations for each document
    docLens    - the length of each document d
    q_n_dk     - DxK matrix with counts of tokens generated by topic k in 
                 document d. This is for query documents only (model document
                 topic assignments are considered fixed: they're the "prior")
    q_n_kt     - DxT matrix with the number of times a distinct word t was
                 generated by topic  across the entire corpus (including 
                 immutable model documents)
    q_n_k      - The K-dimensional vector containing for each topic k the 
                 total number of tokens (not distinct words) generate by k
                 across the entire corpus, query and model
    q_v_dk     - DxK matrix with variances of tokens generated by topic k
                 for document d = \sum_n z_dnk * (1 - z_dnk)
    q_v_kt     - DxT matrix with variances of tokens generated as word t
                 for topic k = \sum_n z_dnk * w_dnt * (1 - z_dnk)
    q_v_k      - The K-dimensional vector containing for each topic k the
                 total variance = \sum_d \sum_n z_dnk * (1 - z_dnk)
    topicPrior - a scalar providing the scale of the symmetric prior over 
                 topics in the model
    vocabPrior - a scalar providing the scale of the symmetric prior over
                 vocabularies in the model
    '''
    
    cdef:
        int itr, d, n, t, k
        float *mems = new_array_f32(K)
        float denom, diff
        float sot, esot
        float topicPrior = <float> topicPriorDbl
        float vocabPrior = <float> vocabPriorDbl
        float term1, term2, term3
        
    try:
        with nogil:
            for itr in range(iterations):
                for d in range(D_query):
                    for n in range(docLens[d]):
                        t = W_list[d,n]
                        denom = 0.0
                        for k in range(K):
                            term1 = topicPrior     + q_n_dk[d,k] - z_dnk[d,n,k]
                            term2 = vocabPrior     + q_n_kt[k,t] - z_dnk[d,n,k]
                            term3 = vocabPrior * T + q_n_k[k]    - z_dnk[d,n,k]

                            mems[k] = (term1 * term2)

                            sot  = 0
                            sot -= q_v_dk[d,k] / (2 * term1 * term1)
                            sot -= q_v_kt[k,t] / (2 * term2 * term2)
                            sot += q_v_k[k]    / (2 * term3 * term3)

                            esot     = exp_approx(sot) if sot > -75.5 else 1E-33
                            mems[k] *= esot # exp_approx(sot)
                            mems[k] /= term3

                            denom += mems[k]
                            
                        for k in range(K):
                            mems[k] /= denom
                            # if is_invalid_prob_f32(mems[k]):# \
                            #     #or (d == 8095 and n==10 and k == (K-1)):
                            #     with gil:
                            #         for j in range(K):
                            #             print ("DEBUG: mems[%d]@d=%d,n=%d = %f z[%d,%d,%d] = %f" % (k, d, n, mems[j], d, n, k, z_dnk[d,n,j]))
                            #         print ("Iteration %d: mems[%d] = %f" % (itr, k, mems[k]))
                            #         print ("topicPrior + q_n_dk[%d,%d] - z_dnk[%d,%d,%d] = %f + %f - %f = %f" % (d, k, d, n, k, topicPrior, q_n_dk[d,k], z_dnk[d,n,k], topicPrior + q_n_dk[d,k] - z_dnk[d,n,k]))
                            #         print ("vocabPrior + q_n_kt[%d,%d] - z_dnk[%d,%d,%d] = %f + %f - %f = %f" % (k, t, d, n, k, vocabPrior, q_n_kt[k,t], z_dnk[d,n,k], vocabPrior + q_n_kt[k,t] - z_dnk[d,n,k]))
                            #         print ("T * vocabPrior + q_n_k[%d] - z_dnk[%d,%d,%d] = %f * %f + %f - %f = %f" % (k, d, n, k, T, vocabPrior, q_n_k[k], z_dnk[d,n,k], T * vocabPrior + q_n_k[k] - z_dnk[d,n,k]))
                            #         print ("q_v_dk[%d,%d] = %f, q_v_kt[%d,%d]=  %f, q_v_k[%d] = %f" % (d,k, q_v_dk[d,k], k, t, q_v_kt[k,t], k, q_v_k[k]))
                            #         if is_invalid_prob_f32(mems[k]):
                            #             return
                            
                            diff         = mems[k] - z_dnk[d,n,k]
                            q_n_dk[d,k] += diff
                            q_n_kt[k,t] += diff
                            q_n_k[k]    += diff

                            diff         = mems[k] - (z_dnk[d,n,k] * (1 - z_dnk[d,n,k]))
                            q_v_dk[d,k] += diff
                            q_v_kt[k,t] += diff
                            q_v_k[k]    += diff

                            z_dnk[d,n,k] = mems[k]
    finally:
        free(mems)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_f64(int iterations, int D_query, int D_train, int K, int T, \
                 int[:,:] W_list, int[:] docLens, \
                 double[:,:] q_n_dk, double[:,:] q_n_kt, double[:] q_n_k, \
                 double[:,:] q_v_dk, double[:,:] q_v_kt, double[:] q_v_k, \
                 double[:,:,:] z_dnk,\
                 double topicPriorDbl, double vocabPriorDbl):
    '''
    Performs the given number of iterations as part of the training
    procedure. There are two corpora, the model corpus of files, whose
    topic assignments are fixed and who constitute the prior (in this case
    the prior represented by real instead of pseudo-counts) and the query
    corpus for which we estimate topic assignments. We have the query-specific
    matrix of tokens generated by topic k for corpus d, then we have corpus
    wide matrices (though still prefixed by "q_") of word-counts per topic
    (the vocabularly distributions) and topic counts over all (the topic
    prior)

    Params:
    iterations - the number of iterations to perform
    D_query    - the number of query documents
    D_train    - the number of training documents
    K          - the number of topics
    T          - the number of possible words
    W_list     - a jagged DxN_d array of word-observations for each document
    docLens    - the length of each document d
    q_n_dk     - DxK matrix with counts of tokens generated by topic k in
                 document d. This is for query documents only (model document
                 topic assignments are considered fixed: they're the "prior")
    q_n_kt     - DxT matrix with the number of times a distinct word t was
                 generated by topic  across the entire corpus (including
                 immutable model documents)
    q_n_k      - The K-dimensional vector containing for each topic k the
                 total number of tokens (not distinct words) generate by k
                 across the entire corpus, query and model
    q_v_dk     - DxK matrix with variances of tokens generated by topic k
                 for document d = \sum_n z_dnk * (1 - z_dnk)
    q_v_kt     - DxT matrix with variances of tokens generated as word t
                 for topic k = \sum_n z_dnk * w_dnt * (1 - z_dnk)
    q_v_k      - The K-dimensional vector containing for each topic k the
                 total variance = \sum_d \sum_n z_dnk * (1 - z_dnk)
    topicPrior - a scalar providing the scale of the symmetric prior over
                 topics in the model
    vocabPrior - a scalar providing the scale of the symmetric prior over
                 vocabularies in the model
    '''

    cdef:
        int itr, d, n, t, k
        double *mems = new_array_f64(K)
        double denom, diff
        double sot
        double topicPrior = <double> topicPriorDbl
        double vocabPrior = <double> vocabPriorDbl
        float term1, term2, term3

    try:
        with nogil:
            for itr in range(iterations):
                for d in range(D_query):
                    for n in range(docLens[d]):
                        t = W_list[d,n]
                        denom = 0.0
                        for k in range(K):
                            term1 = topicPrior     + q_n_dk[d,k] - z_dnk[d,n,k]
                            term2 = vocabPrior     + q_n_kt[k,t] - z_dnk[d,n,k]
                            term3 = vocabPrior * T + q_n_k[k]    - z_dnk[d,n,k]

                            mems[k] = (term1 * term2) / term3

                            sot  = 0
                            sot -= q_v_dk[d,k] / (2 * term1 * term1)
                            sot -= q_v_kt[k,t] / (2 * term2 * term2)
                            sot += q_v_k[k]    / (2 * term3 * term3)

                            mems[k] *= exp_approx(sot)

                            denom += mems[k]

                        for k in range(K):
                            mems[k] /= denom
                            if is_invalid_prob_f64(mems[k]):
                                with gil:
                                    print ("Iteration %d: mems[%d] = %f" % (itr, k, mems[k]))
                                    print ("topicPrior + q_n_dk[%d,%d] - z_dnk[%d,%d,%d] = %f + %f - %f = %f" % (d, k, d, n, k, topicPrior, q_n_dk[d,k], z_dnk[d,n,k], topicPrior + q_n_dk[d,k] - z_dnk[d,n,k]))
                                    print ("vocabPrior + q_n_kt[%d,%d] - z_dnk[%d,%d,%d] = %f + %f - %f = %f" % (k, t, d, n, k, vocabPrior, q_n_kt[k,t], z_dnk[d,n,k], vocabPrior + q_n_kt[k,t] - z_dnk[d,n,k]))
                                    print ("T * vocabPrior + q_n_k[%d] - z_dnk[%d,%d,%d] = %f * %f + %f - %f = %f" % (k, d, n, k, T, vocabPrior, q_n_k[k], z_dnk[d,n,k], T * vocabPrior + q_n_k[k] - z_dnk[d,n,k]))
                                    return

                            diff         = mems[k] - z_dnk[d,n,k]
                            q_n_dk[d,k] += diff
                            q_n_kt[k,t] += diff
                            q_n_k[k]    += diff

                            diff         = mems[k] - (z_dnk[d,n,k] * (1 - z_dnk[d,n,k]))
                            q_v_dk[d,k] += diff
                            q_v_kt[k,t] += diff
                            q_v_k[k]    += diff

                            z_dnk[d,n,k] = mems[k]
    finally:
        free(mems)


cdef double *new_array_f64(int size) nogil:
    '''
    Wraps malloc: returns a 1-dim array of doubles with the given number of
    entries. 
    '''
    return <double *> malloc(size * sizeof(double))

cdef float *new_array_f32(int size) nogil:
    '''
    Wraps malloc: returns a 1-dim array of floats with the given number of
    entries. 
    '''
    return <float *> malloc(size * sizeof(float))

cdef bint is_invalid_prob_f64 (double zdnk) nogil:
    '''
    Return true if the given probability is NaN, and INF, or not in the
    range -0.001..1.001 inclusive.
    '''
    return not \
        (is_real_number_f64(zdnk) \
        and zdnk >= -0.1 \
        and zdnk <= 1.1)
        
cdef bint is_invalid_prob_f32 (float zdnk) nogil:
    '''
    Return true if the given probability is NaN, and INF, or not in the
    range -0.001..1.001 inclusive.
    '''
    return not \
        (is_real_number_f32(zdnk) \
        and zdnk >= -0.1 \
        and zdnk <= 1.1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def jagged_entropy_f32 (float[:,:,:] z_dnk, int[:] docLens):
    '''
    Calculates the entropy of distribution over all topics for
    all words for all documents. This is a jagged 3-dim array
    stored as a matrix for convenience. Being jagged, we have to
    do a manual loop instead of vectorizing (for the benefit of
    counts the "non-existent" values are set to zero, but this
    clearly doesn't work when logging elements
    '''
    cdef:
        float entropy = 0.0
        int D = z_dnk.shape[0]
        int K = z_dnk.shape[2]
        int d,n,k
    
    with nogil:
        for d in range(D):
            for n in range(docLens[d]):
                for k in range(K):
                    entropy -= z_dnk[d,n,k] * log(z_dnk[d,n,k])

    return entropy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def jagged_entropy_f64 (double[:,:,:] z_dnk, int[:] docLens):
    '''
    Calculates the entropy of distribution over all topics for
    all words for all documents. This is a jagged 3-dim array
    stored as a matrix for convenience. Being jagged, we have to
    do a manual loop instead of vectorizing (for the benefit of
    counts the "non-existent" values are set to zero, but this
    clearly doesn't work when logging elements
    '''
    cdef:
        double entropy = 0.0
        int D = z_dnk.shape[0]
        int K = z_dnk.shape[2]
        int d,n,k
    
    with nogil:
        for d in range(D):
            for n in range(docLens[d]):
                for k in range(K):
                    entropy -= z_dnk[d,n,k] * log(z_dnk[d,n,k])

    return entropy

cdef bint is_real_number_f64 (double x) nogil:
    return  -DBL_MAX <= x and x <= +DBL_MAX

cdef bint is_real_number_f32 (float x) nogil:
    return  -FLT_MAX <= x and x <= +FLT_MAX

