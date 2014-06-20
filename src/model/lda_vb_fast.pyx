'''
Contains a number of functions required to implement LDA/VB
in a fast manner. The implementation also relies on the word list 
functions in lda_cvb_fast.pyx.

As is typically the case, there are multiple implementations for multiple
different datatypes. 

Compilation Notes
=================
This code expects a compiler with OpenMP support, and will multi-thread
certain operations where it offers benefit. On GCC this means you must 
link to the "gomp" library. You will also need to link to the standard C
math library and the Gnu scientific library (libgsl)

To Dos
=================
TODO for the fastButInaccurate options, consider multithreading across
the D documents, ignoring the serial dependence on z_dnk, and then
recalculate the counts from z_dnk afterwards.
'''

cimport cython
import numpy as np
cimport numpy as np

from cython.parallel cimport parallel, prange
from libc.stdlib cimport rand, srand, malloc, free
from libc.math cimport log, exp, sqrt, fabs, isnan, isinf
from libc.float cimport DBL_MAX, DBL_MIN, FLT_MAX, FLT_MIN
#from openmp cimport omp_set_num_threads

cdef int MaxInnerItrs = 40
cdef int MinInnerIters = 3

cdef extern from "gsl/gsl_sf_result.h":
    ctypedef struct gsl_sf_result:
        double val
        double err
cdef extern from "gsl/gsl_sf_psi.h":
    double gsl_sf_psi(double x) nogil
    int    gsl_sf_psi_e(double x, gsl_sf_result *result) nogil
    
cdef double digamma (double value) nogil:
    if value < 1E-300:
        value = 1E-300
    return gsl_sf_psi (value)

#    cdef:
#        gsl_sf_result result
#    
#    result.val = 0.0
#    result.err = 0.0
#    
#    with gil:
#        print ("Taking digamma of %g" % (value))
#    if gsl_sf_psi_e (value, &result) != 0:
#        with gil:
#            print ("Invalid input value for digamma %g" % (value,))
#        return 1E-100
#    return result.val
  

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_f32(int iterations, int D, int K, int T, \
                 int[:,:] W_list, int[:] docLens, \
                 float topicPrior, float vocabPrior, \
                 float[:,:] z_dnk, float[:,:] topicDists, float[:,:] vocabDists):
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
    topicPrior - the K dimensional vector with the topic prior
    vocabPrior - the T dimensional vector with the vocabulary prior
    z_dnk      - a max_d(N_d) x K dimensional matrix, containing all possible topic
                 assignments for a single document
    topicDists - the D x K matrix of per-document, topic probabilities. This _must_
                 be in C (i.e row-major) format.
    vocabDists - the K x T matrix of per-topic word probabilties
    '''
    
    cdef:
        int        itr, totalItrs, innerItrs, d, n, k, t
        float[:]   oldMems       = np.ndarray(shape=(K,),  dtype=np.float32)
        float[:,:] oldVocabDists = np.ndarray(shape=(K,T), dtype=np.float32)
        float[:,:] newVocabDists = vocabDists
        float      norm    = 0.0
        float      epsilon = 0.01
        
    with nogil:
        totalItrs = 0
        for itr in range(iterations):
            oldVocabDists, newVocabDists = newVocabDists, oldVocabDists
            newVocabDists[:,:] = 0.0
            
            for d in range(D):
                # For each document reset the topic probabilities and iterate to
                # convergence. This means we don't have to store the per-token
                # topic probabilties z_dnk for all documents, which is a huge structure
                oldMems[:]      = topicDists[d,:]
                topicDists[d,:] = 1./K

                innerItrs = 0
                while (l1_dist_f32 (oldMems, topicDists[d,:]) > epsilon) and (innerItrs < MaxInnerItrs):
                    totalItrs += 1
                    innerItrs += 1
                    
                    for n in range(docLens[d]):
                        norm = 0.0
                        for k in range(K):
                            z_dnk[n,k] = oldVocabDists[k,W_list[d,n]] * exp(digamma(topicDists[d,k]))
                            if is_invalid(z_dnk[n,k]):
                                with gil:
                                    print ("Invalid probability value: i=%d:%d z[%d,%d,%d] = %f. exp(Psi(topicDists[%d,%d])) = exp(Psi(%f)) = exp(%f) = %f, oldVocabDists[k,W_list[d,n]] = oldVocabDists[%d,%d] = %f" % (itr, totalItrs, d, n, k, z_dnk[n,k], d, k, topicDists[d,k], digamma(topicDists[d,k]), exp(digamma(topicDists[d,k])), k, W_list[d,n], oldVocabDists[k,W_list[d,n]]))
                                z_dnk[n,k] = 0
                                
                            norm += z_dnk[n,k]
                            
                        for k in range(K):
                            z_dnk[n,k] /= norm

                    for n in range(docLens[d]):
                        norm = 0.0
                        for k in range(K):
                            topicDists[d,k] = topicPrior + z_dnk[n,k]
                            norm += topicDists[d,k]
                            
                        for k in range(K):
                            topicDists[d,k] /= norm
                
                if innerItrs == MaxInnerItrs:
                    with gil:
                        print ("Iterated to max for document %d" % d)
                            
                
                # Once converged, update the vocabulary distribution
                for k in range(K):
                    norm = 0.0
                    for n in range(docLens[d]):
                        t = W_list[d,n]
                        newVocabDists[k,t] += z_dnk[n,k]
                        norm += newVocabDists[k,t]
                            
                    for t in range(T):
                        newVocabDists[k,t] /= norm
                
        # Just before we return, make sure the vocabDists memoryview that
        # was passed in has the latest vocabulary distributions
        if iterations % 2 == 0:
            vocabDists[:,:] = newVocabDists
            
    return totalItrs
                        

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float l1_dist_f32 (float[:] left, float[:] right) nogil:
    cdef:
        int i = 0
        float result = 0.0
        
    for i in range(left.shape[0]):
        result += fabs(left[i] - right[i])
    
    return result 


cdef bint is_invalid (double zdnk) nogil:
    return isnan(zdnk) \
        or isinf(zdnk) \
        or zdnk < -0.001
#        or zdnk > 1.001



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_f64(int iterations, int D, int K, int T, \
                 int[:,:] W_list, int[:] docLens, \
                 double topicPrior, double vocabPrior, \
                 double[:,:] z_dnk, double[:,:] topicDists, double[:,:] vocabDists):
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
    topicPrior - the K dimensional vector with the topic prior
    vocabPrior - the T dimensional vector with the vocabulary prior
    z_dnk      - a max_d(N_d) x K dimensional matrix, containing all possible topic
                 assignments for a single document
    topicDists - the D x K matrix of per-document, topic probabilities. This _must_
                 be in C (i.e row-major) format.
    vocabDists - the K x T matrix of per-topic word probabilties
    '''
    
    cdef:
        int         d, n, k, t
        int         itr, innerItrs, totalItrs
        double[:,:] oldVocabDists = np.ndarray(shape=(K,T), dtype=np.float64)
        double[:,:] newVocabDists = vocabDists
        double[:]   oldMems       = np.ndarray(shape=(K,), dtype=np.float64)
        double      max  = 1E-311
        double      norm = 0.0
        double[:]   vocabNorm = np.ndarray(shape=(K,), dtype=np.float64)
        double      epsilon = 0.01 / K
        
    totalItrs = 0
    for itr in range(iterations):
        oldVocabDists, newVocabDists = newVocabDists, oldVocabDists
        
        vocabNorm[:]       = vocabPrior * T
        newVocabDists[:,:] = vocabPrior
        
        with nogil:
        
            # NOTE THIS CODE COPY AND PASTED INTO lda_vb.var_bound() !
            for d in range(D):
                # For each document reset the topic probabilities and iterate to
                # convergence. This means we don't have to store the per-token
                # topic probabilties z_dnk for all documents, which is a huge saving
                oldMems[:]      = topicDists[d,:]
                topicDists[d,:] = 1./K
                innerItrs = 0
                
                while ((innerItrs < MinInnerIters) or (l1_dist_f64 (oldMems, topicDists[d,:]) > epsilon)) \
                and (innerItrs < MaxInnerItrs):
                    oldMems[:] = topicDists[d,:]
                    totalItrs += 1
                    innerItrs += 1
                    
                    # Determine the topic assignment for each individual token...
                    for n in range(docLens[d]):
                        norm = 0.0
                        max  = 1E-311
                        
                        # Work in log-space to avoid underflow
                        for k in range(K):
                            z_dnk[n,k] = log(oldVocabDists[k,W_list[d,n]]) + digamma(topicDists[d,k])
                            if z_dnk[n,k] > max:
                                max = z_dnk[n,k]

                        # Scale before converting to standard space so inference is feasible
                        for k in range(K):
                            z_dnk[n,k] = exp(z_dnk[n,k] - max)
                            norm += z_dnk[n,k]
                     
                        # Normalize the token probability, and check it's valid
                        for k in range(K):
                            z_dnk[n,k] /= norm
                            if is_invalid(z_dnk[n,k]):
                                with gil:
                                    print ("Iteration %d:%d z_dnk[%d,%d] = %f, norm = %g" \
                                           % (itr, totalItrs, n, k, z_dnk[n,k], norm))

                    # Use all the individual word topic assignments to determine
                    # the topic mixture exhibited by this document
                    topicDists[d,:] = topicPrior
                    norm = topicPrior * K
                    for n in range(docLens[d]):
                        for k in range(K):
                            topicDists[d,k] += z_dnk[n,k]
                            norm += z_dnk[n,k]
                            
                    for k in range(K):
                        topicDists[d,k] /= norm        
                
                # Once we've found document d's topic distribution, we
                # use that to build the new vocabulary distribution
                for k in range(K):
                    for n in range(docLens[d]):
                        t = W_list[d,n]
                        newVocabDists[k,t] += z_dnk[n,k]
                        vocabNorm[k] += z_dnk[n,k]
                        
                        if is_invalid(newVocabDists[k,t]):
                            with gil:
                                print ("newVocabDist[%d,%d] = %f, z_dnk[%d,%d] = %f" \
                                      % (k, t, newVocabDists[k,t], n, k, z_dnk[n,k]))
                            
            # With all document processed, normalize the vocabulary
            for k in prange(K):
                for t in range(T):
                    newVocabDists[k,t] /= vocabNorm[k]
                
        # Just before we return, make sure the vocabDists memoryview that
        # was passed in has the latest vocabulary distributions
        if iterations % 2 == 0:
            vocabDists[:,:] = newVocabDists
            
    print ("Average inner iterations %f" % (float(totalItrs) / (D*iterations)))
    return totalItrs                        



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double l1_dist_f64 (double[:] left, double[:] right) nogil:
    cdef:
        int i = 0
        double result = 0.0
        
    for i in range(left.shape[0]):
        result += fabs(left[i] - right[i])
    
    return result 



