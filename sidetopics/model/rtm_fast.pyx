'''
Routines for inference in the RTM model. Unlike the LDA model, we store the
mean of per-token topic-distributions (z_dn) in the topicDists structure.
To convert this to what it's meant to be - a Dirichlet hyperparameter,
multiply each row by the associated document's row-count and add the topic
prior. To get the expected value, it's simply a matter of adjusting the
average to downweight the existing value by one observation and add in
a scaled version of the topic prior.

@author: bryanfeeney
'''

cimport cython
import numpy as np
cimport numpy as np


import scipy.special as fns

from model.lda_vb_fast cimport initAtRandom_f64, l1_dist_f64
from libc.math cimport exp, erf

cdef int MaxInnerItrs = 100
cdef int MinInnerIters = 3

cdef extern from "gsl/gsl_sf_result.h":
    ctypedef struct gsl_sf_result:
        double val
        double err
cdef extern from "gsl/gsl_sf_psi.h":
    double gsl_sf_psi(double x) nogil
    double gsl_sf_psi_1(double x) nogil
    
cdef double digamma (double value) nogil:
    if value < 1E-300:
        value = 1E-300
    return gsl_sf_psi (value)

cdef double trigamma (double value) nogil:
    if value < 1E-300:
        value = 1E-300
    return gsl_sf_psi_1 (value)


cdef double SqrtTwo = 1.414213562373095048801688724209
cdef double OneOverSqrtTwo = 1. / SqrtTwo
cdef np.ndarray[np.float64_t, ndim=1] probit_f64(np.ndarray[np.float64_t, ndim=1] x):
    '''
    Returns probit of every element of the given numpy array
    '''
    return 0.5 * fns.erfc(-x * OneOverSqrtTwo)

def probit(X):
    '''
    Returns probit of every element of the given numpy array
    '''
    assert X.dtype is np.float64, "Only implemented for 64-bit double-precision floats"
    return probit_f64 (X)

cdef probit_sf64 (double x) nogil:
    return 0.5 * erfc (-x * OneOverSqrtTwo)

cdef double SqrtTwoPi = 2.5066282746310002
cdef double OneOverSqrtTwoPi = 1. / SqrtTwoPi
cdef np.ndarray[np.float64_t, ndim=1] normpdf_f64(np.ndarray[np.float64_t, ndim=1] x):
    '''
    Returns standard normal PDF of every element of the given numpy array
    
    np.exp(-x**2/2) / SqrtTwoPi
    
    While a bit complex, for large arrays the implementation below runs about
    1.75 times faster than the simple expression above
    '''
    result = x.copy()
    result *= x
    result *= -0.5
    np.exp(result, out=result)
    result *= OneOverSqrtTwoPi
    return result

cdef normpdf_sf64 (double x) nogil:
    return OneOverSqrtTwoPi * exp (-0.5 * x * x)

cdef bint is_invalid (double zdnk) nogil:
    return is_nan(zdnk) \
        or zdnk < -0.001 \
#        or zdnk > 1.1 \ # Called before normalization
#         or isinf(zdnk) \
#        or zdnk > 1.001

cdef bint is_nan(double num) nogil:
    '''
    Work around the fact that this isn't defined on the cluster
    '''
    return num != num


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterate_f64(int iterations, int D, int K, int T, \
                 int[:,:] W_list, int[:] docLens, \
                 double[:] topicPrior, double vocabPrior, \
                 double[:,:] z_dnk, double[:,:] topicMeans, double[:,:] vocabDists):
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
    D          - the number of documents
    K          - the number of topics
    T          - the number of possible words
    W_list     - a jagged DxN_d array of word-observations for each document
    docLens    - the length of each document d
    topicPrior - the K dimensional vector with the topic prior
    vocabPrior - the T dimensional vector with the vocabulary prior
    z_dnk      - a max_d(N_d) x K dimensional matrix, containing all possible topic
                 assignments for a single document
    topicMeans - the D x K matrix of averages of per-token probabilities.
    vocabDists - the K x T matrix of per-topic word probabilties
    '''
    
    cdef:
        int         d, n, k, t
        int         itr, totalItrs
        double[:,:] diVocabDists    = np.ndarray(shape=(K,T), dtype=np.float64)
        double[:]   diSumVocabDists = np.ndarray(shape=(K,), dtype=np.float64) 
        double[:]   oldMems         = np.ndarray(shape=(K,), dtype=np.float64)

        double      topicPriorSum
        double[:]   oldTopicPrior
        double[:]   tmp
        double[:,:] count
        double[:]   num
        double      dnm
        
    
    totalItrs = 0
    for itr in range(iterations):
        diVocabDists    = fns.digamma(vocabDists)
        diSumVocabDists = fns.digamma(np.sum (vocabDists, axis=1))
        
        vocabDists[:,:] = vocabPrior
        topicPriorSum = np.sum(topicPrior)
        
        for d in range(D):
            # Figure out document d's topic distribution, this is
            # written into topicDists and z_dnk
            totalItrs += infer_topics_f64(d, D, K, \
                                          W_list, docLens, \
                                          topicPrior, topicPriorSum, \
                                          z_dnk, oldMems, topicMeans, \
                                          diVocabDists, diSumVocabDists)

            # Then use those to gradually update our new vocabulary
            for k in range(K):
                for n in range(docLens[d]):
                    t = W_list[d,n]
                    vocabDists[k,t] += z_dnk[n,k]
        
                    
        # And update the prior on the topic distribution. We
        # do this with the GIL, as built-in numpy is likely faster
#         count = np.multiply(topicDists, docLens[:,None])
#         count /= (np.sum(topicDists, axis=1))[:,np.newaxis]
#         countSum = np.sum(count, axis=1)
#         for k in range(K):
#             topicPrior[k] = 1.0
#         for _ in range(1000):
#             oldTopicPrior = np.copy(topicPrior)
#               
#             num = np.sum(fns.psi(np.add (count, topicPrior[None, :])), axis=0) - D * fns.psi(topicPrior)
#             dnm = np.sum(fns.psi(countSum + np.sum(topicPrior)), axis=0) - D * fns.psi(np.sum(topicPrior))
#               
#             tmp = np.divide(num, dnm)
#             for k in range(K):
#                 topicPrior[k] *= tmp[k]
#               
#             if la.norm(np.subtract(oldTopicPrior, topicPrior), 1) < (0.001 * K):
#                 break
     
    # Just before we return, make sure the vocabDists memoryview that
    # was passed in has the latest vocabulary distributions
            
    print ("Average inner iterations %f" % (float(totalItrs) / (D*iterations)))
    
    return totalItrs                        



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def query_f64(int D, int T, int K, \
                 int[:,:] W_list, int[:] docLens, \
                 double[:] topicPrior, double[:,:] z_dnk, double[:,:] topicDists, 
                 double[:,:] vocabDists):
    cdef:
        int         d
        double[:]   oldMems = np.ndarray(shape=(K,),  dtype=np.float64)
        double[:,:] diVocabDists
        double[:]   diSumVocabDists
        double      topicPriorSum = np.sum(topicPrior)
    
    diVocabDists    = fns.digamma(vocabDists)
    diSumVocabDists = fns.digamma(np.sum(vocabDists, axis=1))
    
    with nogil:
        for d in range(D):
            infer_topics_f64(d, D, K, \
                 W_list, docLens, \
                 topicPrior, topicPriorSum,
                 z_dnk, 
                 oldMems, topicDists, 
                 diVocabDists, diSumVocabDists)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int infer_topics_f64(int d, int D, int K, \
                 int[:,:] W_list, int[:] docLens, \
                 int[:] X_ptr, int[:] X_indices, \
                 double[:] weights, double[:], factor, \
                 double[:] topicPrior, double topicPriorSum, \
                 double[:,:] z_dnk,  \
                 double[:] oldMeans, double[:,:] topicMeans, \
                 double[:,:] diVocabDists, double[:] diSumVocabDists):
    '''             
    Infers the topic assignments for a given document d with a fixed vocab and
    topic prior. The topicDists and z_dnk are mutated in-place. Everything else
    remains constant.
    '''
    cdef:
        int         k
        int         n
        int         innerItrs
        double      max  = -1E+300
        double      norm = 0.0
        double      epsilon = 0.01 / K
        double      post
        double      beta_kt
        double      diTopicDist
        double[:]   elemProd
        double      a

    
    # For each document reset the topic probabilities and iterate to
    # convergence. This means we don't have to store the per-token
    # topic probabilties z_dnk for all documents, which is a huge saving
    oldMeans[:]      = 1./K
    innerItrs = 0


    while ((innerItrs < MinInnerIters) or (l1_dist_f64 (oldMeans, topicMeans[d,:]) > epsilon)) \
    and (innerItrs < MaxInnerItrs):
        innerItrs += 1

        # Reset the topic distribution, convert the the old one into topic means.
        oldMeans        = topicMeans[d,:]
        topicMeans[d,:] = 0

        innerItrs += 1

        # Calculate the scaling factor for all n
        for p_idx in range(X_ptr[d], X_ptr[d+1]):
            p = X_indices[p_idx]
            elemProd = np.multiply (oldMeans[d,:], topicMeans[p,:])
            a = np.dot(weights, elemProd)
            a = normpdf_sf64(a) / probit_sf64(a)
            a /= docLens[d]
            factor += a * weights * topicMeans[p]


        # Determine the topic assignment for each individual token...
        for n in range(docLens[d]):
            norm = 0.0
            max  = -1E+300
            
            # Work in log-space to avoid underflow
            for k in range(K):
                diTopicDist = digamma(topicPrior[k] + docLens[d] * topicMeans[d,k])
                
                z_dnk[n,k] = diVocabDists[k,W_list[d,n]] - diSumVocabDists[k] + diTopicDist + factor[k]
                if z_dnk[n,k] > max:
                    max = z_dnk[n,k]

            # Scale before converting to standard space so inference is feasible
            for k in range(K):
                z_dnk[n,k] = exp(z_dnk[n,k] - max)
                norm += z_dnk[n,k]
         
            # Normalize the token probability, and check it's valid
            for k in range(K):
                z_dnk[n,k] /= norm
                if is_invalid(z_dnk[n,k]):
                    with gil:
                        print ("Inner iteration %d z_dnk[%d,%d] = %f, norm = %g" \
                               % (innerItrs, n, k, z_dnk[n,k], norm))

        # Use all the individual word topic assignments to determine
        # the topic mixture exhibited by this document
        for n in range(docLens[d]):
            for k in range(K):
                topicMeans[d,:] += z_dnk[n,:]
        topicMeans[d,:] /= docLens[d]

    return innerItrs


