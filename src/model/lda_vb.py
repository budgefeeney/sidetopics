# -*- coding: utf-8 -*-
'''
Implements a correlated topic model, similar to that described by Blei
but using the Bouchard product of sigmoid bounds instead of Laplace
approximation.

Created on 17 Jan 2014

@author: bryanfeeney
'''

import os # Configuration for PyxImport later on. Requires GCC
os.environ['CC']  = os.environ['HOME'] + "/bin/cc"

from math import log
from math import pi
from math import e

from collections import namedtuple
import numpy as np
import scipy.special as fns
import numpy.random as rd
import sys

import model.lda_cvb_fast as lda_cvb
import model.lda_vb_fast as compiled

from util.sparse_elementwise import sparseScalarProductOfSafeLnDot

# ==============================================================
# CONSTANTS
# ==============================================================

DTYPE=np.float32 # A default, generally we should specify this in the model setup

LN_OF_2_PI   = log(2 * pi)
LN_OF_2_PI_E = log(2 * pi * e)

DEBUG=False

MODEL_NAME="lda/vb"

# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')                            

QueryState = namedtuple ( \
    'QueryState', \
    'W_list docLens topicPriorPost topicDists'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K topicPrior vocabPrior wordDists dtype name'
)

# ==============================================================
# PUBLIC API
# ==============================================================

def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState(\
        model.K, \
        model.topicPrior, \
        model.vocabPrior, \
        None if model.wordDists is None else model.wordDists.copy(), \
        model.dtype,       \
        model.name)

def newModelAtRandom(W, K, topicPrior=None, vocabPrior=None, dtype=DTYPE):
    '''
    Creates a new LDA ModelState for the given training set and
    the given number of topics. Everything is instantiated purely
    at random. This contains all parameters independent of of
    the dataset (e.g. learnt priors)
    
    Param:
    W - the DxT document-term matrix of T terms in D documents
        which will be used for training.
    K - the number of topics
    topicPrior - the prior over topics, either a scalar or a K-dimensional vector
    vocabPrior - the prior over vocabs, either a scalar or a T-dimensional vector
    dtype      - the datatype to be used throughout.
    
    Return:
    A ModelState object
    '''
    assert K > 1, "There must be at least two topics"
    T = W.shape[1]
    
    if topicPrior is None:
        topicPrior = 50.0 / K # From Griffiths and Steyvers 2004
    if vocabPrior is None:
        vocabPrior = 0.01 # Also from G&S
        
    wordDists = np.empty((K,T), dtype=dtype)
    
    return ModelState(K, topicPrior, vocabPrior, wordDists, dtype, MODEL_NAME)


def newQueryState(W, modelState):
    '''
    Creates a new LDA QueryState object. This contains all
    parameters and random variables tied to individual
    datapoints.
    
    Param:
    W - the DxT document-term matrix used for training or
        querying.
    modelState - the model state object
    
    REturn:
    A CtmQueryState object
    '''
    K =  modelState.K
    
    D,_ = W.shape
    W_list, docLens = toWordList(W)
    maxN = int(np.max(docLens)) # bizarre Numpy 1.7 bug in rd.dirichlet/reshape
    
    # Initialise the per-token assignments at random according to the dirichlet hyper
    prior = contantVector((K,), modelState.topicPrior)
    topicDists = rd.dirichlet(prior, size=D)
    topicPosterior = modelState.topicPrior.copy()

    return QueryState(W_list, docLens, topicPosterior, topicDists)

def contantVector(shape, defaultValue):
    # return np.full(shape, defaultValue)
    result = np.ndarray(shape=shape)
    result.fill(defaultValue)
    return result

def toWordList (w_csr):
    docLens = np.squeeze(np.asarray(w_csr.sum(axis=1))).astype(np.int32)
    
    if w_csr.dtype == np.int32:
        return lda_cvb.toWordList_i32 (w_csr.indptr, w_csr.indices, w_csr.data, docLens), docLens
    elif w_csr.dtype == np.float32:
        return lda_cvb.toWordList_f32 (w_csr.indptr, w_csr.indices, w_csr.data, docLens), docLens
    elif w_csr.dtype == np.float64:
        return lda_cvb.toWordList_f64 (w_csr.indptr, w_csr.indices, w_csr.data, docLens), docLens
    else:
        raise ValueError("No implementation defined for dtype = " + str(w_csr.dtype))

def newTrainPlan(iterations=100, epsilon=0.01, logFrequency=10, fastButInaccurate=False, debug=DEBUG):
    '''
    Create a training plan determining how many iterations we
    process, how often we plot the results, how often we log
    the variational bound, etc.
    '''
    return TrainPlan(iterations, epsilon, logFrequency, fastButInaccurate, debug)

def train (W, X, modelState, queryState, trainPlan):
    '''
    Infers the topic distributions in general, and specifically for
    each individual datapoint.

    Params:
    W - the DxT document-term matrix
    X - The DxF document-feature matrix, which is IGNORED in this case
    modelState - the actual LDA model. In a training run (query = False) this
                 will be mutated in place, and then returned.
    queryState - the query results - essentially all the "local" variables
                 matched to the given observations. This will be mutated in-place
                 and then returned.
    trainPlan  - how to execute the training process (e.g. iterations,
                 log-interval etc.)
    query      - 

    Return:
    The updated model object (note parameters are updated in place, so make a 
    defensive copy if you want it)
    The query object with the update query parameters
    '''
    iterations, epsilon, logFrequency, fastButInaccurate, debug = \
        trainPlan.iterations, trainPlan.epsilon, trainPlan.logFrequency, trainPlan.fastButInaccurate, trainPlan.debug           
    W_list, docLens, topicPosterior, topicDists = \
        queryState.W_list, queryState.docLens, queryState.topicPosterior, queryState.topicDists
    K, topicPrior, vocabPrior, wordDists, dtype = \
        modelState.K, modelState.topicPrior, modelState.vocabPrior, modelState.vocabDists, modelState.dtype
    
    D,T = W.shape
    
    # Quick sanity check
    if np.any(docLens < 1):
        raise ValueError ("Input document-term matrix contains at least one document with no words")
    
    # Book-keeping for logs
    logPoints    = 1 if logFrequency == 0 else iterations // logFrequency
    boundIters   = np.zeros(shape=(logPoints,))
    boundValues  = np.zeros(shape=(logPoints,))
    likelyValues = np.zeros(shape=(logPoints,))
    bvIdx = 0
    
    # Instead of storing the full topic assignments for every individual word, we
    # re-estimate from scratch. I.e for the memberships z which is DxNxT in dimension,
    # we only store a 1xNxT = NxT part. 
    z_dnk = np.empty((docLens.max(), K), dtype=dtype)
 
    # Select the training iterations function appropriate for the dtype
    do_iterations = compiled.iterate_f32 \
                    if modelState.dtype == np.float32 \
                    else compiled.iterate_f32 # fixme...
    
    # Iterate in segments, pausing to take measures of the bound / likelihood
    segIters  = logFrequency
    remainder = iterations - segIters * (logPoints - 1)
    for segment in range(logPoints - 1):
        do_iterations (segIters, D, K, T, \
                 W_list, docLens, \
                 topicPrior, vocabPrior, \
                 z_dnk, topicDists, wordDists)
    
        boundIters[bvIdx]   = segment * segIters
        boundValues[bvIdx]  = 0
        likelyValues[bvIdx] = 0
        bvIdx += 1
    
    # Final batch of iterations.
    do_iterations (remainder, D, K, T, \
                 W_list, docLens, \
                 topicPrior, vocabPrior, \
                 z_dnk, topicDists, wordDists)
    
    boundIters[bvIdx]   = iterations - 1
    boundValues[bvIdx]  = 0
    likelyValues[bvIdx] = 0
   
            
    return ModelState(K, topicPrior, vocabPrior, wordDists, modelState.dtype, modelState.name), \
           QueryState(W_list, docLens, topicDists), \
           (boundIters, boundValues, likelyValues)
  
  
def var_bound_intermediate (W, model, query, n_kt, n_k):
    model = ModelState(\
        model.K, \
        model.topicPrior, \
        model.vocabPrior, \
        model.n_dk, \
        n_kt, \
        n_k, \
        model.dtype, \
        model.name)
    
    return var_bound (W, model, query)

def log_likely_intermediate (W, model, query, n_kt, n_k):
    model = ModelState(\
        model.K, \
        model.topicPrior, \
        model.vocabPrior, \
        model.n_dk, \
        n_kt, \
        n_k, \
        model.dtype, \
        model.name)
    
    return log_likelihood (W, model, query)

def query(W, X, modelState, queryState, queryPlan):
    '''
    Given a _trained_ model, attempts to predict the topics for each of
    the inputs.

    Params:
    W - The query words to which we assign topics
    X - This is ignored, and can be omitted
    modelState - the _trained_ model
    queryState - the query state generated for the query dataset
    queryPlan  - used in this case as we need to tighten up the approx

    Returns:
    The model state and query state, in that order. The model state is
    unchanged, the query is.
    '''
    modelState, queryState, _ = train(W, X, modelState, queryState, queryPlan, query=True)
    return modelState, queryState


def perplexity (W, modelState, queryState):
    '''
    Return the perplexity of this model.

    Perplexity is a sort of normalized likelihood, applicable to textual
    data. Specifically it's the reciprocal of the geometric mean of the
    likelihoods of each individual word in the corpus.
    '''
    return np.exp (-log_likelihood (W, modelState, queryState) / np.sum(W.data))


def log_likelihood (W, modelState, queryState):
    '''
    Return the log-likelihood of the given data W according to the model
    and the parameters inferred for the entries in W stored in the
    queryState object.
    
    Actually returns a vector of D document specific log likelihoods
    '''
    return sparseScalarProductOfSafeLnDot(W, queryState.topicDists, modelState.wordDists).sum()
    

def var_bound(W, modelState, queryState):
    '''
    Determines the variational bounds. Values are mutated in place, but are
    reset afterwards to their initial values. So it's safe to call in a serial
    manner.
    '''
    # Unpack the the structs, for ease of access and efficiency
    D,T   = W.shape
    K     = modelState.K
    n_kt  = modelState.n_kt
    n_dk  = queryState.n_dk
    n_k   = queryState.n_k
    z_dnk = queryState.z_dnk
    a     = modelState.topicPrior
    b     = modelState.vocabPrior
    
    docLens = queryState.docLens
    
    bound = 0
    
    # Expected value of the p(W,Z). Note everything else marginalized out, and
    # we're using a 0-th order Taylor expansion.
    try:
        bound += D * (fns.gammaln(K * a) - K * fns.gammaln(a))
        bound += K * (fns.gammaln(T * b) - T * fns.gammaln(b))
    
        bound -= np.sum (fns.gammaln(K * a + docLens))
        bound += np.sum (fns.gammaln(a + n_dk))
    
        bound -= np.sum (fns.gammaln(T * b + n_k))
        bound += np.sum (fns.gammaln(b + n_kt))
    
        # The entropy of z_dnk. Have to do this in a loop as z_dnk is
        # is jagged in it's third dimension.
        if modelState.dtype == np.float32:
            bound -= compiled.jagged_entropy_f32 (z_dnk, docLens)
        elif modelState.dtype == np.float64:
            bound -= compiled.jagged_entropy_f64 (z_dnk, docLens)
        else:
            raise ValueError ("No implementation defined for dtype " + str(modelState.dtype))
    except OverflowError:
        print("Overflow error encountered, returning zero")
        return 0
    
    return bound

def vocab(modelState):
    '''
    Return the vocabulary inferred by this model as a KxT matrix of T
    terms for each of the K topics
    '''
    return modelState.vocabDists



# ==============================================================
# PUBLIC HELPERS
# ==============================================================

def printStderr(msg):
    sys.stdout.flush()
    sys.stderr.write(msg + '\n')
    sys.stderr.flush()


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate

@static_var("old_bound", 0)
def _debug_with_bound (itr, var_value, var_name, W, K, topicMean, sigT, vocab, dtype, means, varcs, A, n):
    if np.isnan(var_value).any():
        printStderr ("WARNING: " + var_name + " contains NaNs")
    if np.isinf(var_value).any():
        printStderr ("WARNING: " + var_name + " contains INFs")
    if var_value.dtype != dtype:
        printStderr ("WARNING: dtype(" + var_name + ") = " + str(var_value.dtype))

    old_bound = _debug_with_bound.old_bound
    bound     = var_bound(W, ModelState(K, topicMean, sigT, vocab, A, dtype, MODEL_NAME), QueryState(means, varcs, n))
    diff = "" if old_bound == 0 else "%15.4f" % (bound - old_bound)
    _debug_with_bound.old_bound = bound

    if int(bound - old_bound) < 0:
        printStderr ("Iter %3d Update %-15s Bound %22f (%15s)" % (itr, var_name, bound, diff))
    else:
        print ("Iter %3d Update %-15s Bound %22f (%15s)" % (itr, var_name, bound, diff))

def _debug_with_nothing (itr, var_value, var_name, W, K, topicMean, sigT, vocab, dtype, means, varcs, A, n):
    pass
