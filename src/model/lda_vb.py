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
import numpy.random as rd
import scipy.special as fns
import sys
import time

import model.lda_vb_fast as compiled

from util.sparse_elementwise import sparseScalarProductOfSafeLnDot
from util.overflow_safe import safe_log
from util.misc import constantArray, converged, clamp
from model.lda_cvb import toWordList


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
    'W_list docLens topicDists'\
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
        model.topicPrior.copy(), \
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
        topicPrior = constantArray((K,), 50.0 / K, dtype) # From Griffiths and Steyvers 2004
    if vocabPrior is None:
        vocabPrior = 0.01 # Also from G&S
    
    vocabPriorVec = constantArray((T,), vocabPrior, dtype)
    wordDists = rd.dirichlet(vocabPriorVec, size=K).astype(dtype)
    
    # Peturb to avoid zero probabilities
    wordDists += 1./T
    wordDists /= (wordDists.sum(axis=1))[:,np.newaxis]
    
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
    
    Return:
    A CtmQueryState object
    '''
    D,_ = W.shape
    print("Converting Bag of Words matrix to List of List representation... ", end="")
    W_list, docLens = toWordList(W)
    print("Done")
    
    # Initialise the per-token assignments at random according to the dirichlet hyper
    topicDists = rd.dirichlet(modelState.topicPrior, size=D).astype(modelState.dtype)

    return QueryState(W_list, docLens, topicDists)


def newTrainPlan(iterations=100, epsilon=2, logFrequency=10, fastButInaccurate=False, debug=DEBUG):
    '''
    Create a training plan determining how many iterations we
    process, how often we plot the results, how often we log
    the variational bound, etc.
    
    epsilon is oddly measured, we just evaluate the angle of the line segment between
    the last value of the bound and the current, and if it's less than the given angle,
    then stop.
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
    W_list, docLens, topicDists = \
        queryState.W_list, queryState.docLens, queryState.topicDists
    K, topicPrior, vocabPrior, wordDists, dtype = \
        modelState.K, modelState.topicPrior, modelState.vocabPrior, modelState.wordDists, modelState.dtype
    
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
    z_dnk = np.empty((docLens.max(), K), dtype=dtype, order='F')
 
    # Select the training iterations function appropriate for the dtype
    current_micro_time = lambda: int(time.time())
    do_iterations = compiled.iterate_f32 \
                    if modelState.dtype == np.float32 \
                    else compiled.iterate_f64
#    do_iterations = iterate # pure Python
    
    # Iterate in segments, pausing to take measures of the bound / likelihood
    segIters  = logFrequency
    remainder = iterations - segIters * (logPoints - 1)
    totalItrs = 0
    for segment in range(logPoints - 1):
        start = current_micro_time()
        totalItrs += do_iterations (segIters, D, K, T, \
                 W_list, docLens, \
                 topicPrior, vocabPrior, \
                 z_dnk, topicDists, wordDists)
        
        duration = current_micro_time() - start
    
        boundIters[bvIdx]   = segment * segIters
        boundValues[bvIdx]  = var_bound(W, modelState, queryState)
        likelyValues[bvIdx] = log_likelihood(W, modelState, queryState)
        bvIdx += 1
        
        if converged (boundIters, boundValues, bvIdx, epsilon, minIters=5):
            boundIters, boundValues, likelyValues = clamp (boundIters, boundValues, likelyValues, bvIdx)
            return ModelState(K, topicPrior, vocabPrior, wordDists, modelState.dtype, modelState.name), \
                QueryState(W_list, docLens, topicDists), \
                (boundIters, boundValues, likelyValues)
        
        print ("Segment %d/%d Total Iterations %d Duration %d Bound %10.2f Likelihood %10.2f" % (segment, logPoints, totalItrs, duration, boundValues[bvIdx - 1], likelyValues[bvIdx - 1]))
    
    # Final batch of iterations.
    do_iterations (remainder, D, K, T, \
                 W_list, docLens, \
                 topicPrior, vocabPrior, \
                 z_dnk, topicDists, wordDists)
    
    boundIters[bvIdx]   = iterations - 1
    boundValues[bvIdx]  = var_bound(W, modelState, queryState)
    likelyValues[bvIdx] = log_likelihood(W, modelState, queryState)
   
            
    return ModelState(K, topicPrior, vocabPrior, wordDists, modelState.dtype, modelState.name), \
           QueryState(W_list, docLens, topicDists), \
           (boundIters, boundValues, likelyValues)

MaxInnerItrs = 20
MinInnerItrs = 3
#@jit never really worked very well...
def iterate (iterations, D, K, T, \
             W_list, docLens, \
             topicPrior, vocabPrior, \
             z_dnk, topicDists, wordDists):
    
    totalItrs = 0
    epsilon = 0.01 / K
    oldWordDists = np.empty(wordDists.shape, wordDists.dtype)
    newWordDists = wordDists
    
    
    for _ in range(iterations):
        oldWordDists, newWordDists = newWordDists, oldWordDists
        lnWordDists = safe_log(oldWordDists, out=oldWordDists)
        newWordDists.fill (vocabPrior)
        
        for d in range(D):
            oldTopics = topicDists[d,:].copy()
            topicDists[d,:]= 1./ K
            lnWordProbs = lnWordDists[:,W_list[d,0:docLens[d]]]
            
            innerItrs = 0
            while ((innerItrs < MaxInnerItrs) or (np.sum(np.abs(oldTopics - topicDists[d,:])) > epsilon)) \
            and (innerItrs < MaxInnerItrs):
                diTopic     = fns.digamma(topicDists[d,:])
                z_dnk[:docLens[d],:] = lnWordProbs.T + diTopic[np.newaxis,:]
                
                # We've been working in log-space till now, before we go to true
                # probability space rescale so we don't underflow everywhere
                maxes  = z_dnk.max(axis=1)
                z_dnk -= maxes[:,np.newaxis]
                np.exp(z_dnk, out=z_dnk)
                
                # Now normalize so probabilities sum to one
                sums   = z_dnk.sum(axis=1)
                z_dnk /= sums[:,np.newaxis]            # Update vocabulary: hard to do with a list representation

                # Now use it to infer the topic distribution
                topicDists[d,:] = topicPrior + np.sum(z_dnk[:docLens[d],:], axis=0)
                topicDists[d,:] /= np.sum(topicDists[d,:])
                
                innerItrs += 1
            
            totalItrs += innerItrs
            for k in range(K):
                for n in range(docLens[d]):
                    newWordDists[k,W_list[d,n]] += z_dnk[n,k]
            newWordDists /= newWordDists.sum(axis=1)[:,np.newaxis]
        
    return totalItrs



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
    W_list, docLens, topicDists = \
        queryState.W_list, queryState.docLens, queryState.topicDists
    K, topicPrior, wordDists, dtype = \
        modelState.K, modelState.topicPrior, modelState.wordDists, modelState.dtype
   
    D,T = W.shape
    z_dnk = np.empty((docLens.max(), K), dtype=dtype, order='F')
    
    
    if modelState.dtype == np.float32:
        for _ in range(queryPlan.iterations):
            compiled.query_f32 (D, K, \
                     W_list, docLens, \
                     topicPrior, z_dnk, topicDists, 
                     wordDists)
    else:
        for _ in range(queryPlan.iterations):
            compiled.query_f64 (D, K, \
                     W_list, docLens, \
                     topicPrior, z_dnk, topicDists, 
                     wordDists)
        
   
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
    

def var_bound(W, modelState, queryState, z_dnk = None):
    '''
    Determines the variational bounds.
    '''
    # Unpack the the structs, for ease of access and efficiency
    W_list, docLens, topicDists = \
        queryState.W_list, queryState.docLens, queryState.topicDists
    K, topicPrior, vocabPrior, wordDists, dtype = \
        modelState.K, modelState.topicPrior, modelState.vocabPrior, modelState.wordDists, modelState.dtype
    
    D,T = W.shape
    maxN = docLens.max()
    if z_dnk == None:
        z_dnk = np.empty(shape=(maxN, K), dtype=dtype)
        
    diWordDists = fns.digamma(wordDists.copy()) - fns.digamma(wordDists.sum(axis=1))[:,np.newaxis]
    lnWordDists = np.log(wordDists, out=wordDists)
   
    bound = 0
    
    # Expected Probablity
    #
    
    # P(topics|topicPrior)
    diTopicDists = fns.digamma(topicDists) - fns.digamma(topicDists.sum(axis=1))[:,np.newaxis]
    ln_b_topic = fns.gammaln(topicPrior.sum()) - fns.gammaln(topicPrior).sum()
    bound += D * ln_b_topic \
           + np.sum((topicPrior - 1) * diTopicDists)
    
    # and its entropy
    ent = fns.gammaln(topicDists.sum(axis=1)).sum() - fns.gammaln(topicDists).sum() \
        + np.sum ((topicDists - 1) * diTopicDists)
    
    bound -= ent
    
    # P(z|topic) is tricky as we don't actually store this. However
    # we make a single, simple estimate for this case.
    # NOTE COPY AND PASTED FROM iterate_f32  / iterate_f64 (-ish)
    for d in range(D):
        lnWordProbs = lnWordDists[:,W_list[d,0:docLens[d]]]
        diTopic     = fns.digamma(topicDists[d,:])
        z_dnk[0:docLens[d],:] = lnWordProbs.T + diTopic[np.newaxis,:]
        
        # We've been working in log-space till now, before we go to true
        # probability space rescale so we don't underflow everywhere
        maxes  = z_dnk.max(axis=1)
        z_dnk -= maxes[:,np.newaxis]
        np.exp(z_dnk, out=z_dnk)
        
        # Now normalize so probabilities sum to one
        sums   = z_dnk.sum(axis=1)
        z_dnk /= sums[:,np.newaxis]
#        z_dnk[docLens[d]:maxN,:] = 0 # zero probablities for words that don't exist
        
        # Now use to calculate  E[ln p(Z|topics), E[ln p(W|Z) and H[Z] in that order
        diTopic -= fns.digamma(np.sum(topicDists[d,:]))
        bound += np.sum(z_dnk * diTopic[np.newaxis,:])
        bound += np.sum(z_dnk[0:docLens[d],:].T * diWordDists[:,W_list[d,0:docLens[d]]])
        bound -= np.sum(z_dnk[0:docLens[d],:] * safe_log(z_dnk[0:docLens[d],:]))
        
    # p(vocabDists|vocabPrior)
    wordDists = np.exp(lnWordDists, out=lnWordDists)
    
    ln_b_vocab = fns.gammaln(T * vocabPrior) - T * fns.gammaln(vocabPrior)
    bound += K * ln_b_vocab \
           + (vocabPrior - 1) * np.sum(diWordDists)
    
    # and its entropy
    ent = fns.gammaln(wordDists.sum(axis=1)).sum() - fns.gammaln(wordDists).sum() \
        + np.sum ((wordDists - 1) * diWordDists)
    
    bound -= ent   
    
    return bound

def vocab(modelState):
    '''
    Return the vocabulary inferred by this model as a KxT matrix of T
    terms for each of the K topics
    '''
    return modelState.wordDists



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
