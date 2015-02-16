# -*- coding: utf-8 -*-
'''
Implements a correlated topic model, similar to that described by Blei
but using the Bouchard product of sigmoid bounds instead of Laplace
approximation.

Created on 17 Jan 2014

@author: bryanfeeney
'''

import os # Configuration for PyxImport later on. Requires GCC
import sys
os.environ['CC']  = os.environ['HOME'] + "/bin/cc"

from math import log
from math import pi
from math import e


from collections import namedtuple
import numpy as np
import scipy.special as fns
import scipy.linalg as la
import numpy.random as rd
import sys

import model.lda_gibbs_fast as compiled

from util.misc import constantArray
from util.sparse_elementwise import sparseScalarProductOfSafeLnDot

# ==============================================================
# CONSTANTS
# ==============================================================

LN_OF_2_PI   = log(2 * pi)
LN_OF_2_PI_E = log(2 * pi * e)

DEBUG=False

DTYPE = np.float64
MODEL_NAME = "lda/gibbs"

# ==============================================================
# TUPLES
# ==============================================================


TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations burnIn thin logFrequency debug')                            

QueryState = namedtuple ( \
    'QueryState', \
    'w_list z_list docLens topicSum numSamples'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K T topicPrior vocabPrior topicSum vocabSum numSamples dtype name'
)

# ==============================================================
# PUBLIC API
# ==============================================================


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
    T = W.shape[1]
    
    assert K > 1,     "There must be at least two topics"
    assert K < 256,   "There can be no more than 256 topics"
    assert T < 65536, "There can be no more than 65,536 unique words"
    
    if topicPrior is None:
        topicPrior = constantArray((K,), 50.0 / K, dtype=dtype) # From Griffiths and Steyvers 2004
    if type(topicPrior) == float or type(topicPrior) == int:
        topicPrior = constantArray((K,), topicPrior, dtype=dtype)
    if vocabPrior is None:
        vocabPrior = constantArray((T,), 0.1, dtype=dtype) # Also from G&S
        
    topicSum  = None # These start out at none until we actually
    vocabSum  = None # go ahead and train this model.
    numSamples = 0
    
    return ModelState(K, T, topicPrior, vocabPrior, topicSum, vocabSum, numSamples, dtype, MODEL_NAME)


def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState(\
        model.K, \
        model.T, \
        None if model.topicPrior is None else model.topicPrior.copy(), \
        model.vocabPrior, \
        None if model.topicSum is None else model.topicSum.copy(), \
        None if model.vocabSum is None else model.vocabSum.copy(), \
        model.numSamples, \
        model.dtype,      \
        model.name)

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
    A query object
    '''
    K =  modelState.K
    
    D,T = W.shape
    print("Converting {:,}x{:,} document-term matrix to list of lists... ".format(D,T), end="")
    w_list, docLens = compiled.flatten(W)
    print("Done")
    
    
    # Initialise the per-token assignments at random according to the dirichlet hyper
    print ("Sampling the {:,} ({:,}) per-token topic distributions... ".format(w_list.shape[0], docLens.sum()), end="")
    z_list = rd.randint(0, K, w_list.shape[0]).astype(np.uint8)
    print("Done")
    
    return QueryState(w_list, z_list, docLens, None, 0)


def newTrainPlan (iterations, burnIn, thin = 10, logFrequency = 100, debug = False):
    return TrainPlan(iterations, burnIn, thin, logFrequency, debug)


def train (W, X, model, query, plan):
    iterations, burnIn, thin, _, _ = \
        plan.iterations, plan.burnIn, plan.thin, plan.logFrequency, plan.debug
    w_list, z_list, docLens, _, _ = \
        query.w_list, query.z_list, query.docLens, query.topicSum, query.numSamples
    K, T, topicPrior, vocabPrior, _, _, _, dtype, name = \
        model.K, model.T, model.topicPrior, model.vocabPrior, model.topicSum, model.vocabSum, model.numSamples, model.dtype, model.name
    
    assert model.dtype == np.float64, "This is only implemented for 64-bit floats"
    D = docLens.shape[0]
    
    ndk = np.zeros((D,K), dtype=np.int32)
    nkv = np.zeros((K,T), dtype=np.int32)
    nk  = np.zeros((K,),  dtype=np.int32)
    
    topicSum = np.zeros((D,K), dtype=dtype)
    vocabSum = np.zeros((K,T), dtype=dtype)
    
    compiled.initGlobalRng(0xC0FFEE)
    compiled.sumSuffStats(w_list, z_list, docLens, ndk, nkv, nk)
    
    # Burn in
    print ("Burning")
    compiled.sample (burnIn, burnIn + 1, w_list, z_list, docLens, \
            ndk, nkv, nk, topicSum, vocabSum, \
            topicPrior, vocabPrior, False)
    
    # True samples
    print ("Sampling")
    numSamples = compiled.sample (iterations - burnIn, thin, w_list, z_list, docLens, \
            ndk, nkv, nk, topicSum, vocabSum, \
            topicPrior, vocabPrior, False)
    
#     compiled.freeGlobalRng()
    
    return \
        ModelState (K, T, topicPrior, vocabPrior, topicSum, vocabSum, numSamples, dtype, name), \
        QueryState (w_list, z_list, docLens, topicSum, numSamples), \
        (np.zeros(1), np.zeros(1), np.zeros(1))


def query (W, X, model, query, plan):
    iterations, burnIn, thin, _, _ = \
        plan.iterations, plan.burnIn, plan.thin, plan.logFrequency, plan.debug
    w_list, z_list, docLens, _, _ = \
        query.w_list, query.z_list, query.docLens, query.topicSum, query.numSamples
    K, T, topicPrior, vocabPrior, _, _, _, dtype, name = \
        model.K, model.T, model.topicPrior, model.vocabPrior, model.topicSum, model.vocabSum, model.numSamples, model.dtype, model.name
    
    assert model.dtype == np.float64, "This is only implements for 64-bit floats"
    D = docLens.shape[0]
    
    compiled.setGlobalRngSeed(0xC0FFEE)
    
    ndk = model.topicSum.copy()
    nkv = model.vocabSum.copy()
    nk  = np.zeros((K,),  dtype=np.int32)
    
    topicSum = np.zeros((D,K), dtype=dtype)
    vocabSum = model.vocabSum
    
    compiled.sumSuffStats(w_list, z_list, docLens, ndk, nkv, nk)
    
    # Burn in
    compiled.sample (burnIn, burnIn + 1, w_list, z_list, docLens, \
            ndk, nkv, nk, topicSum, vocabSum, \
            topicPrior, vocabPrior, True)
    
    # True samples
    numSamples = compiled.sample (iterations - burnIn, thin, w_list, z_list, docLens, \
            ndk, nkv, nk, topicSum, vocabSum, \
            topicPrior, vocabPrior, True)
    
    return \
        ModelState (K, T, topicPrior, vocabPrior, topicSum, vocabSum, numSamples, dtype, name), \
        QueryState (w_list, z_list, docLens, topicSum, numSamples), \
        (np.zeros(1), np.zeros(1), np.zeros(1))


def topicDist(query):
    '''
    Returns the topic distribution for the given query
    ''' 
    topicDist = query.topicSum.copy()
    topicDist /= query.numSamples
    return topicDist


def vocab(model):
    '''
    Returns the word distributions for the given query
    ''' 
    vocabDist = model.vocabSum.copy()
    vocabDist /= model.numSamples
    return vocabDist

    
def log_likelihood (W, model, query):
    '''
    Return the log-likelihood of the given data W according to the model
    and the parameters inferred for the entries in W stored in the
    queryState object.
    
    '''
    return sparseScalarProductOfSafeLnDot(W, topicDist(query), vocab(model)).sum()

def perplexity (W, modelState, queryState):
    '''
    Return the perplexity of this model.

    Perplexity is a sort of normalized likelihood, applicable to textual
    data. Specifically it's the reciprocal of the geometric mean of the
    likelihoods of each individual word in the corpus.
    '''
    return np.exp (-log_likelihood (W, modelState, queryState) / np.sum(W.data))

