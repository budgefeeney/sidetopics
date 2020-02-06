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

import sidetopics.model.lda_gibbs_fast as compiled

from sidetopics.util.misc import constantArray
from sidetopics.util.sparse_elementwise import sparseScalarProductOfSafeLnDot
import sidetopics.model.lda_vb_python as lda_vb

# ==============================================================
# CONSTANTS
# ==============================================================

LN_OF_2_PI   = log(2 * pi)
LN_OF_2_PI_E = log(2 * pi * e)

DEBUG=False

DTYPE = np.float64
MODEL_NAME = "lda/gibbs"

VocabPrior = lda_vb.VocabPrior

# ==============================================================
# TUPLES
# ==============================================================


TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations burnIn thin logFrequency debug')

QueryState = namedtuple ( \
    'QueryState', \
    'w_list z_list docLens topicSum numSamples processed'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K T topicPrior vocabPrior topicSum vocabSum numSamples processed dtype name'
)

# ==============================================================
# PUBLIC API
# ==============================================================


def newModelAtRandom(data, K, topicPrior=None, vocabPrior=None, dtype=DTYPE):
    '''
    Creates a new LDA ModelState for the given training set and
    the given number of topics. Everything is instantiated purely
    at random. This contains all parameters independent of of
    the dataset (e.g. learnt priors)

    Param:
    data - the dataset of words, features and links of which only words are used in this model
    K - the number of topics
    topicPrior - the prior over topics, either a scalar or a K-dimensional vector
    vocabPrior - the prior over vocabs, either a scalar or a T-dimensional vector
    dtype      - the datatype to be used throughout.

    Return:
    A ModelState object
    '''
    T = data.words.shape[1]

    assert K > 1,     "There must be at least two topics"
    assert K < 256,   "There can be no more than 256 topics"
    assert T < 65536, "There can be no more than 65,536 unique words"

    if topicPrior is None:
        topicPrior = constantArray((K,), 50.0 / K, dtype=dtype) # From Griffiths and Steyvers 2004
    if type(topicPrior) == float or type(topicPrior) == int:
        topicPrior = constantArray((K,), topicPrior, dtype=dtype)
    if vocabPrior is None:
        vocabPrior = constantArray((T,), 0.1, dtype=dtype) # Also from G&S
    elif type(vocabPrior) is float:
        vocabPrior = constantArray((T,), vocabPrior, dtype=dtype) # Also from G&S

    topicSum  = None # These start out at none until we actually
    vocabSum  = None # go ahead and train this model.
    numSamples = 0

    return ModelState(K, T, topicPrior, vocabPrior, topicSum, vocabSum, numSamples, False, dtype, MODEL_NAME)


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
        model.processed, \
        model.dtype,      \
        model.name)

def newQueryState(data, modelState, debug=False):
    '''
    Creates a new LDA QueryState object. This contains all
    parameters and random variables tied to individual
    datapoints.

    Param:
    data - the dataset of words, features and links of which only words are used in this model
    modelState - the model state object

    Return:
    A query object
    '''
    K =  modelState.K

    D,T = data.words.shape
    if debug: print("Converting {:,}x{:,} document-term matrix to list of lists... ".format(D,T), end="")
    w_list, docLens = compiled.flatten(data.words.astype(np.int32))
    docLens = docLens.astype(np.int32)
    if debug: print("Done")

    # Initialise the per-token assignments at random according to the dirichlet hyper
    if debug: print ("Sampling the {:,} ({:,}) per-token topic distributions... ".format(w_list.shape[0], docLens.sum()), end="")
    z_list = rd.randint(0, K, w_list.shape[0]).astype(np.uint8)
    if debug: print("Done")

    return QueryState(w_list, z_list, docLens, None, 0, False)


def pruneQueryState(query, indices):
    '''
    Returns a query state corresponding to the given indices only
    '''
    return QueryState( \
        None, \
        None, \
        query.docLens[indices], \
        query.topicSum[indices, :], \
        query.numSamples,
        query.processed
        ) # TODO The Nones break this in many cases, but suffice for LDA-supported recommenders


def newTrainPlan(iterations, burnIn = -1, thin = -1, logFrequency = 100, fastButInaccurate=False, debug=False):
    if burnIn < 0:
        burnIn = iterations // 2

    if thin < 0:
        thin = 5 if iterations <= 100 \
            else 10 if iterations <= 1000 \
            else 50

    return TrainPlan(iterations + burnIn, burnIn, thin, logFrequency, debug)


def seed_rng(seed):
    rd.seed(0xC0FFEE)
    compiled.initGlobalRng(0xC0FFEE)


def train (_, model, query, plan):
    iterations, burnIn, thin, _, debug = \
        plan.iterations, plan.burnIn, plan.thin, plan.logFrequency, plan.debug
    w_list, z_list, docLens, _, _ = \
        query.w_list, query.z_list, query.docLens, query.topicSum, query.numSamples
    K, T, topicPrior, vocabPrior, _, _, _, dtype, name = \
        model.K, model.T, model.topicPrior, model.vocabPrior, model.topicSum, model.vocabSum, model.numSamples, model.dtype, model.name

    assert model.dtype == np.float64, "This is only implemented for 64-bit floats"
    D = docLens.shape[0]

    # These are appropriately initialised by compile.sumSuffStats() below
    ndk = np.zeros((D,K), dtype=np.int32)
    nkv = np.zeros((K,T), dtype=np.int32)
    nk  = np.zeros((K,),  dtype=np.int32)

    topicSum = np.zeros((D, K), dtype=dtype) # FIXME Check this.
    vocabSum = np.zeros((K, T), dtype=dtype)

    compiled.sumSuffStats(w_list, z_list, docLens, ndk, nkv, nk)

    # Burn in
    if debug: print ("Burning")
    compiled.sample (burnIn, burnIn + 1, w_list, z_list, docLens, \
            ndk, nkv, nk, topicSum, vocabSum, \
            topicPrior, vocabPrior, False, debug)

    # True samples
    if debug: print ("Sampling")
    numSamples = compiled.sample (iterations - burnIn, thin, w_list, z_list, docLens, \
            ndk, nkv, nk, topicSum, vocabSum, \
            topicPrior, vocabPrior, False, debug)

#     compiled.freeGlobalRng()

    return \
        ModelState (K, T, topicPrior, vocabPrior, topicSum, vocabSum, numSamples, True, dtype, name), \
        QueryState (w_list, z_list, docLens, topicSum, numSamples, True), \
        (np.zeros(1), np.zeros(1), np.zeros(1))


def query (data, model, query, plan):
    iterations, burnIn, thin, _, debug = \
        plan.iterations, plan.burnIn, plan.thin, plan.logFrequency, plan.debug
    w_list, z_list, docLens, _, _ = \
        query.w_list, query.z_list, query.docLens, query.topicSum, query.numSamples
    K, T, topicPrior, vocabPrior, _, _, _, dtype, name = \
        model.K, model.T, model.topicPrior, model.vocabPrior, model.topicSum, model.vocabSum, model.numSamples, model.dtype, model.name

    assert model.dtype == np.float64, "This is only implements for 64-bit floats"
    D = docLens.shape[0]

    ndk = np.zeros((D, K), dtype=np.int32)
    nkv = (wordDists(model) * 1000000).astype(np.int32)
    nk  = nkv.sum(axis=1).astype(np.int32)
    adjustedVocabPrior = np.zeros((T,), dtype=model.dtype) # already incorporated into nkv

    topicSum = np.zeros((D,K), dtype=dtype)
    vocabSum = model.vocabSum

    compiled.sumSuffStats(w_list, z_list, docLens, ndk, nkv, nk)

    # Burn in
    compiled.sample (burnIn, burnIn + 1, w_list, z_list, docLens, \
            ndk, nkv, nk, topicSum, vocabSum, \
            topicPrior, adjustedVocabPrior, True, debug)

    # True samples
    numSamples = compiled.sample (iterations - burnIn, thin, w_list, z_list, docLens, \
            ndk, nkv, nk, topicSum, vocabSum, \
            topicPrior, adjustedVocabPrior, True, debug)

    return \
        ModelState (K, T, topicPrior, vocabPrior, topicSum, vocabSum, numSamples, model.processed, dtype, name), \
        QueryState (w_list, z_list, docLens, topicSum, numSamples, True)



def topicDists(query):
    '''
    Returns the topic distribution for the given query
    '''
    topicDist = query.topicSum.copy()
    topicDist /= query.numSamples

    return topicDist

def wordDists(model):
    '''
    Returns the word distributions for the given query
    '''
    vocabDist = model.vocabSum.copy()
    vocabDist /= model.numSamples
    return vocabDist


def log_likelihood (data, model, query, topicDistOverride=None):
    '''
    Return the log-likelihood of the given data W according to the model
    and the parameters inferred for the entries in W stored in the
    queryState object.

    '''
    W = data.words if data.words.dtype is model.dtype else data.words.astype(model.dtype)
    tops = topicDistOverride \
        if topicDistOverride is not None \
        else topicDists(query)
    return sparseScalarProductOfSafeLnDot(W, tops, wordDists(model)).sum()
