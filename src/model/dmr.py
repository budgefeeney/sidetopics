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
import scipy.optimize as optim
import scipy.special as fns
import scipy.sparse as ssp
import time

import numba as nb

import model.dmr_fast as compiled

from util.misc import constantArray
from util.sparse_elementwise import sparseScalarProductOfSafeLnDot, sparseScalarProductOf

# ==============================================================
# CONSTANTS
# ==============================================================

LN_OF_2_PI   = log(2 * pi)
LN_OF_2_PI_E = log(2 * pi * e)

DEBUG=False

DTYPE = np.float64
MODEL_NAME = "dmr/gibbs_em"

Sigma = 0.01

# ==============================================================
# TUPLES
# ==============================================================


TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations burnIn thin weightUpdateInterval logFrequency debug')

QueryState = namedtuple ( \
    'QueryState', \
    'w_list z_list docLens topicSum numSamples'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K T weights topicPrior vocabPrior n_dk_samples topicSum vocabSum numSamples dtype name'
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
    T, F = data.words.shape[1], data.feats.shape[1]
    
    assert K > 1,     "There must be at least two topics"
    assert K < 256,   "There can be no more than 256 topics"
    assert T < 65536, "There can be no more than 65,536 unique words"
    
    if topicPrior is None:
        topicPrior = constantArray((K,), 50.0 / K, dtype=dtype) # From Griffiths and Steyvers 2004
    if type(topicPrior) == float or type(topicPrior) == int:
        topicPrior = constantArray((K,), topicPrior, dtype=dtype)
    if vocabPrior is None:
        vocabPrior = constantArray((T,), 0.1, dtype=dtype) # Also from G&S
        
    n_dk_samples = None # These start out at none until we actually
    topicSum     = None
    vocabSum     = None # go ahead and train this model.
    numSamples   = 0
    weights      = np.ones((K, F)) * 0.05
    
    return ModelState(K, T, weights, topicPrior, vocabPrior, n_dk_samples, topicSum, vocabSum, numSamples, dtype, MODEL_NAME)


def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState(\
        model.K, \
        model.T, \
        model.weights.copy(), \
        None if model.topicPrior is None else model.topicPrior.copy(), \
        model.vocabPrior, \
        None if model.n_dk_samples is None else model.n_dk_samples.copy(), \
        None if model.topicSum is None else model.topicSum.copy(), \
        None if model.vocabSum is None else model.vocabSum.copy(), \
        model.numSamples, \
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
    w_list, docLens = compiled.flatten(data.words)
    docLens = np.asarray(docLens, dtype=np.int32)
    if debug: print("Done")
    
    # Initialise the per-token assignments at random according to the dirichlet hyper
    if debug: print ("Sampling the {:,} ({:,}) per-token topic distributions... ".format(w_list.shape[0], docLens.sum()), end="")
    z_list = rd.randint(0, K, w_list.shape[0]).astype(np.uint8)
    if debug: print("Done")
    
    return QueryState(w_list, z_list, docLens, None, 0)


def newTrainPlan (iterations, burnIn = -1, thin = -1, weightUpdateInterval = -1, logFrequency = 100, fastButInaccurate=False, debug = False):
    if burnIn < 0:
        burnIn = iterations // 5
        iterations += burnIn

    if thin < 0:
        thin = 5 if iterations <= 100 \
            else 10 if iterations <= 1000 \
            else 50

    if weightUpdateInterval < 0:
        weightUpdateInterval = thin * 5

    return TrainPlan(iterations, burnIn, thin, weightUpdateInterval, logFrequency, debug)


def train (data, model, query, plan):
    iterations, burnIn, thin, weightUpdateInterval, _, debug = \
        plan.iterations, plan.burnIn, plan.thin, plan.weightUpdateInterval, plan.logFrequency, plan.debug
    w_list, z_list, docLens = \
        query.w_list, query.z_list, query.docLens
    K, T, weights, topicPrior, vocabPrior, _, _, _, dtype, name = \
        model.K, model.T, model.weights, model.topicPrior, model.vocabPrior, model.topicSum, model.vocabSum, model.numSamples, model.dtype, model.name
    
    assert model.dtype == np.float64, "This is only implemented for 64-bit floats"
    D = docLens.shape[0]
    X = data.feats
    assert docLens.max() < 65536, "This only works for documents with fewer than 65,536 words"

    ndk = np.zeros((D,K), dtype=np.uint16)
    nkv = np.zeros((K,T), dtype=np.int32)
    nk  = np.zeros((K,),  dtype=np.int32)

    num_samples = (iterations - burnIn) // thin
    n_dk_samples = np.zeros((D,K,num_samples), dtype=np.uint16)
    topicSum = np.zeros((D,K), dtype=dtype)
    vocabSum = np.zeros((K,T), dtype=dtype)
    
    compiled.initGlobalRng(0xC0FFEE)
    compiled.sumSuffStats(w_list, z_list, docLens, ndk, nkv, nk)
    
    # Burn in
    alphas = X.dot(weights.T)
    if debug: print ("Burning")
    compiled.sample (burnIn, burnIn + 1, w_list, z_list, docLens, \
            alphas, ndk, nkv, nk, n_dk_samples, topicSum, vocabSum, \
            vocabPrior, False, debug)
    
    # True samples
    if debug: print ("Training")
    sample_count = 0
    for _ in range(0, iterations - burnIn, weightUpdateInterval):
        alphas[:,:] = X.dot(weights.T)
        sample_count += compiled.sample (weightUpdateInterval, thin, w_list, z_list, docLens, \
                alphas, ndk, nkv, nk, n_dk_samples, topicSum, vocabSum, \
                vocabPrior, False, debug)

        # g = gradient(weights[0,:], 0, weights, sample_count, n_dk_samples, X, Sigma)
        # o = objective(weights[0,:], 0, weights, sample_count, n_dk_samples, X, Sigma)
        # o_old = objective_old(weights[0,:], 0, weights, sample_count, n_dk_samples, X, Sigma)

        updateWeights(n_dk_samples, sample_count, X, weights)
    
#     compiled.freeGlobalRng()
    
    return \
        ModelState (K, T, weights, topicPrior, vocabPrior, n_dk_samples, topicSum, vocabSum, num_samples, dtype, name), \
        QueryState (w_list, z_list, docLens, topicSum, num_samples), \
        (np.zeros(1), np.zeros(1), np.zeros(1))


def updateWeights(n_dk_samples, sample_count, X, weights):
    for k in range(weights.shape[0]):
        print ("Updating weights for topics %d" % k, end="... ")
        opt_result = optim.minimize(objective, weights[k,:], args=(k, weights, sample_count, n_dk_samples, X, Sigma), jac=gradient, method='L-BFGS-B', options={ "maxiter": 20}, bounds=[(1E-30, log(1E+30))] * weights.shape[1])

        if np.any(np.isnan(opt_result.x)) or np.any(np.isinf(opt_result.x)):
            print ("Returned a vector including NaNs or Infs... ")
        else:
            weights[k,:] = np.squeeze(np.asarray(opt_result.x))

        if not opt_result.success:
            print("Optimization error : %s " % opt_result.message)
        else:
            print("Done")

BatchSize=5000
def objective(weights, k, W, sample_count, n_dk_samples, X, sigma):
    D, K = X.shape[0], W.shape[0]
    result = 0.0

    alpha = np.empty((BatchSize, K), dtype=np.float64)
    for d in range(0, D, BatchSize):
        max_d = min(D, d + BatchSize)
        top   = max_d - d

        alpha[:top,:] = X[d:max_d,:].dot(W.T)
        alpha[:top,k] = X[d:max_d,:].dot(weights)
        np.exp(alpha[:top], out=alpha[:top])

        result += fns.gammaln(alpha[:top].sum(axis=1)).sum()
        result -= fns.gammaln(alpha[:top].sum(axis=1)[:,np.newaxis] + n_dk_samples[d:max_d,:,:sample_count].sum(axis=1)).sum() / sample_count
        result += fns.gammaln(alpha[:top,k,np.newaxis] + n_dk_samples[d:max_d,k,:sample_count]).sum() / sample_count
        result -= fns.gammaln(alpha[:top,k]).sum()

    result -= 0.5 / sigma * np.sum(weights * weights)

    return -result


def gradient(weights, k, W, sample_count, n_dk_samples, X, sigma):
    D, K = X.shape[0], W.shape[0]

    result = 0.0
    alpha = np.empty((BatchSize, K), dtype=np.float64)
    scale = np.empty((BatchSize,),   dtype=np.float64)
    for d in range(0, D, BatchSize):
        max_d = min(D, d + BatchSize)
        top   = max_d - d

        alpha[:top,:] = X[d:max_d,:].dot(W.T)
        alpha[:top,k] = X[d:max_d,:].dot(weights)
        np.exp(alpha[:top], out=alpha[:top])

        alpha_sum = alpha[:top].sum(axis=1)
        scale[:top]  = fns.digamma(alpha_sum)
        scale[:top] -= fns.digamma(alpha_sum[:,np.newaxis] + n_dk_samples[d:max_d,:,:sample_count].sum(axis=1)).sum(axis=1) / sample_count
        scale[:top] += fns.digamma(alpha[:top,k,np.newaxis] + n_dk_samples[d:max_d,k,:sample_count]).sum(axis=1) / sample_count
        scale[:top] -= fns.digamma(alpha[:top,k])

        P_1 = ssp.diags(alpha[:top,k], 0).dot(X[d:max_d,:])
        P_2 = ssp.diags(scale[:top], 0).dot(P_1)

        result += np.array(P_2.sum(axis=0))

    result -= weights / sigma

    return -np.squeeze(np.asarray(result))


def query (data, model, query, plan):
    iterations, burnIn, thin, _, debug = \
        plan.iterations, plan.burnIn, plan.thin, plan.logFrequency, plan.debug
    w_list, z_list, docLens, _, _ = \
        query.w_list, query.z_list, query.docLens, query.topicSum, query.numSamples
    K, T, weights, topicPrior, vocabPrior, _, _, _, dtype, name = \
        model.K, model.T, model.weights, model.topicPrior, model.vocabPrior, model.topicSum, model.vocabSum, model.numSamples, model.dtype, model.name
    
    assert model.dtype == np.float64, "This is only implements for 64-bit floats"
    D = docLens.shape[0]
    X = data.feats
    
    compiled.initGlobalRng(0xC0FFEE)

    num_samples = (iterations - burnIn) // thin
    n_dk_samples = np.zeros((D,K,num_samples), dtype=np.uint16)

    ndk = np.zeros((D, K), dtype=np.uint16)
    nkv = (wordDists(model) * 1000000).astype(np.int32)
    nk  = nkv.sum(axis=1).astype(np.int32)
    adjustedVocabPrior = np.zeros((T,), dtype=model.dtype) # already incorporated into nkv
    
    topicSum = np.zeros((D,K), dtype=dtype)
    vocabSum = model.vocabSum
    
    compiled.sumSuffStats(w_list, z_list, docLens, ndk, nkv, nk)
    
    # Burn in
    alphas = X.dot(weights.T)
    compiled.sample (burnIn, burnIn + 1, w_list, z_list, docLens, \
            alphas, ndk, nkv, nk, n_dk_samples, topicSum, vocabSum, \
            vocabPrior, True, debug)
    
    # True samples
    sample_count = compiled.sample (iterations, thin, w_list, z_list, docLens, \
            alphas, ndk, nkv, nk, n_dk_samples, topicSum, vocabSum, \
            vocabPrior, True, debug)
    
    return \
        ModelState (K, T, weights, topicPrior, vocabPrior, n_dk_samples, topicSum, vocabSum, num_samples, dtype, name), \
        QueryState (w_list, z_list, docLens, topicSum, num_samples)



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

    
def log_likelihood (data, model, query):
    '''
    Return the log-likelihood of the given data W according to the model
    and the parameters inferred for the entries in W stored in the
    queryState object.
    
    '''
    W = data.words if data.words.dtype is model.dtype else data.words.astype(model.dtype)
    return sparseScalarProductOfSafeLnDot(W, topicDists(query), wordDists(model)).sum()


def current_time_millis():
    return int(round(time.time() * 1000))