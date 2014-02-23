# -*- coding: utf-8 -*-
'''
Implements a correlated topic model, similar to that described by Blei
but using the Bouchard product of sigmoid bounds instead of Laplace
approximation.

Created on 17 Jan 2014

@author: bryanfeeney
'''

from math import log
from math import pi
from math import e

from collections import namedtuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as ssp
import numpy.random as rd
import sys

from util.overflow_safe import safe_log_one_plus_exp_of
from util.array_utils import normalizerows_ip
from util.sigmoid_utils import rowwise_softmax
from util.sparse_elementwise import sparseScalarQuotientOfDot, \
    sparseScalarProductOfSafeLnDot, scaledSumOfLnOnePlusExp
    
# ==============================================================
# CONSTANTS
# ==============================================================

DTYPE=np.float32 # A default, generally we should specify this in the model setup

LN_OF_2_PI   = log(2 * pi)
LN_OF_2_PI_E = log(2 * pi * e)

DEBUG=False

MODEL_NAME="ctm/bouchard"

# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')                            

QueryState = namedtuple ( \
    'QueryState', \
    'means varcs lxi s docLens'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K topicMean sigT vocab dtype name'
)

# ==============================================================
# PUBLIC API
# ==============================================================

def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState(model.K, model.topicMean.copy(), model.sigT.copy(), model.vocab.copy(), model.dtype, model.name)

def newModelAtRandom(W, K, dtype=DTYPE):
    '''
    Creates a new CtmModelState for the given training set and
    the given number of topics. Everything is instantiated purely
    at random. This contains all parameters independent of of
    the dataset (e.g. learnt priors)
    
    Param:
    W - the DxT document-term matrix of T terms in D documents
        which will be used for training.
    K - the number of topics
    
    Return:
    A CtmModelState object
    '''
    assert K > 1, "There must be at least two topics"
    
    _,T = W.shape
    vocab     = normalizerows_ip(rd.random((K,T)).astype(dtype))
    topicMean = rd.random((K,)).astype(dtype)
    topicMean /= np.sum(topicMean)
    
#    isigT = np.eye(K)
#    sigT  = la.inv(isigT)
    sigT  = np.eye(K, dtype=dtype)
    
    return ModelState(K, topicMean, sigT, vocab, dtype, MODEL_NAME)

def newQueryState(W, modelState):
    '''
    Creates a new CTM Query state object. This contains all
    parameters and random variables tied to individual
    datapoints.
    
    Param:
    W - the DxT document-term matrix used for training or
        querying.
    modelState - the model state object
    
    REturn:
    A CtmQueryState object
    '''
    K, vocab, dtype =  modelState.K, modelState.vocab, modelState.dtype
    
    D,T = W.shape
    assert T == vocab.shape[1], "The number of terms in the document-term matrix (" + str(T) + ") differs from that in the model-states vocabulary parameter " + str(vocab.shape[1])
    docLens = np.squeeze(np.asarray(W.sum(axis=1)))
    
    means = normalizerows_ip(rd.random((D,K)).astype(dtype))
    varcs = np.ones((D,K), dtype=dtype)
    
    s = np.ndarray(shape=(D,), dtype=dtype)
    s.fill(0)
    
    lxi = negJakkolaOfDerivedXi(means, varcs, s)
    
    return QueryState(means, varcs, lxi, s, docLens)


def newTrainPlan(iterations = 100, epsilon=0.01, logFrequency=10, fastButInaccurate=False, debug=DEBUG):
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
    modelState - the actual CTM model
    queryState - the query results - essentially all the "local" variables
                 matched to the given observations
    trainPlan  - how to execute the training process (e.g. iterations,
                 log-interval etc.)
                 
    Return:
    A new model object with the updated model (note parameters are
    updated in place, so make a defensive copy if you want it)
    A new query object with the update query parameters
    '''
    D,_ = W.shape
    
    # Unpack the the structs, for ease of access and efficiency
    iterations, epsilon, logFrequency, diagonalPriorCov, debug = trainPlan.iterations, trainPlan.epsilon, trainPlan.logFrequency, trainPlan.fastButInaccurate, trainPlan.debug
    means, varcs, lxi, s, n = queryState.means, queryState.varcs, queryState.lxi, queryState.s, queryState.docLens
    K, topicMean, sigT, vocab, dtype = modelState.K, modelState.topicMean, modelState.sigT, modelState.vocab, modelState.dtype
    
    # Book-keeping for logs
    boundIters  = np.zeros(shape=(iterations // logFrequency,))
    boundValues = np.zeros(shape=(iterations // logFrequency,))
    bvIdx = 0
    debugFn = _debug_with_bound if debug else _debug_with_nothing
    
    # Initialize some working variables
    isigT = la.inv(sigT)
    R = W.copy()
    
    s.fill(0)
    priorSigt_diag = np.ndarray(shape=(K,), dtype=dtype)
    priorSigt_diag.fill (0.001)
    
    # Iterate over parameters
    for itr in range(iterations):
        
        # We start with the M-Step, so the parameters are consistent with our
        # initialisation of the RVs when we do the E-Step
        
        # Update the mean and covariance of the prior
        topicMean = means.mean(axis = 0)
        debugFn (itr, topicMean, "topicMean", W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n)
        
        sigT = np.cov(means.T) if sigT.dtype == np.float64 else np.cov(means.T).astype(dtype)
        sigT.flat[::K+1] += varcs.mean(axis=0)
        if diagonalPriorCov:
            diag = np.diag(sigT)
            sigT = np.diag(diag)
            isigT = np.diag(1./ diag)
        else:
            isigT = la.inv(sigT)
        debugFn (itr, sigT, "sigT", W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n)
        
        
        # Building Blocks - termporarily replaces means with exp(means)
        expMeans = np.exp(means, out=means)
        R = sparseScalarQuotientOfDot(W, expMeans, vocab, out=R)
        V = expMeans * R.dot(vocab.T)
        
        # Update the vocabulary
        vocab *= (R.T.dot(expMeans)).T # Awkward order to maintain sparsity (R is sparse, expMeans is dense)
        vocab = normalizerows_ip(vocab)
        vocab += 1E-30 if dtype == np.float32 else 1E-300
        
        # Reset the means to their original form, and log effect of vocab update
        means = np.log(expMeans, out=expMeans)
        debugFn (itr, vocab, "vocab", W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n)
        
        # And now this is the E-Step, though it's followed by updates for the
        # parameters also that handle the log-sum-exp approximation.
        
        # Update the Variances
        varcs = 1./(2 * n[:,np.newaxis] * lxi + isigT.flat[::K+1])
        debugFn (itr, varcs, "varcs", W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n)
        
        # Update the Means
        vMat   = (2  * s[:,np.newaxis] * lxi - 0.5) * n[:,np.newaxis] + V
        rhsMat = vMat + isigT.dot(topicMean)
        if diagonalPriorCov:
            means = varcs * rhsMat
        else:
            for d in range(D):
                means[d,:] = la.inv(isigT + ssp.diags(n[d] * 2 * lxi[d,:], 0)).dot(rhsMat[d,:])
        
        vMat   = (2  * s[:,np.newaxis] * lxi - 0.5) * n[:,np.newaxis] + V
        rhsMat = vMat + isigT.dot(topicMean)
        for d in range(D):
            means[d,:] = la.inv(isigT + ssp.diags(n[d] * 2 * lxi[d,:], 0)).dot(rhsMat[d,:])
        debugFn (itr, means, "means", W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n)
        
        # Update the approximation parameters
        lxi = negJakkolaOfDerivedXi(means, varcs, s)
        debugFn (itr, lxi, "lxi", W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n)
        
        # s can sometimes grow unboundedly
        # Follow Bouchard's suggested approach of fixing it at zero
        #
        s = (np.sum(lxi * means, axis=1) + 0.25 * K - 0.5) / np.sum(lxi, axis=1)
        debugFn (itr, s, "s", W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n)
        
        if logFrequency > 0 and itr % logFrequency == 0:
            modelState = ModelState(K, topicMean, sigT, vocab, dtype, MODEL_NAME)
            queryState = QueryState(means, varcs, lxi, s, n)
            
            boundValues[bvIdx] = var_bound(W, modelState, queryState)
            boundIters[bvIdx]  = itr
            print ("\nIteration %d: bound %f" % (itr, boundValues[bvIdx]))
            if bvIdx > 0 and  boundValues[bvIdx - 1] > boundValues[bvIdx]:
                printStderr ("ERROR: bound degradation: %f > %f" % (boundValues[bvIdx - 1], boundValues[bvIdx]))
#             print ("Means: min=%f, avg=%f, max=%f\n\n" % (means.min(), means.mean(), means.max()))
            bvIdx += 1
    
    return \
        ModelState(K, topicMean, sigT, vocab, dtype, MODEL_NAME), \
        QueryState(means, varcs, lxi, s, n), \
        (boundIters, boundValues)
    

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
    D = W.shape[0]
    
    iterations, epsilon, logFrequency, fastButInaccurate, debug = queryPlan.iterations, queryPlan.epsilon, queryPlan.logFrequency, queryPlan.fastButInaccurate, queryPlan.debug
    means, varcs, lxi, s, n = queryState.means, queryState.varcs, queryState.lxi, queryState.s, queryState.docLens
    K, topicMean, sigT, vocab, dtype = modelState.K, modelState.topicMean, modelState.sigT, modelState.vocab, modelState.dtype
    
    # Necessary temp variables (notably the count of topic to word assignments
    # per topic per doc)
    isigT = la.inv(sigT)
    expMeans = np.exp(means, out=means) # Do in-place to save memory
    R = sparseScalarQuotientOfDot(W, expMeans, vocab)
    S = expMeans * R.dot(vocab.T)
    means = np.log(expMeans, out=expMeans) # Revert in-place exp()
        
    # Enable logging or not. If enabled, we need the inner product of the feat matrix
    debugFn = _debug_with_bound if debug else _debug_with_nothing
    
    # Iterate over parameters
    for itr in range(iterations):
        # Update the Means
        vMat   = (2  * s[:,np.newaxis] * lxi - 0.5) * n[:,np.newaxis] + S
        rhsMat = vMat + isigT.dot(topicMean)
        for d in range(D):
            try:
                means[d,:] = la.inv(isigT + ssp.diags(n[d] * 2 * lxi[d,:], 0)).dot(rhsMat[d,:])
            except ValueError as e:
                print(str(e))
                print ("Ah")
        debugFn (itr, means, "means", W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n)
        
        # Update the Variances
        varcs = 1./(2 * n[:,np.newaxis] * lxi + isigT.flat[::K+1])
        debugFn (itr, varcs, "varcs", W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n)
        
        # Update the approximation parameters
        lxi = negJakkolaOfDerivedXi(means, varcs, s)
        debugFn (itr, lxi, "lxi", W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n)
        
        # s can sometimes grow unboundedly
        # Follow Bouchard's suggested approach of fixing it at zero
        #
        s = (np.sum(lxi * means, axis=1) + 0.25 * K - 0.5) / np.sum(lxi, axis=1)
        debugFn (itr, s, "s", W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n)
        
    return modelState, QueryState (means, varcs, lxi, s, n)


def verifyProper(X, xName):
    '''
    Checks there's no NaNs or Infs
    '''
    if np.isnan(X).any():
        print (xName + " contains NaNs")
    if np.isinf(X).any():
        print (xName + " contains Infs")

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
    '''
    return np.sum( \
        sparseScalarProductOfSafeLnDot(\
            W, \
            rowwise_softmax(queryState.means), \
            modelState.vocab \
        ).data \
    )
    
def var_bound(W, modelState, queryState):
    '''
    Determines the variational bounds. Values are mutated in place, but are
    reset afterwards to their initial values. So it's safe to call in a serial
    manner.
    '''
    
    # Unpack the the structs, for ease of access and efficiency
    D,_ = W.shape
    means, varcs, lxi, s, docLens = queryState.means, queryState.varcs, queryState.lxi, queryState.s, queryState.docLens
    K, topicMean, sigT, vocab     = modelState.K, modelState.topicMean, modelState.sigT, modelState.vocab
    
    # Calculate some implicit  variables
    xi = _deriveXi(means, varcs, s)
    isigT = la.inv(sigT)
    
    bound = 0
    
    # Distribution over document topics
    bound -= (D*K)/2. * LN_OF_2_PI
    bound -= D/2. * la.det(sigT)
    diff   = means - topicMean[np.newaxis,:]
    bound -= 0.5 * np.sum (diff.dot(isigT) * diff)
    bound -= 0.5 * np.sum(varcs * np.diag(isigT)[np.newaxis,:]) # = -0.5 * sum_d tr(V_d \Sigma^{-1}) when V_d is diagonal only.
       
    # And its entropy
    bound += 0.5 * D * K * LN_OF_2_PI_E + 0.5 * np.sum(np.log(varcs)) 
    
    # Distribution over word-topic assignments
    # This also takes into account all the variables that 
    # constitute the bound on log(sum_j exp(mean_j)) and
    # also incorporates the implicit entropy of Z_dvk
    bound -= np.sum((means*means + varcs*varcs) * docLens[:,np.newaxis] * lxi)
    bound += np.sum(means * 2 * docLens[:,np.newaxis] * s[:,np.newaxis] * lxi)
    bound += np.sum(means * -0.5 * docLens[:,np.newaxis])
    # The last term of line 1 gets cancelled out by part of the first term in line 2
    # so neither are included here
    
    expMeans = np.exp(means, out=means)
    bound -= -np.sum(sparseScalarProductOfSafeLnDot(W, expMeans, vocab).data)
    
    bound -= np.sum(docLens[:,np.newaxis] * lxi * ((s*s)[:,np.newaxis] - (xi * xi)))
    bound += np.sum(0.5 * docLens[:,np.newaxis] * (s[:,np.newaxis] + xi))
#    bound -= np.sum(docLens[:,np.newaxis] * safe_log_one_plus_exp_of(xi))
    bound -= scaledSumOfLnOnePlusExp(docLens, xi)
    
    bound -= np.dot(s, docLens)
    
    means = np.log(expMeans, out=expMeans)
    
    return bound
        
        
        
        

# ==============================================================
# PUBLIC HELPERS
# ==============================================================

def printStderr(msg):
    sys.stdout.flush()
    sys.stderr.write(msg + '\n')
    sys.stderr.flush()

def negJakkola(vec):
    '''
    The negated version of the Jakkola expression which was used in Bouchard's NIPS
    2007 softmax bound
    
    CTM Source reads: y = .5./x.*(1./(1+exp(-x)) -.5);
    '''
    
    # COPY AND PASTE BETWEEN THIS AND negJakkolaOfDerivedXi()
    return 0.5/vec * (1./(1 + np.exp(-vec)) - 0.5)

def negJakkolaOfDerivedXi(means, varcs, s, d = None):
    '''
    The negated version of the Jakkola expression which was used in Bouchard's NIPS '07
    softmax bound calculated using an estimate of xi derived from lambda, nu, and s
    
    means   - the DxK matrix of means of the topic distribution for each document.
    varcs   - the DxK the vector of variances of the topic distribution
    s       - The Dx1 vector of offsets.
    d       - the document index (for lambda and nu). If not specified we construct
              the full matrix of A(xi_dk)
    '''
    err = errorMsg(means)
    if err is not None:
        print ("Means " + err)
        print ("")
        
    err = errorMsg(s)
    if err is not None:
        print ("s " + err)
        print ("")
        
    err = errorMsg(varcs)
    if err is not None:
        print ("lxi " + err)
        print ("")
    
    # COPY AND PASTE BETWEEN THIS AND negJakkola()
    if d is not None:
        vec = (np.sqrt (means[d,:]**2 - 2 * means[d,:] * s[d] + s[d]**2 + varcs[d,:]**2))
        return 0.5/vec * (1./(1 + np.exp(-vec)) - 0.5)
    else:
        mat = _deriveXi(means, varcs, s)
        return 0.5/mat * (1./(1 + np.exp(-mat)) - 0.5)
    

def jakkolaOfDerivedXi(means, varcs, s, d = None):
    '''
    The standard version of the Jakkola expression which was used in Bouchard's NIPS '07
    softmax bound calculated using an estimate of xi derived from lambda, nu, and s
    
    means - the DxK matrix of means of the topic distribution for each document
    varcs - the DxK the vector of variances of the topic distribution
    s    - The Dx1 vector of offsets.
    d    - the document index (for lambda and nu). If not specified we construct
           the full matrix of A(xi_dk)
    '''
    err = errorMsg(means)
    if err is not None:
        print ("Means " + err)
        print ("")
        
    err = errorMsg(s)
    if err is not None:
        print ("s " + err)
        print ("")
        
    err = errorMsg(varcs)
    if err is not None:
        print ("lxi " + err)
        print ("")
    
    # COPY AND PASTE BETWEEN THIS AND negJakkola()
    if d is not None:
        vec = (np.sqrt (means[d,:]**2 -2 *means[d,:] * s[d] + s[d]**2 + varcs[d,:]**2))
        return 0.5/vec * (1./(1 + np.exp(-vec)) - 0.5)
    else:
        mat = _deriveXi(means, varcs, s)
        return 0.5/mat * (0.5 - 1./(1 + np.exp(-mat)))



# ==============================================================
# PRIVATE HELPERS
# ==============================================================

def _deriveXi (means, varcs, s):
    '''
    Derives a value for xi. This is not normally needed directly, as we
    normally just work with the negJakkola() function of it
    '''
    return np.sqrt(means**2 - 2 * means * s[:,np.newaxis] + (s**2)[:,np.newaxis] + varcs**2)   

def _debug_with_bound (itr, var_value, var_name, W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n):
    if np.isnan(var_value).any():
        printStderr ("WARNING: " + var_name + " contains NaNs")
    if np.isinf(var_value).any():
        printStderr ("WARNING: " + var_name + " contains INFs")
        
    print ("Iter %3d Update %s Bound %f" % (itr, var_name, var_bound(W, ModelState(K, topicMean, sigT, vocab, dtype, MODEL_NAME), QueryState(means, varcs, lxi, s, n)))) 


def _debug_with_nothing (itr, var_value, var_name, W, K, topicMean, sigT, vocab, dtype, means, varcs, lxi, s, n):   
    pass

def errorMsg(mat):
    '''
    If somethings wrong, return an error message, otherwise return None
    '''
    if np.isnan(mat).any() and np.isinf(mat).any():
        return "Matrix has NaNs and INFs"
    elif np.isnan(mat).any() :
        return "Matrix has NaNs"
    elif np.isinf(mat).any():
        return "Matrix has INFs"
    else:
        return None
    
