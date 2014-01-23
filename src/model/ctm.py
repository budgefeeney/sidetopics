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
import scipy.sparse.linalg as sla
import numpy.random as rd
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

from util.overflow_safe import safe_log, safe_log_one_plus_exp_of, safe_log_det
from util.array_utils import normalizerows_ip, rowwise_softmax
from util.sparse_elementwise import sparseScalarProductOf, \
    sparseScalarProductOfDot, sparseScalarQuotientOfDot, \
    entropyOfDot, sparseScalarProductOfSafeLnDot
    
# ==============================================================
# CONSTANTS
# ==============================================================

MAX_X_TICKS_PER_PLOT = 50
DTYPE = np.float32

LN_OF_2_PI   = log(2 * pi)
LN_OF_2_PI_E = log(2 * pi * e)

DEBUG=True

# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency plot plotFile plotIncremental fastButInaccurate')                            

QueryState = namedtuple ( \
    'QueryState', \
    'expMeans varcs lxi s docLens'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K topicMean sigT vocab dtype'
)

# ==============================================================
# PUBLIC API
# ==============================================================

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
    
    sigT = rd.random((K,K)).astype(dtype)
    sigT = sigT.T.dot(sigT)
    
    return ModelState(K, topicMean, sigT, vocab, dtype)

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
    K, topicMean, sigT, vocab, dtype =  modelState.K, modelState.topicMean, modelState.sigT, modelState.vocab, modelState.dtype
    
    D,T = W.shape
    assert T == vocab.shape[1], "The number of terms in the document-term matrix (" + str(T) + ") differs from that in the model-states vocabulary parameter " + str(vocab.shape[1])
    docLens = np.squeeze(np.asarray(W.sum(axis=1)))
    
    means = normalizerows_ip(rd.random((D,K)).astype(dtype))
    varcs = normalizerows_ip(rd.random((D,K)).astype(dtype))
    
    s = np.ndarray(shape=(D,), dtype=dtype)
    s.fill(0)
    
    lxi = negJakkolaOfDerivedXi(means, varcs, s)
    
    expMeans = np.exp(means, out=means)
    
    return QueryState(expMeans, varcs, lxi, s, docLens)


def newTrainPlan(iterations = 100, epsilon=0.01, logFrequency=10, plot=False, plotFile=None, plotIncremental=False, fastButInaccurate=False):
    '''
    Create a training plan determining how many iterations we
    process, how often we plot the results, how often we log
    the variational bound, etc.
    '''
    return TrainPlan(iterations, epsilon, logFrequency, plot, plotFile, plotIncremental, fastButInaccurate)


def train (W, modelState, queryState, trainPlan):
    '''
    Infers the topic distributions in general, and specifically for
    each individual datapoint.
    
    Params:
    W - the DxT document-term matrix
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
    iterations, epsilon, logFrequency, plot, plotFile, plotIncremental, fastButInaccurate = trainPlan.iterations, trainPlan.epsilon, trainPlan.logFrequency, trainPlan.plot, trainPlan.plotFile, trainPlan.plotIncremental, trainPlan.fastButInaccurate
    expMeans, varcs, lxi, s, n = queryState.expMeans, queryState.varcs, queryState.lxi, queryState.s, queryState.docLens
    K, topicMean, sigT, vocab, dtype = modelState.K, modelState.topicMean, modelState.sigT, modelState.vocab, modelState.dtype
    
    # Book-keeping for logs
    boundIters  = np.zeros(iterations / logFrequency)
    boundValues = np.zeros(iterations / logFrequency)
    bvIdx = 0
    
    # Initialize some working variables
    isigT = la.inv(sigT)
    R = W.copy()
    
    # Iterate over parameters
    for iter in range(iterations):
        
        # Building Blocks
        R = sparseScalarQuotientOfDot(W, expMeans, vocab, out=R)
        V = expMeans * R.dot(vocab.T)
        
        # Update the Mean
        means  = np.log(expMeans, out=expMeans)
        vMat   = (2  * s[:,np.newaxis] * lxi - 0.5) * n[:,np.newaxis] + V
        rhsMat = vMat + isigT.dot(topicMean)
        nD     = n[:,np.newaxis] * 2 * lxi
        for d in range(D):
            isigT.flat[::K+1] += n[d] * 2 * lxi[d,:]
            means[d,:] = la.inv(isigT).dot(rhsMat[d,:])
            isigT.flat[::K+1] -= n[d] * 2 * lxi[d,:]
            
        
        expMeans = np.exp(means, out=means)
        
        
        if logFrequency > 0 and iter % logFrequency == 0:
            modelState = ModelState(K, topicMean, sigT, vocab, dtype)
            queryState = QueryState(expMeans, varcs, lxi, s, n)
            
            boundValues[bvIdx] = var_bound(W, modelState, queryState)
            boundIters[bvIdx]  = iter
            print ("Iteration %d: bound %f\n" % (iter, boundValues[bvIdx]))
            bvIdx += 1
            
    if plot:
        plt.plot(boundIters, boundValues)
        plt.xlabel("Iterations")
        plt.ylabel("Variational Bound")
        plt.show()
        
    
    return \
        ModelState(K, topicMean, sigT, vocab, dtype), \
        QueryState(expMeans, varcs, lxi, s, n)
    
    
    
def var_bound(W, modelState, queryState):
    '''
    Determines the variational bounds. Values are mutated in place, but are
    reset afterwards to their initial values. So it's safe to call in a serial
    manner.
    '''
    
    # Unpack the the structs, for ease of access and efficiency
    D,_ = W.shape
    expMeans, varcs, lxi, s, docLens    = queryState.expMeans, queryState.varcs, queryState.lxi, queryState.s, queryState.docLens
    K, topicMean, sigT, vocab = modelState.K, modelState.topicMean, modelState.sigT, modelState.vocab
    
    # Calculate some implicit 
    means = np.log(expMeans, out=expMeans)
    xi    = np.sqrt (means * means + 2 * means * s[:, np.newaxis] + (s * s)[:,np.newaxis] + varcs*varcs)
    
    bound = 0
    
    # Distribution over document topics
    bound -= (D*K)/2. * np.log(2*pi)
    bound -= D/2. * la.det(sigT)
    diff   = topicMean[np.newaxis,:] - means
    bound -= 0.5 * np.sum (diff.dot(la.inv(sigT)) * diff)
    
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
    
    expMeans = np.exp(means, out=means)
    bound -= -np.sum(sparseScalarProductOfSafeLnDot(W, expMeans, vocab).data)
    
    bound -= np.sum(docLens[:,np.newaxis] * lxi * ((s*s)[:,np.newaxis] - (xi * xi)))
    bound += np.sum(0.5 * docLens[:,np.newaxis] * (s[:,np.newaxis] + xi))
    bound -= np.sum(docLens[:,np.newaxis] * safe_log_one_plus_exp_of(xi))
    
    bound -= np.dot(s, docLens)
    
    return bound
        
        
        
        

# ==============================================================
# PUBLIC HELPERS
# ==============================================================

def negJakkola(vec):
    '''
    The negated version of the Jakkola expression which was used in Bouchard's NIPS
    2007 softmax bound
    
    CTM Source reads: y = .5./x.*(1./(1+exp(-x)) -.5);
    '''
    
    # COPY AND PASTE BETWEEN THIS AND negJakkolaOfDerivedXi()
    return 0.5/vec * (1./(1 + np.exp(-vec)) - 0.5)

def negJakkolaOfDerivedXi(lmda, nu, s, d = None):
    '''
    The negated version of the Jakkola expression which was used in Bouchard's NIPS '07
    softmax bound calculated using an estimate of xi derived from lambda, nu, and s
    
    lmda    - the DxK matrix of means of the topic distribution for each document.
              Note that this is different to all other implementations of neg-Jaakkola which assume
              the topic distribution is provided, and not e to the power of it.
    nu      - the DxK the vector of variances of the topic distribution
    s       - The Dx1 vector of offsets.
    d       - the document index (for lambda and nu). If not specified we construct
              the full matrix of A(xi_dk)
    '''
    
    # COPY AND PASTE BETWEEN THIS AND negJakkola()
    if d is not None:
        vec = (np.sqrt (lmda[d,:]**2 - 2 * lmda[d,:] * s[d] + s[d]**2 + nu[d,:]**2))
        return 0.5/vec * (1./(1 + np.exp(-vec)) - 0.5)
    else:
        mat = _deriveXi(lmda, nu, s)
        return 0.5/mat * (1./(1 + np.exp(-mat)) - 0.5)
    

def jakkolaOfDerivedXi(lmda, nu, s, d = None):
    '''
    The standard version of the Jakkola expression which was used in Bouchard's NIPS '07
    softmax bound calculated using an estimate of xi derived from lambda, nu, and s
    
    lmda - the DxK matrix of means of the topic distribution for each document
    nu   - the DxK the vector of variances of the topic distribution
    s    - The Dx1 vector of offsets.
    d    - the document index (for lambda and nu). If not specified we construct
           the full matrix of A(xi_dk)
    '''
    
    # COPY AND PASTE BETWEEN THIS AND negJakkola()
    if d is not None:
        vec = (np.sqrt (lmda[d,:]**2 -2 *lmda[d,:] * s[d] + s[d]**2 + nu[d,:]**2))
        return 0.5/vec * (1./(1 + np.exp(-vec)) - 0.5)
    else:
        mat = _deriveXi(lmda, nu, s)
        return 0.5/mat * (0.5 - 1./(1 + np.exp(-mat)))



# ==============================================================
# PRIVATE HELPERS
# ==============================================================

def _deriveXi (lmda, nu, s):
    '''
    Derives a value for xi. This is not normally needed directly, as we
    normally just work with the negJakkola() function of it
    '''
    return np.sqrt(lmda**2 - 2 * lmda * s[:,np.newaxis] + (s**2)[:,np.newaxis] + nu**2)   


