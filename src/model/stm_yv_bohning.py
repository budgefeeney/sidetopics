# -*- coding: utf-8 -*-
'''
Topic model where topics are predicts from a documents features.
Prediction is by means of a matrix 

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
from numba import autojit

from util.overflow_safe import safe_log, safe_log_one_plus_exp_of, safe_log_det
from util.array_utils import normalizerows_ip
from util.sigmoid_utils import rowwise_softmax, selfSoftDot, scaledSelfSoftDot
from util.sparse_elementwise import sparseScalarProductOf, \
    sparseScalarProductOfDot, sparseScalarQuotientOfDot, \
    entropyOfDot, sparseScalarProductOfSafeLnDot
from model.stm_yv import lnDetOfDiagMat, safeDet, static_var
from model.ctm_bohning import printStderr, LN_OF_2_PI, \
    LN_OF_2_PI_E, newModelAtRandom as newCtmModelAtRandom
from model.ctm import verifyProper
    
# ==============================================================
# CONSTANTS
# ==============================================================

DTYPE=np.float32 # A default, generally we should specify this in the model setup

DEBUG=True

# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency plot plotFile plotIncremental fastButInaccurate')                            

QueryState = namedtuple ( \
    'QueryState', \
    'means varcs docLens'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'F P K A R_A fv Y R_Y lfv V sigT vocab Ab dtype'
)

# ==============================================================
# PUBLIC API
# ==============================================================

def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState(model.K, model.topicMean.copy(), model.sigT.copy(), model.vocab.copy(), model.dtype)

def newModelAtRandom(X, W, P, K, featVar, latFeatVar, dtype=DTYPE):
    '''
    Creates a new CtmModelState for the given training set and
    the given number of topics. Everything is instantiated purely
    at random. This contains all parameters independent of of
    the dataset (e.g. learnt priors)
    
    Param:
    X - The DxF document-feature matrix of F features associated
        with the D documents
    W - the DxT document-term matrix of T terms in D documents
        which will be used for training.
    P - The size of the latent feature-space P << F
    K - the number of topics
    featVar - the prior variance of the feature-space: this is a
              scalar used to scale an identity matrix
    featVar - the prior variance of the latent feature-space: this
               is a scalar used to scale an identity matrix
    
    Return:
    A ModelState object
    '''
    assert K > 1, "There must be at least two topics"
    
    base = newCtmModelAtRandom(W, K, dtype)
    _,F = X.shape
    Y = rd.random((K,P)).astype(dtype)
    R_Y = latFeatVar * np.eye(P,P, dtype=dtype)
    
    V = rd.random((P,F)).astype(dtype)
    A = Y.dot(V)
    R_A = featVar * np.eye(F,F, dtype=dtype)
    
    return ModelState(F, P, K, A, R_A, featVar, Y, R_Y, latFeatVar, V, base.sigT, base.vocab, base.A, dtype)


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
    
    return QueryState(means, varcs, docLens)


def newTrainPlan(iterations = 100, epsilon=0.01, logFrequency=10, plot=False, plotFile=None, plotIncremental=False, fastButInaccurate=False):
    '''
    Create a training plan determining how many iterations we
    process, how often we plot the results, how often we log
    the variational bound, etc.
    '''
    return TrainPlan(iterations, epsilon, logFrequency, plot, plotFile, plotIncremental, fastButInaccurate)

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
    updated in place, so make a defensive copy if you want itr)
    A new query object with the update query parameters
    ''' 
    D,_ = W.shape
    
    # Unpack the the structs, for ease of access and efficiency
    iterations, epsilon, logFrequency, plot, plotFile, plotIncremental, fastButInaccurate = trainPlan.iterations, trainPlan.epsilon, trainPlan.logFrequency, trainPlan.plot, trainPlan.plotFile, trainPlan.plotIncremental, trainPlan.fastButInaccurate
    means, varcs, n = queryState.means, queryState.varcs, queryState.docLens
    F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, Ab, dtype = modelState.F, modelState.P, modelState.K, modelState.A, modelState.R_A, modelState.fv, modelState.Y, modelState.R_Y, modelState.lfv, modelState.V, modelState.sigT, modelState.vocab, modelState.Ab, modelState.dtype
    
    # Book-keeping for logs
    boundIters  = np.zeros(shape=(iterations // logFrequency,))
    boundValues = np.zeros(shape=(iterations // logFrequency,))
    bvIdx = 0
    debugFn = _debug_with_bound if DEBUG else _debug_with_nothing
    
    # Initialize some working variables
    isigT = la.inv(sigT)
    R = W.copy()
    sigT_regularizer = 0.001
    
    aI_P = 1./lfv  * ssp.eye(P, dtype=dtype)
    tI_F = 1./fv * ssp.eye(F, dtype=dtype)
    
    print("Creating posterior covariance of A, this will take some time...")
    XTX = X.T.dot(X)
    R_A = XTX
    R_A = R_A.todense()      # dense inverse typically as fast or faster than sparse inverse
    R_A.flat[::F+1] += 1./fv # and the result is usually dense in any case
    R_A = la.inv(R_A)
    print("Covariance matrix calculated, launching inference")
    
    R_Y_base = R_Y.copy()
    
    priorSigt_diag = np.ndarray(shape=(K,), dtype=dtype)
    priorSigt_diag.fill (0.001)
    
    # Iterate over parameters
    for itr in range(iterations):
        
        # We start with the M-Step, so the parameters are consistent with our
        # initialisation of the RVs when we do the E-Step
        
        # Update the covariance of the prior
        diff_a_yv = (A-Y.dot(V))
        diff_m_xa = (means-X.dot(A.T))
        
        sigT  = 1./lfv * (Y.dot(Y.T))
        sigT += 1./fv * diff_a_yv.dot(diff_a_yv.T)
        sigT += diff_m_xa.T.dot(diff_m_xa)
        sigT.flat[::K+1] += varcs.sum(axis=0)
        sigT /= (P+F+D)
        sigT.flat[::K+1] += sigT_regularizer
        
        # ...and then diagonalize itr
        sigT = np.diag(sigT.flat[::K+1])
        # ...and finally invert itr.
        isigT = np.diag(np.reciprocal(sigT.flat[::K+1]))
        debugFn (itr, sigT, "sigT", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, dtype, means, varcs, Ab, n)
        
        # Building Blocks - termporarily replaces means with exp(means)
        expMeans = np.exp(means, out=means)
        R = sparseScalarQuotientOfDot(W, expMeans, vocab, out=R)
        S = expMeans * R.dot(vocab.T)
        
        # Update the vocabulary
        vocab *= (R.T.dot(expMeans)).T # Awkward order to maintain sparsity (R is sparse, expMeans is dense)
        vocab = normalizerows_ip(vocab)
        vocab += 1E-30 if dtype==np.float32 else 1E-300 # Just to ensure that we don't get zero probabilities in the absence of a proper prior
        
        # Reset the means to their original form, and log effect of vocab update
#        R = sparseScalarQuotientOfDot(W, expMeans, vocab, out=R)
#        S = expMeans * R.dot(vocab.T)
        
        means = np.log(expMeans, out=expMeans)
        debugFn (itr, vocab, "vocab", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, dtype, means, varcs, Ab, n)
        
        #
        # And now this is the E-Step
        # 
        
        # Update the distribution on the latent space
        R_Y_base = aI_P + 1/fv * V.dot(V.T)
        R_Y = la.inv(R_Y_base)
        debugFn (itr, R_Y, "R_Y", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, dtype, means, varcs, Ab, n)
        
        Y = 1./fv * A.dot(V.T).dot(R_Y)
        debugFn (itr, Y, "Y", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, dtype, means, varcs, Ab, n)
        
        # Update the mapping from the features to topics
        A = (1./fv * (Y).dot(V) + (X.T.dot(means)).T).dot(R_A)
        debugFn (itr, A, "A", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, dtype, means, varcs, Ab, n)
        
        # Update the Variances
        varcs = 1./((n * (K-1.)/K)[:,np.newaxis] + isigT.flat[::K+1])
        debugFn (itr, varcs, "varcs", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, dtype, means, varcs, Ab, n)
        
        # Update the Means
        rhs = S.copy()
        rhs += n[:,np.newaxis] * means.dot(Ab)
        rhs -= n[:,np.newaxis] * rowwise_softmax(means, out=means)
        rhs += X.dot(A.T).dot(isigT)
        lhs = np.reciprocal(np.diag(isigT)[np.newaxis,:] + n[:,np.newaxis] * Ab)  # inverse of D diagonal matrices...
        
        means = lhs * rhs # as LHS is a diagonal matrix for all d, itr's equivalent to a Hadamard product
        debugFn (itr, means, "means", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, dtype, means, varcs, Ab, n)
        
        if logFrequency > 0 and itr % logFrequency == 0:
            modelState = ModelState(F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, Ab, dtype)
            queryState = QueryState(means, varcs, n)
            
            boundValues[bvIdx] = var_bound(W, X, modelState, queryState, XTX)
            boundIters[bvIdx]  = itr
            print ("\nIteration %d: bound %f" % (itr, boundValues[bvIdx]))
            if bvIdx > 0 and  boundValues[bvIdx - 1] > boundValues[bvIdx]:
                printStderr ("ERROR: bound degradation: %f > %f" % (boundValues[bvIdx - 1], boundValues[bvIdx]))
            print ("Means: min=%f, avg=%f, max=%f\n\n" % (means.min(), means.mean(), means.max()))
            bvIdx += 1
            
    if plot:
        plt.plot(boundIters[5:], boundValues[5:])
        plt.xlabel("Iterations")
        plt.ylabel("Variational Bound")
        plt.show()
        
    
    return \
        ModelState(F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, Ab, dtype), \
        QueryState(means, varcs, n)
    

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
    
def var_bound(W, X, modelState, queryState, XTX=None):
    '''
    Determines the variational bounds. Values are mutated in place, but are
    reset afterwards to their initial values. So it's safe to call in a serial
    manner.
    '''
    
    # Unpack the the structs, for ease of access and efficiency
    D,_ = W.shape
    means, varcs, docLens = queryState.means, queryState.varcs, queryState.docLens
    F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, Ab, dtype = modelState.F, modelState.P, modelState.K, modelState.A, modelState.R_A, modelState.fv, modelState.Y, modelState.R_Y, modelState.lfv, modelState.V, modelState.sigT, modelState.vocab, modelState.Ab, modelState.dtype
    
    # Calculate some implicit  variables
    isigT = la.inv(sigT)
    lnDetSigT = lnDetOfDiagMat(sigT)
    verifyProper(lnDetSigT, "lnDetSigT")
    
    if XTX is None:
        XTX = X.T.dot(X)
    
    bound = 0
    
    # Distribution over latent space
    bound -= (P*K)/2. * LN_OF_2_PI
    bound -= P * lnDetSigT
    bound -= K * P * log(lfv)
    bound -= 0.5 * np.sum(1./lfv * isigT.dot(Y) * Y)
    bound -= 0.5 * K * np.trace(R_Y)
    
    # And its entropy
    detR_Y = safeDet(R_Y, "R_Y")
    bound += 0.5 * LN_OF_2_PI_E + P/2. * lnDetSigT + K/2. * log(detR_Y)
    
    # Distribution over mapping from features to topics
    diff   = (A - Y.dot(V))
    bound -= (F*K)/2. * LN_OF_2_PI
    bound -= F * lnDetSigT
    bound -= K * P * log(fv)
    bound -= 0.5 * np.sum (1./lfv * isigT.dot(diff) * diff)
    bound -= 0.5 * K * np.trace(R_A)
    
    # And its entropy
    detR_A = safeDet(R_A, "R_A")
    bound += 0.5 * LN_OF_2_PI_E + F/2. * lnDetSigT + K/2. * log(detR_A)
    
    # Distribution over document topics
    bound -= (D*K)/2. * LN_OF_2_PI
    bound -= D/2. * lnDetSigT
    diff   = means - X.dot(A.T)
    bound -= 0.5 * np.sum (diff.dot(isigT) * diff)
    bound -= 0.5 * np.sum(varcs * np.diag(isigT)[np.newaxis,:]) # = -0.5 * sum_d tr(V_d \Sigma^{-1}) when V_d is diagonal only.
    bound -= 0.5 * K * np.trace(XTX.dot(R_A))
       
    # And its entropy
    bound += 0.5 * D * K * LN_OF_2_PI_E + 0.5 * np.sum(np.log(varcs)) 
        
    # Distribution over word-topic assignments
    expMeans = np.exp(means, out=means)
    R = sparseScalarQuotientOfDot(W, expMeans, vocab)  # D x V   [W / TB] is the quotient of the original over the reconstructed doc-term matrix
    V = expMeans * (R.dot(vocab.T)) # D x K
    means = np.log(expMeans, out=expMeans)
    
    bound += np.sum(means * V)
    bound += np.sum(2 * ssp.diags(docLens,0) * means.dot(A) * means)
    bound -= 2. * scaledSelfSoftDot(means, docLens)
    bound -= 0.5 * np.sum(docLens[:,np.newaxis] * V * (np.diag(A))[np.newaxis,:])
    bound += np.sum(docLens * np.log(np.sum(np.exp(means), axis=1)))
    
    # And its entropy, and the distribution over words
    bound -= np.sum(means * V) 
    bound += np.sum(sparseScalarProductOfSafeLnDot(W, expMeans, vocab).data)
    
    return bound
        

# ==============================================================
# PUBLIC HELPERS
# ==============================================================

@static_var("old_bound", 0)
def _debug_with_bound (itr, var_value, var_name, W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, dtype, means, varcs, Ab, n):
    if np.isnan(var_value).any():
        printStderr ("WARNING: " + var_name + " contains NaNs")
    if np.isinf(var_value).any():
        printStderr ("WARNING: " + var_name + " contains INFs")
    if var_value.dtype != dtype:
        printStderr ("WARNING: dtype(" + var_name + ") = " + str(var_value.dtype))
    
    old_bound = _debug_with_bound.old_bound
    bound     = var_bound(W, X, ModelState(F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, Ab, dtype), QueryState(means, varcs, n), XTX)
    diff = "" if old_bound == 0 else str(bound - old_bound)
    _debug_with_bound.old_bound = bound
    
    print ("Iter %3d Update %s Bound %f (%s)" % (itr, var_name, bound, diff)) 

def _debug_with_nothing (itr, var_value, var_name, W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, dtype, means, varcs, Ab, n):
    pass

