# -*- coding: utf-8 -*-
'''
Topic model where topics are predicts from a documents features.
Prediction is by means of a matrix 

Created on 17 Jan 2014

@author: bryanfeeney
'''

from math import log, isnan

import time
from collections import namedtuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as ssp
import numpy.random as rd
#import numba as nb

from sidetopics.util.array_utils import normalizerows_ip
from sidetopics.util.sigmoid_utils import rowwise_softmax, scaledSelfSoftDot
from sidetopics.util.sparse_elementwise import sparseScalarQuotientOfDot, \
    sparseScalarProductOfSafeLnDot
from sidetopics.model.ctm_bohning import LN_OF_2_PI, LN_OF_2_PI_E,\
    newModelAtRandom as newCtmModelAtRandom
from sidetopics.util.misc import printStderr, static_var, converged, clamp
from sidetopics.util.overflow_safe import safeDet, lnDetOfDiagMat
from sidetopics.model.ctm import verifyProper
from sidetopics.model.common import DataSet
from sidetopics.model.evals import perplexity_from_like
    
# ==============================================================
# CONSTANTS
# ==============================================================

DTYPE=np.float32 # A default, generally we should specify this in the model setup

DEBUG=False

VocabPrior=0.1

MODEL_NAME="stm-yv/bohning/online/fake"

STABLE_SORT_ALG="mergesort"

Tau=5
Kappa=0.5


# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')                            

QueryState = namedtuple ( \
    'QueryState', \
    'means expMeans varcs docLens'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'F P K A R_A fv Y R_Y lfv V sigT vocab vocabPrior Ab dtype name'
)

# ==============================================================
# PUBLIC API
# ==============================================================

def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    def copy(matrix):
        return None if matrix is None else matrix.copy()
    
    return ModelState(model.F, model.P, model.K, \
        copy(model.A), copy(model.R_A), model.fv, \
        copy(model.Y), copy(model.R_Y), model.lfv, \
        copy(model.V), \
        copy(model.sigT), \
        copy(model.vocab), model.vocabPrior, \
        copy(model.Ab), \
        model.dtype, model.name)

def newModelAtRandom(data, P, K, featVar, latFeatVar, vocabPrior=VocabPrior, dtype=DTYPE):
    '''
    Creates a new CtmModelState for the given training set and
    the given number of topics. Everything is instantiated purely
    at random. This contains all parameters independent of of
    the dataset (e.g. learnt priors)
    
    Param:
    data - the dataset of words, features and links of which only words and
           features are used in this model
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
    
    base = newCtmModelAtRandom(data, K, vocabPrior, dtype)
    _,F = data.feats.shape
    Y = rd.random((K,P)).astype(dtype) * 50
    R_Y = latFeatVar * np.eye(P,P, dtype=dtype)
    
    V = rd.random((P,F)).astype(dtype) * 50
    A = Y.dot(V)
    R_A = featVar * np.eye(F,F, dtype=dtype)
    
    return ModelState(F, P, K, A, R_A, featVar, Y, R_Y, latFeatVar, V, base.sigT, base.vocab, base.vocabPrior, base.A, dtype, MODEL_NAME)


def newQueryState(data, model):
    '''
    Creates a new CTM Query state object. This contains all
    parameters and random variables tied to individual
    datapoints.
    
    Param:
    data - the dataset of words, features and links of which only words and
           features are used in this model
    modelState - the model state object
    
    Return:
    A CtmQueryState object
    '''
    K, vocab, dtype =  model.K, model.vocab, model.dtype
    
    D,T = data.words.shape
    assert T == vocab.shape[1], "The number of terms in the document-term matrix (" + str(T) + ") differs from that in the model-states vocabulary parameter " + str(vocab.shape[1])
    docLens = np.squeeze(np.asarray(data.words.sum(axis=1)))

    base     = rd.random((D,K*2)).astype(dtype)
    means    = base[:,:K]
    expMeans = base[:,K:]

    np.log(means, out=means) # Try to start with a system where taking the exp makes sense
    varcs = np.ones((D,K), dtype=dtype)

    return QueryState(means, expMeans, varcs, docLens)


def newTrainPlan(iterations = 100, epsilon=2, logFrequency=10, fastButInaccurate=False, debug=DEBUG):
    '''
    Create a training plan determining how many iterations we
    process, how often we plot the results, how often we log
    the variational bound, etc.
    '''
    return TrainPlan(iterations, epsilon, logFrequency, fastButInaccurate, debug)

def is_undirected_link_predictor():
    '''
    Is this model only for predicting link structure, and only in the case where
    the links are undirected.
    '''
    return False

BatchSize=100000
#@nb.autojit
def train (data, modelState, queryState, trainPlan):
    '''
    Infers the topic distributions in general, and specifically for
    each individual datapoint.
    
    Params:
    data - the dataset of words, features and links of which only words and
           features are used in this model
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
    W, X = data.words, data.feats
    D, _ = W.shape
    
    # Unpack the the structs, for ease of access and efficiency
    iterations, epsilon, logFrequency, fastButInaccurate, debug = trainPlan.iterations, trainPlan.epsilon, trainPlan.logFrequency, trainPlan.fastButInaccurate, trainPlan.debug
    means, expMeans, varcs, docLens = queryState.means, queryState.expMeans, queryState.varcs, queryState.docLens
    F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, Ab, dtype = modelState.F, modelState.P, modelState.K, modelState.A, modelState.R_A, modelState.fv, modelState.Y, modelState.R_Y, modelState.lfv, modelState.V, modelState.sigT, modelState.vocab, modelState.vocabPrior, modelState.Ab, modelState.dtype
    
    # Book-keeping for logs
    boundIters  = np.zeros(shape=(iterations // logFrequency,))
    boundValues = np.zeros(shape=(iterations // logFrequency,))
    boundLikes = np.zeros(shape=(iterations // logFrequency,))
    bvIdx = 0
    debugFn = _debug_with_bound if debug else _debug_with_nothing
    _debug_with_bound.old_bound = 0
    
    # For efficient inference, we need a separate covariance for every unique
    # document length. For products to execute quickly, the doc-term matrix
    # therefore needs to be ordered in ascending terms of document length
    originalDocLens = docLens
    sortIdx = np.argsort(docLens, kind=STABLE_SORT_ALG) # sort needs to be stable in order to be reversible
    W = W[sortIdx,:] # deep sorted copy
    X = X[sortIdx,:]
    means, varcs = means[sortIdx,:], varcs[sortIdx,:]

    docLens = originalDocLens[sortIdx]
    
    lens, inds = np.unique(docLens, return_index=True)
    inds = np.append(inds, [W.shape[0]])
    
    # Initialize some working variables
    R = W.copy()
    
    aI_P = 1./lfv  * ssp.eye(P, dtype=dtype)
    
    print("Creating posterior covariance of A, this will take some time...")
    XTX = X.T.dot(X)
    R_A = XTX
    R_A = R_A.todense()      # dense inverse typically as fast or faster than sparse inverse
    R_A.flat[::F+1] += 1./fv # and the result is usually dense in any case
    R_A = la.inv(R_A)
    print("Covariance matrix calculated, launching inference")


    diff_m_xa = (means-X.dot(A.T))
    means_cov_with_x_a = diff_m_xa.T.dot(diff_m_xa)

    expMeans = np.zeros((BatchSize, K), dtype=dtype)
    R = np.zeros((BatchSize, K), dtype=dtype)
    S = np.zeros((BatchSize, K), dtype=dtype)
    vocabScale = np.ones(vocab.shape, dtype=dtype)
    
    # Iterate over parameters
    batchIter = 0
    for itr in range(iterations):
        
        # We start with the M-Step, so the parameters are consistent with our
        # initialisation of the RVs when we do the E-Step

        # Update the covariance of the prior
        diff_a_yv = (A-Y.dot(V))
        sigT  = 1./lfv * (Y.dot(Y.T))
        sigT += 1./fv * diff_a_yv.dot(diff_a_yv.T)
        sigT += means_cov_with_x_a
        sigT.flat[::K+1] += varcs.sum(axis=0)

        # As small numbers lead to instable inverse estimates, we use the
        # fact that for a scalar a, (a .* X)^-1 = 1/a * X^-1 and use these
        # scales whenever we use the inverse of the unscaled covariance
        sigScale  = 1. / (P+D+F)
        isigScale = 1. / sigScale

        isigT = la.inv(sigT)
        debugFn (itr, sigT, "sigT", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, docLens)
        
        # Update the vocabulary
        vocab *= vocabScale
        vocab += vocabPrior
        vocab = normalizerows_ip(vocab)
        debugFn (itr, vocab, "vocab", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, docLens)
        
        # Finally update the parameter V
        V = la.inv(sigScale * R_Y + Y.T.dot(isigT).dot(Y)).dot(Y.T.dot(isigT).dot(A))
        debugFn (itr, V, "V", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, docLens)
        
        
        #
        # And now this is the E-Step
        # 
        
        # Update the distribution on the latent space
        R_Y_base = aI_P + 1/fv * V.dot(V.T)
        R_Y = la.inv(R_Y_base)
        debugFn (itr, R_Y, "R_Y", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, docLens)
        
        Y = 1./fv * A.dot(V.T).dot(R_Y)
        debugFn (itr, Y, "Y", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, docLens)
        
        # Update the mapping from the features to topics
        A = (1./fv * Y.dot(V) + (X.T.dot(means)).T).dot(R_A)
        debugFn (itr, A, "A", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, docLens)
        
        # Update the Variances
        varcs = 1./((docLens * (K-1.)/K)[:,np.newaxis] + isigScale * isigT.flat[::K+1])
        debugFn (itr, varcs, "varcs", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, docLens)

        # Faster version?
        vocabScale[:,:] = 0
        means_cov_with_x_a[:,:] = 0
        for lenIdx in range(len(lens)):
            nd         = lens[lenIdx]
            start, end = inds[lenIdx], inds[lenIdx + 1]
            lhs        = la.inv(isigT + sigScale * nd * Ab) * sigScale

            for d in range(start, end, BatchSize):
                end_d = min(d + BatchSize, end)
                span  = end_d - d

                expMeans[:span,:] = np.exp(means[d:end_d,:] - means[d:end_d,:].max(axis=1)[:span,np.newaxis], out=expMeans[:span,:])
                R = sparseScalarQuotientOfDot(W[d:end_d,:], expMeans[d:end_d,:], vocab)
                S[:span,:] = expMeans[:span, :] * R.dot(vocab.T)

                # Convert expMeans to a softmax(means)
                expMeans[:span,:] /= expMeans[:span,:].sum(axis=1)[:span,np.newaxis]

                mu   = X[d:end_d,:].dot(A.T)
                rhs  = mu.dot(isigT) * isigScale
                rhs += S[:span,:]
                rhs += docLens[d:end_d,np.newaxis] * means[d:end_d,:].dot(Ab)
                rhs -= docLens[d:end_d,np.newaxis] * expMeans[:span,:] # here expMeans is actually softmax(means)

                means[d:end_d,:] = rhs.dot(lhs) # huh?! Left and right refer to eqn for a single mean: once we're talking a DxK matrix it gets swapped

                expMeans[:span,:] = np.exp(means[d:end_d,:] - means[d:end_d,:].max(axis=1)[:span,np.newaxis], out=expMeans[:span,:])
                R = sparseScalarQuotientOfDot(W[d:end_d,:], expMeans[:span,:], vocab, out=R)

                stepSize = (Tau + batchIter) ** -Kappa
                batchIter += 1

                # Do a gradient update of the vocab
                vocabScale += (R.T.dot(expMeans[:span,:])).T
                # vocabScale *= vocab
                # normalizerows_ip(vocabScale)
                # # vocabScale += vocabPrior
                # vocabScale *= stepSize
                # vocab *= (1 - stepSize)
                # vocab += vocabScale

                diff = (means[d:end_d,:] - mu)
                means_cov_with_x_a += diff.T.dot(diff)

#       print("Vec-Means: %f, %f, %f, %f" % (means.min(), means.mean(), means.std(), means.max()))
        debugFn (itr, means, "means", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, docLens)
        
        if logFrequency > 0 and itr % logFrequency == 0:
            modelState = ModelState(F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT * sigScale, vocab, vocabPrior, Ab, dtype, MODEL_NAME)
            queryState = QueryState(means, expMeans, varcs, docLens)

            boundValues[bvIdx] = var_bound(DataSet(W, feats=X), modelState, queryState, XTX)
            boundLikes[bvIdx]  = log_likelihood(DataSet(W, feats=X), modelState, queryState)
            boundIters[bvIdx]  = itr
            perp = perplexity_from_like(boundLikes[bvIdx], docLens.sum())
            print (time.strftime('%X') + " : Iteration %d: Perplexity %4.0f bound %f" % (itr, perp, boundValues[bvIdx]))
            if bvIdx > 0 and  boundValues[bvIdx - 1] > boundValues[bvIdx]:
                printStderr ("ERROR: bound degradation: %f > %f" % (boundValues[bvIdx - 1], boundValues[bvIdx]))
#           print ("Means: min=%f, avg=%f, max=%f\n\n" % (means.min(), means.mean(), means.max()))

            # Check to see if the improvement in the likelihood has fallen below the threshold
            if bvIdx > 1 and boundIters[bvIdx] > 20:
                lastPerp = perplexity_from_like(boundLikes[bvIdx - 1], docLens.sum())
                if lastPerp - perp < 1:
                    boundIters, boundValues, likelyValues = clamp (boundIters, boundValues, boundLikes, bvIdx)
                    break
            bvIdx += 1
        
    revert_sort = np.argsort(sortIdx, kind=STABLE_SORT_ALG)
    means       = means[revert_sort,:]
    varcs       = varcs[revert_sort,:]
    docLens     = docLens[revert_sort]
    
    return \
        ModelState(F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT * sigScale, vocab, vocabPrior, Ab, dtype, MODEL_NAME), \
        QueryState(means, expMeans, varcs, docLens), \
        (boundIters, boundValues, boundLikes)
 
def query(data, modelState, queryState, queryPlan):
    '''
    Given a _trained_ model, attempts to predict the topics for each of
    the inputs.
    
    Params:
    data - the dataset of words, features and links of which only words and
           features are used in this model
    modelState - the _trained_ model
    queryState - the query state generated for the query dataset
    queryPlan  - used in this case as we need to tighten up the approx
    
    Returns:
    The model state and query state, in that order. The model state is
    unchanged, the query is.
    '''
    W, X = data.words, data.feats
    D, _ = W.shape
    
    # Unpack the the structs, for ease of access and efficiency
    iterations, epsilon, logFrequency, fastButInaccurate, debug = queryPlan.iterations, queryPlan.epsilon, queryPlan.logFrequency, queryPlan.fastButInaccurate, queryPlan.debug
    means, expMeans, varcs, n = queryState.means, queryState.expMeans, queryState.varcs, queryState.docLens
    F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, Ab, dtype = modelState.F, modelState.P, modelState.K, modelState.A, modelState.R_A, modelState.fv, modelState.Y, modelState.R_Y, modelState.lfv, modelState.V, modelState.sigT, modelState.vocab, modelState.vocabPrior, modelState.Ab, modelState.dtype
    
    # Debugging
    debugFn = _debug_with_bound if debug else _debug_with_nothing
    _debug_with_bound.old_bound = 0
    
    # Necessary values
    isigT = la.inv(sigT)

    lastPerp = 1E+300 if dtype is np.float64 else 1E+30
    for itr in range(iterations):
        # Counts of topic assignments
        expMeans = np.exp(means - means.max(axis=1)[:,np.newaxis], out=expMeans)
        R = sparseScalarQuotientOfDot(W, expMeans, vocab)
        S = expMeans * R.dot(vocab.T)

        # the variance
        varcs[:] = 1./((n * (K-1.)/K)[:,np.newaxis] + isigT.flat[::K+1])
        debugFn (itr, varcs, "query-varcs", W, X, None, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, n)
        
        # Update the Means
        rhs = X.dot(A.T).dot(isigT)
        rhs += S
        rhs += n[:,np.newaxis] * means.dot(Ab)
        rhs -= n[:,np.newaxis] * rowwise_softmax(means, out=means)
        
        # Long version
        inverses = dict()
        for d in range(D):
            if not n[d] in inverses:
                inverses[n[d]] = la.inv(isigT + n[d] * Ab)
            lhs = inverses[n[d]]
            means[d,:] = lhs.dot(rhs[d,:])
        debugFn (itr, means, "query-means", W, X, None, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, n)

        like = log_likelihood(data, modelState, QueryState(means, expMeans, varcs, n))
        perp = perplexity_from_like(like, data.word_count)
        if itr > 20 and lastPerp - perp < 1:
            break
        lastPerp = perp

    
    return modelState, queryState # query vars altered in-place
   


def log_likelihood (data, modelState, queryState):
    ''' 
    Return the log-likelihood of the given data W according to the model
    and the parameters inferred for the entries in W stored in the 
    queryState object.
    '''
    return np.sum( \
        sparseScalarProductOfSafeLnDot(\
            data.words, \
            rowwise_softmax(queryState.means), \
            modelState.vocab \
        ).data \
    )
    
def var_bound(data, modelState, queryState, XTX=None):
    '''
    Determines the variational bounds. Values are mutated in place, but are
    reset afterwards to their initial values. So it's safe to call in a serial
    manner.
    '''
    
    # Unpack the the structs, for ease of access and efficiency
    # W, X = data.words, data.feats
    # D, _ = W.shape
    # means, expMeans, varcs, docLens = queryState.means, queryState.expMeans, queryState.varcs, queryState.docLens
    # F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, Ab, dtype = modelState.F, modelState.P, modelState.K, modelState.A, modelState.R_A, modelState.fv, modelState.Y, modelState.R_Y, modelState.lfv, modelState.V, modelState.sigT, modelState.vocab, modelState.Ab, modelState.dtype
    #
    # # Calculate some implicit  variables
    # isigT = la.inv(sigT)
    # lnDetSigT = lnDetOfDiagMat(sigT)
    # verifyProper(lnDetSigT, "lnDetSigT")
    #
    # if XTX is None:
    #     XTX = X.T.dot(X)
    #
    # bound = 0
    #
    # # Distribution over latent space
    # bound -= (P*K)/2. * LN_OF_2_PI
    # bound -= P * lnDetSigT
    # bound -= K * P * log(lfv)
    # bound -= 0.5 * np.sum(1./lfv * isigT.dot(Y) * Y)
    # bound -= 0.5 * K * np.trace(R_Y)
    #
    # # And its entropy
    # detR_Y = safeDet(R_Y, "R_Y")
    # bound += 0.5 * LN_OF_2_PI_E + P/2. * lnDetSigT + K/2. * log(detR_Y)
    #
    # # Distribution over mapping from features to topics
    # diff   = (A - Y.dot(V))
    # bound -= (F*K)/2. * LN_OF_2_PI
    # bound -= F * lnDetSigT
    # bound -= K * P * log(fv)
    # bound -= 0.5 * np.sum (1./lfv * isigT.dot(diff) * diff)
    # bound -= 0.5 * K * np.trace(R_A)
    #
    # # And its entropy
    # detR_A = safeDet(R_A, "R_A")
    # bound += 0.5 * LN_OF_2_PI_E + F/2. * lnDetSigT + K/2. * log(detR_A)
    #
    # # Distribution over document topics
    # bound -= (D*K)/2. * LN_OF_2_PI
    # bound -= D/2. * lnDetSigT
    # diff   = means - X.dot(A.T)
    # bound -= 0.5 * np.sum (diff.dot(isigT) * diff)
    # bound -= 0.5 * np.sum(varcs * np.diag(isigT)[np.newaxis,:]) # = -0.5 * sum_d tr(V_d \Sigma^{-1}) when V_d is diagonal only.
    # bound -= 0.5 * K * np.trace(XTX.dot(R_A))
    #
    # # And its entropy
    # bound += 0.5 * D * K * LN_OF_2_PI_E + 0.5 * np.sum(np.log(varcs))
    #
    # # Distribution over word-topic assignments, and their entropy
    # # and distribution over words. This is re-arranged as we need
    # # means for some parts, and exp(means) for other parts
    # expMeans = np.exp(means - means.max(axis=1)[:,np.newaxis], out=expMeans)
    # R = sparseScalarQuotientOfDot(W, expMeans, vocab)  # D x V   [W / TB] is the quotient of the original over the reconstructed doc-term matrix
    # S = expMeans * (R.dot(vocab.T)) # D x K
    #
    # bound += np.sum(docLens * np.log(np.sum(expMeans, axis=1)))
    # bound += np.sum(sparseScalarProductOfSafeLnDot(W, expMeans, vocab).data)
    #
    # bound += np.sum(means * S)
    # bound += np.sum(2 * ssp.diags(docLens,0) * means.dot(Ab) * means)
    # bound -= 2. * scaledSelfSoftDot(means, docLens)
    # bound -= 0.5 * np.sum(docLens[:,np.newaxis] * S * (np.diag(Ab))[np.newaxis,:])
    #
    # bound -= np.sum(means * S)
    #
    # return bound

    return 0
        

# ==============================================================
# PRIVATE HELPERS
# ==============================================================

@static_var("old_bound", 0)
def _debug_with_bound (itr, var_value, var_name, W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, n):
    if np.isnan(var_value).any():
        printStderr ("WARNING: " + var_name + " contains NaNs")
    if np.isinf(var_value).any():
        printStderr ("WARNING: " + var_name + " contains INFs")
    if var_value.dtype != dtype:
        printStderr ("WARNING: dtype(" + var_name + ") = " + str(var_value.dtype))
    
    modelState = ModelState(F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, Ab, dtype, MODEL_NAME)
    queryState = QueryState(means, means.copy(), varcs, n)
    
    old_bound = _debug_with_bound.old_bound
    bound     = var_bound(DataSet(W, feats=X), modelState, queryState, XTX)
    likely    = log_likelihood(DataSet(W, feats=X), modelState, queryState)
    diff = "" if old_bound == 0 else "%11.2f" % (bound - old_bound)
    _debug_with_bound.old_bound = bound
    
    if isnan(bound) or int(bound - old_bound) < 0:
        printStderr ("Iter %3d Update %-10s Bound %15.2f (%11s    ) Perplexity %4.2f" % (itr, var_name, bound, diff, np.exp(-likely/W.sum())))
    else:
        print ("Iter %3d Update %-10s Bound %15.2f (%11s) Perplexity %4.2f" % (itr, var_name, bound, diff, np.exp(-likely/W.sum())))
    
def _debug_with_nothing (itr, var_value, var_name, W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, Ab, n):
    pass


def isOk(mat):
    if np.isnan(mat).any():
        print ("Matrix has NaNs")
    if np.isinf(mat).any():
        print ("Matrix has INFs")
    if not (np.isnan(mat).any() or np.isinf(mat).any()):
        print ("Matrix is OK")


