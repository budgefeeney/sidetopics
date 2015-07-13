'''
Implements a correlated topic model, similar to that described by Blei
but using the Bouchard product of sigmoid bounds instead of Laplace
approximation.

Created on 17 Jan 2014

@author: bryanfeeney
'''

from collections import namedtuple
from math import log
from model.ctm import printStderr, verifyProper, LN_OF_2_PI, \
    LN_OF_2_PI_E, DTYPE
from util.array_utils import normalizerows_ip
from util.overflow_safe import safe_log_one_plus_exp_of
from util.sparse_elementwise import sparseScalarQuotientOfDot,\
    sparseScalarProductOfSafeLnDot, scaledSumOfLnOnePlusExp
import model.ctm as ctm
import numpy as np
import numpy.random as rd
import scipy.linalg as la
import scipy.sparse as ssp
from math import isnan

from util.sigmoid_utils import rowwise_softmax
from util.misc import static_var, converged, clamp

from model.common import DataSet
from util.overflow_safe import safeDet, lnDetOfDiagMat

import time
from model.evals import perplexity_from_like

#mpl.use('Agg')


    
# ==============================================================
# CONSTANTS
# ==============================================================

DEBUG=False
MODEL_NAME="stm-yv/bouchard"

USE_NIW_PRIOR=True

VocabPrior=1.1

# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')                            

QueryState = namedtuple ( \
    'QueryState', \
    'means expMeans varcs lxi s docLens'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'F P K A R_A fv Y R_Y lfv V sigT vocab vocabPrior dtype name'
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
    
    return ModelState(\
        model.F, model.P, model.K, \
        copy(model.A), copy(model.R_A), model.fv, \
        copy(model.Y), copy(model.R_Y), model.lfv, \
        copy(model.V), \
        copy(model.sigT), \
        copy(model.vocab), \
        model.vocabPrior, \
        model.dtype, \
        model.name)


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
    
    base = ctm.newModelAtRandom(data, K, vocabPrior, dtype)
    _,F = data.feats.shape
    Y = rd.random((K,P)).astype(dtype)
    R_Y = latFeatVar * np.eye(P,P, dtype=dtype)
    
    V = rd.random((P,F)).astype(dtype)
    A = Y.dot(V)
    R_A = featVar * np.eye(F,F, dtype=dtype)
    
    return ModelState(F, P, K, A, R_A, featVar, Y, R_Y, latFeatVar, V, base.sigT, base.vocab, base.vocabPrior, dtype, MODEL_NAME)

def newQueryState(data, modelState):
    '''
    Creates a new CTM Query state object. This contains all
    parameters and random variables tied to individual
    datapoints.
    
    Param:
    data - the dataset of words, features and links of which only words and
           features are used in this model
    modelState - the model state object
    
    REturn:
    A QueryState object
    '''
    base = ctm.newQueryState(data, modelState)

    return QueryState(base.means, base.expMeans, base.varcs, base.lxi, base.s, base.docLens)


def newTrainPlan(iterations = 100, epsilon=2, logFrequency=10, fastButInaccurate=False, debug=DEBUG):
    '''
    Create a training plan determining how many iterations we
    process, how often we plot the results, how often we log
    the variational bound, etc.
    '''
    base = ctm.newTrainPlan(iterations, epsilon, logFrequency, fastButInaccurate, debug)
    return TrainPlan(base.iterations, base.epsilon, base.logFrequency, base.fastButInaccurate, base.debug)


def is_undirected_link_predictor():
    '''
    Is this model only for predicting link structure, and only in the case where
    the links are undirected.
    '''
    return False


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
    updated in place, so make a defensive copy if you want it)
    A new query object with the update query parameters
    '''
    W, X = data.words, data.feats

    assert W.dtype == modelState.dtype
    assert X.dtype == modelState.dtype
    
    D,_ = W.shape
    
    # Unpack the the structs, for ease of access and efficiency
    iterations, epsilon, logFrequency, fastButInaccurate, debug = trainPlan.iterations, trainPlan.epsilon, trainPlan.logFrequency, trainPlan.fastButInaccurate, trainPlan.debug
    means, expMeans, varcs, lxi, s, n = queryState.means, queryState.expMeans, queryState.varcs, queryState.lxi, queryState.s, queryState.docLens
    F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype = modelState.F, modelState.P, modelState.K, modelState.A, modelState.R_A, modelState.fv, modelState.Y, modelState.R_Y, modelState.lfv, modelState.V, modelState.sigT, modelState.vocab, modelState.vocabPrior, modelState.dtype
    
    # Book-keeping for logs
    boundIters  = np.zeros(shape=(iterations // logFrequency,))
    boundValues = np.zeros(shape=(iterations // logFrequency,))
    likeValues  = np.zeros(shape=(iterations // logFrequency,))
    bvIdx = 0
    
    _debug_with_bound.old_bound = 0
    debugFn = _debug_with_bound if debug else _debug_with_nothing
    
    
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
    
    s.fill(0)
    
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
        
        # Diagonalize it
        sigT = np.diag(sigT.flat[::K+1])
        # and invert it.
        isigT = np.diag(np.reciprocal(sigT.flat[::K+1]))
        debugFn (itr, sigT, "sigT", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        # Building Blocks - temporarily replaces means with exp(means)
        expMeans = np.exp(means - means.max(axis=1)[:,np.newaxis], out=expMeans)
        R = sparseScalarQuotientOfDot(W, expMeans, vocab, out=R)
        S = expMeans * R.dot(vocab.T)
        
        # Update the vocabulary
        vocab *= (R.T.dot(expMeans)).T # Awkward order to maintain sparsity (R is sparse, expMeans is dense)
        vocab += vocabPrior
        vocab = normalizerows_ip(vocab)
        
        # Reset the means to their original form, and log effect of vocab update
        debugFn (itr, vocab, "vocab", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        # Finally update the parameter V
        V = la.inv(R_Y + Y.T.dot(isigT).dot(Y)).dot(Y.T.dot(isigT).dot(A))
        debugFn (itr, V, "V", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        # And now this is the E-Step, though it's followed by updates for the
        # parameters also that handle the log-sum-exp approximation.
        
        # Update the distribution on the latent space
        R_Y_base = aI_P + 1/fv * V.dot(V.T)
        R_Y = la.inv(R_Y_base)
        debugFn (itr, R_Y, "R_Y", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        Y = 1./fv * A.dot(V.T).dot(R_Y)
        debugFn (itr, Y, "Y", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        # Update the mapping from the features to topics
        A = (1./fv * (Y).dot(V) + (X.T.dot(means)).T).dot(R_A)
        debugFn (itr, A, "A", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        # Update the Means
        vMat   = (s[:,np.newaxis] * lxi - 0.5) * n[:,np.newaxis] + S
        rhsMat = vMat + X.dot(A.T).dot(isigT) # TODO Verify this
        lhsMat = np.reciprocal(np.diag(isigT)[np.newaxis,:] + n[:,np.newaxis] *  lxi)  # inverse of D diagonal matrices...
        means = lhsMat * rhsMat # as LHS is a diagonal matrix for all d, it's equivalent
                                # do doing a hadamard product for all d
        debugFn (itr, means, "means", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        # Update the Variances
        varcs = 1./(n[:,np.newaxis] * lxi + isigT.flat[::K+1])
        debugFn (itr, varcs, "varcs", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        # Update the approximation parameters
        lxi = 2 * ctm.negJakkolaOfDerivedXi(means, varcs, s)
        debugFn (itr, lxi, "lxi", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        # s can sometimes grow unboundedly
        # Follow Bouchard's suggested approach of fixing it at zero
        #
#         s = (np.sum(lxi * means, axis=1) + 0.25 * K - 0.5) / np.sum(lxi, axis=1)
#         debugFn (itr, s, "s", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        if logFrequency > 0 and itr % logFrequency == 0:
            modelState = ModelState(F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, MODEL_NAME)
            queryState = QueryState(means, expMeans, varcs, lxi, s, n)
            
            boundValues[bvIdx] = var_bound(data, modelState, queryState, XTX)
            likeValues[bvIdx]  = log_likelihood(data, modelState, queryState)
            boundIters[bvIdx]  = itr
            perp = perplexity_from_like(likeValues[bvIdx], n.sum())
            print (time.strftime('%X') + " : Iteration %d: Perplexity %4.2f  bound %f" % (itr, perp, boundValues[bvIdx]))
            if bvIdx > 0 and  boundValues[bvIdx - 1] > boundValues[bvIdx]:
                printStderr ("ERROR: bound degradation: %f > %f" % (boundValues[bvIdx - 1], boundValues[bvIdx]))
#             print ("Means: min=%f, avg=%f, max=%f\n\n" % (means.min(), means.mean(), means.max()))

            # Check to see if the improvment in the likelihood has fallen below the threshold
            if bvIdx > 1 and boundIters[bvIdx] > 50:
                lastPerp = perplexity_from_like(likeValues[bvIdx - 1], n.sum())
                if lastPerp - perp < 1:
                    boundIters, boundValues, likelyValues = clamp (boundIters, boundValues, likeValues, bvIdx)
                    return modelState, queryState, (boundIters, boundValues, likeValues)
            bvIdx += 1


    return \
        ModelState(F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, MODEL_NAME), \
        QueryState(means, expMeans, varcs, lxi, s, n), \
        (boundIters, boundValues, likeValues)

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
    iterations, epsilon, logFrequency, fastButInaccurate, debug = queryPlan.iterations, queryPlan.epsilon, queryPlan.logFrequency, queryPlan.fastButInaccurate, queryPlan.debug
    means, expMeans, varcs, lxi, s, n = queryState.means, queryState.expMeans, queryState.varcs, queryState.lxi, queryState.s, queryState.docLens
    F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype = modelState.F, modelState.P, modelState.K, modelState.A, modelState.R_A, modelState.fv, modelState.Y, modelState.R_Y, modelState.lfv, modelState.V, modelState.sigT, modelState.vocab, modelState.vocabPrior, modelState.dtype

    # Necessary temp variables (notably the count of topic to word assignments
    # per topic per doc)
    isigT = la.inv(sigT)
        
    W,X = data.words, data.feats
        
    # Enable logging or not. If enabled, we need the inner product of the feat matrix
    if debug:
        XTX = X.T.dot(X)
        debugFn = _debug_with_bound
        _debug_with_bound.old_bound=0
    else:
        XTX = None
        debugFn = _debug_with_nothing
    
    # Iterate over parameters
    lastPerp = 1E+300 if dtype is np.float64 else 1E+30
    for itr in range(iterations):
        # Estimate Z_dvk
        expMeans = np.exp(means - means.max(axis=1)[:,np.newaxis], out=expMeans)
        R = sparseScalarQuotientOfDot(W, expMeans, vocab)
        S = expMeans * R.dot(vocab.T)
        
        # Update the Means
        vMat   = (2  * s[:,np.newaxis] * lxi - 0.5) * n[:,np.newaxis] + S
        rhsMat = vMat + X.dot(A.T).dot(isigT) # TODO Verify this
        lhsMat = np.reciprocal(np.diag(isigT)[np.newaxis,:] + n[:,np.newaxis] * 2 * lxi)  # inverse of D diagonal matrices...
        means = lhsMat * rhsMat # as LHS is a diagonal matrix for all d, it's equivalent
                                # to doing a hadamard product for all d
        debugFn (itr, means, "query-means", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        # Update the Variances
        varcs = 1./(2 * n[:,np.newaxis] * lxi + isigT.flat[::K+1])
        debugFn (itr, varcs, "query-varcs", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        # Update the approximation parameters
        lxi = ctm.negJakkolaOfDerivedXi(means, varcs, s)
        debugFn (itr, lxi, "query-lxi", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)
        
        # s can sometimes grow unboundedly
        # Follow Bouchard's suggested approach of fixing it at zero
        #
#         s = (np.sum(lxi * means, axis=1) + 0.25 * K - 0.5) / np.sum(lxi, axis=1)
#         debugFn (itr, s, "s", W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n)

        like = log_likelihood(data, modelState, QueryState(means, expMeans, varcs, lxi, s, n))
        perp = perplexity_from_like(like, data.word_count)
        if itr > 20 and lastPerp - perp < 1:
            break
        lastPerp = perp

    return modelState, QueryState (means, expMeans, varcs, lxi, s, n)


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


    
def var_bound(data, modelState, queryState, XTX = None):
    '''
    Determines the variational bounds. Values are mutated in place, but are
    reset afterwards to their initial values. So it's safe to call in a serial
    manner.
    '''
    # Unpack the the structs, for ease of access and efficiency
    W, X = data.words, data.feats
    D, _ = W.shape
    means, varcs, lxi, s, docLens = queryState.means, queryState.varcs, queryState.lxi, queryState.s, queryState.docLens
    F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, dtype = modelState.F, modelState.P, modelState.K, modelState.A, modelState.R_A, modelState.fv, modelState.Y, modelState.R_Y, modelState.lfv, modelState.V, modelState.sigT, modelState.vocab, modelState.dtype
    
    # Calculate some implicit  variables
    xi = ctm._deriveXi(means, varcs, s)
    isigT = la.inv(sigT)
    #lnDetSigT = np.log(la.det(sigT))
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
    # This also takes into account all the variables that 
    # constitute the bound on log(sum_j exp(mean_j)) and
    # also incorporates the implicit entropy of Z_dvk
    bound -= np.sum((means*means + varcs) * docLens[:,np.newaxis] * lxi)
    bound += np.sum(means * 2 * docLens[:,np.newaxis] * s[:,np.newaxis] * lxi)
    bound += np.sum(means * -0.5 * docLens[:,np.newaxis])
    # The last term of line 1 gets cancelled out by part of the first term in line 2
    # so neither are included here.
    
    row_maxes = means.max(axis=1)
    means -= row_maxes[:,np.newaxis]
    expMeans = np.exp(means, out=means)
    bound -= -np.sum(sparseScalarProductOfSafeLnDot(W, expMeans, vocab).data)
    
    bound -= np.sum(docLens[:,np.newaxis] * lxi * ((s*s)[:,np.newaxis] - (xi * xi)))
    bound += np.sum(0.5 * docLens[:,np.newaxis] * (s[:,np.newaxis] + xi))
#    bound -= np.sum(docLens[:,np.newaxis] * safe_log_one_plus_exp_of(xi))
    bound -= scaledSumOfLnOnePlusExp(docLens, xi)
    
    bound -= np.dot(s, docLens)
    
    means = np.log(expMeans, out=expMeans)
    means += row_maxes[:,np.newaxis]
    
    return bound


@static_var("old_bound", 0)
def _debug_with_bound (itr, var_value, var_name, W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n):
    if np.isnan(var_value).any():
        printStderr ("WARNING: " + var_name + " contains NaNs")
    if np.isinf(var_value).any():
        printStderr ("WARNING: " + var_name + " contains INFs")
    if var_value.dtype != dtype:
        printStderr ("WARNING: dtype(" + var_name + ") = " + str(var_value.dtype))
    
    modelState = ModelState(F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, MODEL_NAME)
    queryState = QueryState(means, varcs, lxi, s, n)
    
    old_bound = _debug_with_bound.old_bound
    bound     = var_bound(DataSet(W, feats=X), modelState, queryState, XTX)
    likely    = log_likelihood(DataSet(W, feats=X), modelState, queryState)
    perp      = np.exp(-likely / W.sum())
    diff = "" if old_bound == 0 else "%11.2f" % (bound - old_bound)
    _debug_with_bound.old_bound = bound
    
    if isnan(bound) or int(bound - old_bound) < 0:
        printStderr ("Iter %3d Update %-10s Bound %15.2f (%11s    ) Perp %4.2f" % (itr, var_name, bound, diff, perp))
    else:
        print ("Iter %3d Update %-10s Bound %15.2f (%11s) Perp %4.2f" % (itr, var_name, bound, diff, perp))

def _debug_with_nothing (iter, var_value, var_name, W, X, XTX, F, P, K, A, R_A, fv, Y, R_Y, lfv, V, sigT, vocab, vocabPrior, dtype, means, varcs, lxi, s, n):
    pass
