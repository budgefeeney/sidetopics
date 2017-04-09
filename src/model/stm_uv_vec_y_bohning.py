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

import time

from collections import namedtuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as ssp
import scipy.special as fns
import numpy.random as rd

from util.array_utils import normalizerows_ip
from util.sigmoid_utils import rowwise_softmax, scaledSelfSoftDot
from util.sparse_elementwise import sparseScalarQuotientOfDot, \
    sparseScalarProductOfSafeLnDot
from util.misc import printStderr, static_var
from util.overflow_safe import safe_log_det
from model.evals import perplexity_from_like
from model.common import DataSet

from math import isnan, ceil, sqrt

    
# ==============================================================
# CONSTANTS
# ==============================================================

DTYPE=np.float32 # A default, generally we should specify this in the model setup

LN_OF_2_PI   = log(2 * pi)
LN_OF_2_PI_E = log(2 * pi * e)

USE_NIW_PRIOR=False
NIW_PSI=0.1             # isotropic prior
NIW_PSEUDO_OBS_MEAN=+2  # set to NIW_NU = K + NIW_NU_STEP # this is called kappa in the code, go figure
NIW_PSEUDO_OBS_VAR=+2   # related to K
NIW_MU=0

VocabPrior = 1.1

DEBUG=False

MODEL_NAME="stm/uv_vecy/bohning"

BatchSize = 1000

# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')                            

QueryState = namedtuple ( \
    'QueryState', \
    'means docLens'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K A U Y V covA tv ltv fv lfv vocab vocabPrior dtype name'
)

# ==============================================================
# PUBLIC API
# ==============================================================

def vec(X):
    return X.flatten(order='F')

def unvec(x, row_count):
    col_count = x.shape[0] / row_count

    return x.reshape(row_count, col_count, order='F')

def wordDists(model):
    return model.vocab

def topicDists(query):
    result  = np.exp(query.topicMean - query.topicMean.sum(axis=1))
    result /= result.sum(axis=1)
    return result

def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    def safe_copy(arr):
        return None if arr is None else arr.copy()

    return ModelState(
        model.K,
        safe_copy(model.A),
        safe_copy(model.U),
        safe_copy(model.Y),
        safe_copy(model.V),
        safe_copy(model.covA),
        model.tv,
        model.ltv,
        model.fv,
        model.lfv,
        safe_copy(model.vocab),
        model.vocabPrior,
        model.dtype,
        model.name
    )

def newModelAtRandom(data, K, Q, P, tv=0.001, ltv=0.001, fv=0.001, lfv=0.001, vocabPrior=VocabPrior, dtype=DTYPE):
    '''
    Creates a new CtmModelState for the given training set and
    the given number of topics. Everything is instantiated purely
    at random. This contains all parameters independent of of
    the dataset (e.g. learnt priors)
    
    Param:
    data - the dataset of words, features and links of which only words are used in this model
    K - the number of topics
    
    Return:
    A CtmModelState object
    '''
    assert K > 1, "There must be at least two topics"
    
    _,T = data.words.shape
    _,F = data.feats.shape

    # Pick some random documents as the vocabulary
    vocab = np.ones((K,T), dtype=dtype)
    for k in range(1, K):
        docLenSum = 0
        while docLenSum < 1000:
            randomDoc  = rd.randint(0, data.doc_count, size=1)
            sample_doc = data.words[randomDoc, :]
            vocab[k, sample_doc.indices] += sample_doc.data
            docLenSum += sample_doc.sum()
        vocab[k,:] /= vocab[k,:].sum()

    # stop-word vocab
    vocab[0,:]  = data.words.sum(axis=0)
    vocab[0,:] /= vocab[0,:].sum()

    U = rd.random((F,P))
    Y = rd.random((P,Q))
    V = rd.random((K,Q))
    A = U.dot(Y).dot(V.T)
    varA = None
    
    return ModelState(K, A, U, Y,  V, varA, tv, ltv, fv, lfv, vocab, vocabPrior, dtype, MODEL_NAME)

def newQueryState(data, modelState):
    '''
    Creates a new CTM Query state object. This contains all
    parameters and random variables tied to individual
    datapoints.
    
    Param:
    data - the dataset of words, features and links of which only words are used in this model
    modelState - the model state object
    
    REturn:
    A CtmQueryState object
    '''
    K, vocab, dtype =  modelState.K, modelState.vocab, modelState.dtype
    
    D,T = data.words.shape
    assert T == vocab.shape[1], "The number of terms in the document-term matrix (" + str(T) + ") differs from that in the model-states vocabulary parameter " + str(vocab.shape[1])
    docLens = np.squeeze(np.asarray(data.words.sum(axis=1)))

    means = normalizerows_ip(rd.random((D,K)).astype(dtype))
    
    return QueryState(means, docLens)


def newTrainPlan(iterations=100, epsilon=2, logFrequency=10, fastButInaccurate=False, debug=DEBUG):
    '''
    Create a training plan determining how many iterations we
    process, how often we plot the results, how often we log
    the variational bound, etc.
    '''
    return TrainPlan(iterations, epsilon, logFrequency, fastButInaccurate, debug)

def train (data, modelState, queryState, trainPlan):
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
    W, X = data.words, data.feats
    D, T = W.shape
    F    = X.shape[1]


    # tmpNumDense = np.array([
    #     4	, 8	, 2	, 0	, 0,
    #     0	, 6	, 0	, 17, 0,
    #     12	, 13	, 1	, 7	, 8,
    #     0	, 5	, 0	, 0	, 0,
    #     0	, 6	, 0	, 0	, 44,
    #     0	, 7	, 2	, 0	, 0], dtype=np.float64).reshape((6,5))
    # tmpNum = ssp.csr_matrix(tmpNumDense)
    #
    # tmpDenomleft = (rd.random((tmpNum.shape[0], 12)) * 5).astype(np.int32).astype(np.float64) / 10
    # tmpDenomRight = (rd.random((12, tmpNum.shape[1])) * 5).astype(np.int32).astype(np.float64)
    #
    # tmpResult = tmpNum.copy()
    # tmpResult = sparseScalarQuotientOfDot(tmpNum, tmpDenomleft, tmpDenomRight)
    #
    # print (str(tmpNum.todense()))
    # print (str(tmpDenomleft.dot(tmpDenomRight)))
    # print (str(tmpResult.todense()))
    
    # Unpack the the structs, for ease of access and efficiency
    iterations, epsilon, logFrequency, diagonalPriorCov, debug = trainPlan.iterations, trainPlan.epsilon, trainPlan.logFrequency, trainPlan.fastButInaccurate, trainPlan.debug
    means, docLens = queryState.means, queryState.docLens
    K, A, U, Y,  V, covA, tv, ltv, fv, lfv, vocab, vocabPrior, dtype = \
        modelState.K, modelState.A, modelState.U, modelState.Y,  modelState.V, modelState.covA, modelState.tv, modelState.ltv, modelState.fv, modelState.lfv, modelState.vocab, modelState.vocabPrior, modelState.dtype

    tp, fp, ltp, lfp = 1./tv, 1./fv, 1./ltv, 1./lfv # turn variances into precisions

    # FIXME Use passed in hypers
    print ("tp = %f tv=%f" % (tp, tv))
    vocabPrior = np.ones(shape=(T,), dtype=modelState.dtype)

    # FIXME undo truncation
    F = 363
    A = A[:F, :]
    X = X[:, :F]
    U = U[:F, :]
    data = DataSet(words=W, feats=X)

    # Book-keeping for logs
    boundIters, boundValues, likelyValues = [], [], []
    
    debugFn = _debug_with_bound if debug else _debug_with_nothing
    
    # Initialize some working variables
    if covA is None:
        precA = (fp * ssp.eye(F) + X.T.dot(X)).todense() # As the inverse is almost always dense
        covA   = la.inv(precA, overwrite_a=True)              # it's faster to densify in advance
    uniqLens = np.unique(docLens)

    debugFn (-1, covA, "covA", W, X, means, docLens, K, A, U, Y,  V, covA, tv, ltv, fv, lfv, vocab, vocabPrior)

    H = 0.5 * (np.eye(K) - np.ones((K,K), dtype=dtype) / K)

    expMeans = means.copy()
    expMeans = np.exp(means - means.max(axis=1)[:,np.newaxis], out=expMeans)
    R = sparseScalarQuotientOfDot(W, expMeans, vocab, out=W.copy())

    lhs = H.copy()
    rhs = expMeans.copy()
    Y_rhs = Y.copy()

    # Iterate over parameters
    for itr in range(iterations):

        # Update U, V given A
        V = try_solve_sym_pos(Y.T.dot(U.T).dot(U).dot(Y), A.T.dot(U).dot(Y).T).T
        V /= V[0,0]
        U = try_solve_sym_pos(Y.dot(V.T).dot(V).dot(Y.T), A.dot(V).dot(Y.T).T).T

        # Update Y given U, V, A
        Y_rhs[:,:] = U.T.dot(A).dot(V)

        Sv, Uv = la.eigh(V.T.dot(V), overwrite_a=True)
        Su, Uu = la.eigh(U.T.dot(U), overwrite_a=True)

        s = np.outer(Sv, Su).flatten()
        s += ltv * lfv
        np.reciprocal(s, out=s)

        M = Uu.T.dot(Y_rhs).dot(Uv)
        M *= unvec(s, row_count=M.shape[0])

        Y = Uu.dot(M).dot(Uv.T)
        debugFn (itr, Y, "Y", W, X, means, docLens, K, A, U, Y,  V, covA, tv, ltv, fv, lfv, vocab, vocabPrior)


        A = covA.dot(fp * U.dot(Y).dot(V.T) + X.T.dot(means))
        debugFn (itr, A, "A", W, X, means, docLens, K, A, U, Y,  V, covA, tv, ltv, fv, lfv, vocab, vocabPrior)


        # And now this is the E-Step, though itr's followed by updates for the
        # parameters also that handle the log-sum-exp approximation.

        # TODO One big sort by size, plus batch it.

        # Update the Means

        rhs[:,:] = expMeans
        rhs *= R.dot(vocab.T)
        rhs += X.dot(A) * tp
        rhs += docLens[:,np.newaxis] * means.dot(H)
        rhs -= docLens[:,np.newaxis] * rowwise_softmax(means, out=means)
        for l in uniqLens:
            inds = np.where(docLens == l)[0]
            lhs[:,:] = l * H
            lhs[np.diag_indices_from(lhs)] += tp
            lhs[:,:] = la.inv(lhs)
            means[inds,:] = rhs[inds,:].dot(lhs) # left and right got switched going from vectors to matrices :-/


        debugFn (itr, means, "means", W, X, means, docLens, K, A, U, Y,  V, covA, tv, ltv, fv, lfv, vocab, vocabPrior)

        # Standard deviation
        # DK        = means.shape[0] * means.shape[1]
        # newTp     = np.sum(means)
        # newTp     = (-newTp * newTp)
        # rhs[:,:]  = means
        # rhs      *= means
        # newTp     = DK * np.sum(rhs) - newTp
        # newTp    /= DK * (DK - 1)
        # newTp     = min(max(newTp, 1E-36), 1E+36)
        # tp        = 1 / newTp
        # if itr % logFrequency == 0:
        #     print ("Iter %3d stdev = %f, prec = %f, np.std^2=%f, np.mean=%f" % (itr, sqrt(newTp), tp, np.std(means.reshape((D*K,))) ** 2, np.mean(means.reshape((D*K,)))))


        # Update the vocabulary
        expMeans = np.exp(means - means.max(axis=1)[:,np.newaxis], out=expMeans)
        R = sparseScalarQuotientOfDot(W, expMeans, vocab, out=R)

        vocab *= (R.T.dot(expMeans)).T # Awkward order to maintain sparsity (R is sparse, expMeans is dense)
        vocab += vocabPrior
        vocab = normalizerows_ip(vocab)

        debugFn (itr, vocab, "vocab", W, X, means, docLens, K, A, U, Y,  V, covA, tv, ltv, fv, lfv, vocab, vocabPrior)
        # print ("Iter %3d Vocab.min = %f" % (itr, vocab.min()))

        # Update the vocab prior
        # vocabPrior = estimate_dirichlet_param (vocab, vocabPrior)
        # print ("Iter %3d VocabPrior.(min, max) = (%f, %f) VocabPrior.mean=%f" % (itr, vocabPrior.min(), vocabPrior.max(), vocabPrior.mean()))


        if logFrequency > 0 and itr % logFrequency == 0:
            modelState = ModelState(K, A, U, Y,  V, covA, tv, ltv, fv, lfv, vocab, vocabPrior, dtype, modelState.name)
            queryState = QueryState(means, docLens)
            
            boundValues.append(var_bound(data, modelState, queryState))
            likelyValues.append(log_likelihood(data, modelState, queryState))
            boundIters.append(itr)
            
            print (time.strftime('%X') + " : Iteration %d: bound %f \t Perplexity: %.2f" % (itr, boundValues[-1], perplexity_from_like(likelyValues[-1], docLens.sum())))
            if len(boundValues) > 1:
                if boundValues[-2] > boundValues[-1]:
                    if debug: printStderr ("ERROR: bound degradation: %f > %f" % (boundValues[-2], boundValues[-1]))
        
                # Check to see if the improvement in the bound has fallen below the threshold
                if itr > 100 and len(likelyValues) > 3 \
                    and abs(perplexity_from_like(likelyValues[-1], docLens.sum()) - perplexity_from_like(likelyValues[-2], docLens.sum())) < 1.0:
                    break

    return \
        ModelState(K, A, U, Y,  V, covA, tv, ltv, fv, lfv, vocab, vocabPrior, dtype, modelState.name), \
        QueryState(means, expMeans, docLens), \
        (np.array(boundIters), np.array(boundValues), np.array(likelyValues))


def estimate_dirichlet_param(samples, param):
    '''
    Uses a Newton-Raphson scheme to estimating the parameter of a
    K-dimensional Dirichlet distribution

    :param samples: an NxK matrix of K-dimensional vectors drawn from
    a Dirichlet distribution
    :param param: the old value of the paramter. This is overwritten
    :return: a K-dimensional vector which is the new
    '''

    N, K = samples.shape
    p = np.sum (np.log (samples), axis=0)

    for _ in range(60):
        g  = -N * fns.digamma (param)
        g +=  N * fns.digamma (param.sum())
        g += p

        q = -N * fns.polygamma(1, param)
        np.reciprocal(q, out=q)

        z = N * fns.polygamma(1, param.sum())

        b  = np.sum (g * q)
        b /= 1/z + q.sum()

        param -= (g - b) * q

        print ("%.2f" % param.mean(), end=" --> ")
    print

    return param




def try_solve_sym_pos(A, b):
    try:
        return la.solve(A, b, sym_pos=True, overwrite_a=True, overwrite_b=True)
    except la.LinAlgError:
        solution, _, _, _ = la.lstsq(A, b, overwrite_a=True, overwrite_b=True)
        return solution

def query(data, modelState, queryState, queryPlan):
    '''
    Given a _trained_ model, attempts to predict the topics for each of
    the inputs.
    
    Params:
    data - the dataset of words, features and links of which only words are used in this model
    modelState - the _trained_ model
    queryState - the query state generated for the query dataset
    queryPlan  - used in this case as we need to tighten up the approx
    
    Returns:
    The model state and query state, in that order. The model state is
    unchanged, the query is.
    '''
    iterations, epsilon, logFrequency, diagonalPriorCov, debug = queryPlan.iterations, queryPlan.epsilon, queryPlan.logFrequency, queryPlan.fastButInaccurate, queryPlan.debug
    means, expMeans, docLens = queryState.means, queryState.expMeans, queryState.docLens
    K, A, U, Y,  V, covA, tv, ltv, fv, lfv, vocab, vocabPrior, dtype = \
        modelState.K, modelState.A, modelState.U, modelState.Y,  modelState.V, modelState.covA, modelState.tv, modelState.ltv, modelState.fv, modelState.lfv, modelState.vocab, modelState.vocabPrior, modelState.dtype

    debugFn = _debug_with_bound if debug else _debug_with_nothing
    W = data.words
    D = W.shape[0]
    
    # Necessary temp variables (notably the count of topic to word assignments
    # per topic per doc)
    isigT = la.inv(sigT)
    
    # Update the Variances
    varcs = 1./((n * (K-1.)/K)[:,np.newaxis] + isigT.flat[::K+1])
    debugFn (0, varcs, "varcs", W, K, topicMean, sigT, vocab, vocabPrior, dtype, means, varcs, A, n)
    
    lastPerp = 1E+300 if dtype is np.float64 else 1E+30
    R = W.copy()
    for itr in range(iterations):
        expMeans = np.exp(means - means.max(axis=1)[:,np.newaxis], out=expMeans)
        R = sparseScalarQuotientOfDot(W, expMeans, vocab, out=R)
        V = expMeans * R.dot(vocab.T)
        
        # Update the Means
        rhs = V.copy()
        rhs += n[:,np.newaxis] * means.dot(A) + isigT.dot(topicMean)
        rhs -= n[:,np.newaxis] * rowwise_softmax(means, out=means)
        if diagonalPriorCov:
            means = varcs * rhs
        else:
            for d in range(D):
                means[d,:] = la.inv(isigT + n[d] * A).dot(rhs[d,:])
        
        debugFn (itr, means, "means", W, K, topicMean, sigT, vocab, vocabPrior, dtype, means, varcs, A, n)
        
        like = log_likelihood(data, modelState, QueryState(means, expMeans, varcs, n))
        perp = perplexity_from_like(like, data.word_count)
        if itr > 20 and lastPerp - perp < 1:
            break
        lastPerp = perp

    return modelState, queryState


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
    
def var_bound(data, modelState, queryState):
    '''
    Determines the variational bounds. Values are mutated in place, but are
    reset afterwards to their initial values. So it's safe to call in repeatedly.
    '''
    
    # Unpack the the structs, for ease of access and efficiency
    W,X = data.words, data.feats
    D,T,F = W.shape[0], W.shape[1], X.shape[1]
    means, docLens = queryState.means, queryState.docLens
    K, A, U, Y, V, covA, tv, ltv, fv, lfv, vocab, vocabPrior, dtype = \
        modelState.K, modelState.A, modelState.U, modelState.Y,  modelState.V, modelState.covA, modelState.tv, modelState.ltv, modelState.fv, modelState.lfv, modelState.vocab, modelState.vocabPrior, modelState.dtype

    H = 0.5 * (np.eye(K) - np.ones((K,K), dtype=dtype) / K)
    Log2Pi = log(2 * pi)

    bound = 0

    # U and V are parameters with no distribution

    #
    # Y has a normal distribution, it's covariance is unfortunately an expensive computation
    #
    P, Q  = U.shape[1], V.shape[1]
    covY  = np.eye(P*Q) * (lfv * ltv)
    covY += np.kron (V.T.dot(V), U.T.dot(U))
    covY  = la.inv(covY, overwrite_a=True)

    # The expected likelihood of Y
    bound -= 0.5 * P * Q * Log2Pi
    bound -= 0.5 * P * Q * log(ltv * lfv)
    bound -= 0.5 / (lfv * ltv) * np.sum (Y * Y) # 5x faster than np.trace(Y.dot(Y.T))
    bound -= 0.5 * np.trace (covY) * (lfv * ltv)
    # the traces of the posterior+prior covariance products cancel out across likelihoods

    # The entropy of Y
    bound += 0.5 * P * Q * (Log2Pi + 1) + 0.5 * safe_log_det(covY)

    #
    # A has a normal distribution/
    #
    F, K = A.shape[0], A.shape[1]
    diff = A - U.dot(Y).dot(V.T)
    diff *= diff

    # The expected likelihood of A
    bound -= 0.5 * K * F * Log2Pi
    bound -= 0.5 * K * F * log (tv * fv)
    bound -= 0.5 / (fv * tv) * np.sum(diff)

    # The entropy of A
    bound += 0.5 * F * K * (Log2Pi + 1) + 0.5 * K * safe_log_det(covA)

    #
    # Theta, the matrix of means, has a normal distribution. Its row-covarince is diagonal
    # (i.e. it's several independent multi-var normal distros). The posterior is made
    # up of D K-dimensional normals with diagonal covariances
    #
    # We iterate through the topics in batches, to control memory use
    batchSize     = min (BatchSize, D)
    batchCount    = ceil (D / batchSize)
    feats = np.ndarray(shape=(batchSize, F), dtype=dtype)
    tops  = np.ndarray(shape=(batchSize, K), dtype=dtype)
    trace = 0
    for b in range(0, batchCount):
        start = b * batchSize
        end   = min (start + batchSize, D)
        batchSize = min(batchSize, end - start)

        feats[:batchSize,:] = X[start:end, :].toarray()
        np.dot(feats[:batchSize, :], A, out=tops[:batchSize, :])
        tops[:batchSize, :] -= means[start:end,:]
        tops[:batchSize, :] *= tops[:batchSize, :]
        trace += np.sum(tops[:batchSize, :])
    feats = None

    # The expected likelihood of the topic-assignments
    bound -= 0.5 * D * K * Log2Pi
    bound -= 0.5 * D * K * log(tv)
    bound -= 0.5 / tv * trace

    bound -= 0.5 * tv * np.sum(covA) # this trace doesn't cancel as we
                                     # don't have a posterior on tv
    # The entropy of the topic-assignments
    bound += 0.5 * D * K * (Log2Pi + 1) + 0.5 * np.sum(covA)


    # Distribution over word-topic assignments and words and the formers
    # entropy. This is somewhat jumbled to avoid repeatedly taking the
    # exp and log of the means
    # Again we batch this for safety
    batchSize  = min (BatchSize, D)
    batchCount = ceil (D / batchSize)
    V          = np.ndarray(shape=(batchSize, K), dtype=dtype)
    for b in range(0, batchCount):
        start = b * batchSize
        end   = min (start + batchSize, D)
        batchSize = min(batchSize, end - start)

        meansBatch   = means[start:end,:]
        docLensBatch = docLens[start:end]

        np.exp(meansBatch - meansBatch.max(axis=1)[:,np.newaxis], out=tops[:batchSize,:])
        expMeansBatch = tops[:batchSize,:]
        R = sparseScalarQuotientOfDot(W, expMeansBatch, vocab, start=start, end=end)  # BatchSize x V:   [W / TB] is the quotient of the original over the reconstructed doc-term matrix
        V[:batchSize,:] = expMeansBatch * (R[:batchSize,:].dot(vocab.T)) # BatchSize x K
        VBatch = V[:batchSize,:]

        bound += np.sum(docLensBatch * np.log(np.sum(expMeansBatch, axis=1)))
        bound += np.sum(sparseScalarProductOfSafeLnDot(W, expMeansBatch, vocab, start=start, end=end).data)

        bound += np.sum(meansBatch * VBatch)
        bound += np.sum(2 * ssp.diags(docLensBatch,0) * meansBatch.dot(H) * meansBatch)
        bound -= 2. * scaledSelfSoftDot(meansBatch, docLensBatch)
        bound -= 0.5 * np.sum(docLensBatch[:,np.newaxis] * VBatch * (np.diag(H))[np.newaxis,:])

        bound -= np.sum(meansBatch * VBatch)
    
    
    return bound
        

# ==============================================================
# PUBLIC HELPERS
# ==============================================================

@static_var("old_bound", 0)
def _debug_with_bound (itr, var_value, var_name, W, X, means, docLens, K, A, U, Y, V, covA, tv, ltv, fv, lfv, vocab, vocabPrior):
    if np.isnan(var_value).any():
        printStderr ("WARNING: " + var_name + " contains NaNs")
    if np.isinf(var_value).any():
        printStderr ("WARNING: " + var_name + " contains INFs")
    dtype = A.dtype
    
    old_bound = _debug_with_bound.old_bound
    data      = DataSet(W, X)
    model     = ModelState(K, A, U, Y, V, covA, tv, ltv, fv, lfv, vocab, vocabPrior, dtype, MODEL_NAME)
    query     = QueryState(means, docLens)
    bound     = var_bound(data, model, query)
    diff = "" if old_bound == 0 else "%15.4f" % (bound - old_bound)
    _debug_with_bound.old_bound = bound
    
    addendum = ""

    perp = np.exp(-log_likelihood(data, model, query) / data.word_count)
    
    if isnan(bound):
        printStderr ("Bound is NaN")
    elif int(bound - old_bound) < 0:
        printStderr ("Iter %3d Update %-15s Bound %22f (%15s) Perplexity %5.1f     %s" % (itr, var_name, bound, diff, perp, addendum))
    else:
        print ("Iter %3d Update %-15s Bound %22f (%15s) Perplexity %5.1f     %s" % (itr, var_name, bound, diff, perp, addendum))

def _debug_with_nothing (itr, var_value, var_name, W, X, means, docLens, K, A, U, Y, V, covA, tv, ltv, fv, lfv, vocab, vocabPrior):
    pass

