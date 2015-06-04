# -*- coding: utf-8 -*-
'''
Implements a Matrix-Variate Correlated Relational Model

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
from util.sigmoid_utils import rowwise_softmax, scaledSelfSoftDot, \
    colwise_softmax
from util.sparse_elementwise import sparseScalarQuotientOfDot, \
    sparseScalarQuotientOfNormedDot, sparseScalarProductOfSafeLnDot
from util.misc import printStderr, static_var
from util.overflow_safe import safe_log_det, safe_log
from model.evals import perplexity_from_like

from math import isnan

    
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

VocabPrior = 0.1

DEBUG=False

MODEL_NAME="mtm/vb2"

# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')                            

QueryState = namedtuple ( \
    'QueryState', \
    'means varcs docLens'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K topicMean topicCov vocab A dtype name'
)

# ==============================================================
# PUBLIC API
# ==============================================================

def wordDists(model):
    return model.vocab

def is_undirected_link_predictor():
    return False

def topicDists(query):
    result  = np.exp(query.topicMean - query.topicMean.sum(axis=1))
    result /= result.sum(axis=1)
    return result

def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState(model.K, model.topicMean.copy(), model.topicCov.copy(), model.vocab.copy(), model.A.copy(), model.dtype, model.name)

def newModelAtRandom(data, K, dtype=DTYPE):
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

    # Pick some random documents as the vocabulary
    vocab = np.ones((K, T), dtype=dtype)
    doc_ids = rd.randint(0, data.doc_count, size=K)
    for k in range(K):
        sample_doc = data.words[doc_ids[k], :]
        vocab[k, sample_doc.indices] += sample_doc.data # use plus equals in case we
        vocab[k, :] /= vocab[k, :].sum()                # later use multiple docs per
                                                        # vocab component
    topicMean = rd.random((K,)).astype(dtype)
    topicMean /= np.sum(topicMean)
    
#    itopicCov = np.eye(K)
#    topicCov  = la.inv(itopicCov)
    topicCov  = np.eye(K, dtype=dtype)
    
    A = np.eye(K, dtype=dtype) - 1./K
    
    return ModelState(K, topicMean, topicCov, vocab, A, dtype, MODEL_NAME)

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
    varcs = np.ones((D,K), dtype=dtype)
    
    return QueryState(means, varcs, docLens)


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
    W, L, LT, X = data.words, data.links, ssp.csr_matrix(data.links.T), data.feats
    D,_ = W.shape
    out_links = np.squeeze(np.asarray(data.links.sum(axis=1)))

    # Unpack the the structs, for ease of access and efficiency
    iterations, epsilon, logFrequency, diagonalPriorCov, debug = trainPlan.iterations, trainPlan.epsilon, trainPlan.logFrequency, trainPlan.fastButInaccurate, trainPlan.debug
    means, varcs, docLens = queryState.means, queryState.varcs, queryState.docLens
    K, topicMean, topicCov, vocab, A, dtype = modelState.K, modelState.topicMean, modelState.topicCov, modelState.vocab, modelState.A, modelState.dtype

    emit_counts = docLens + out_links

    # Book-keeping for logs
    boundIters, boundValues, likelyValues = [], [], []

    if debug:
        debugFn = _debug_with_bound

        initLikely = log_likelihood(data, modelState, queryState)
        initPerp   = perplexity_from_like(initLikely, data.word_count)
        print ("Initial perplexity is: %.2f" % initPerp)
    else:
        debugFn = _debug_with_nothing

    # Initialize some working variables
    W_weight  = W.copy()
    L_weight  = L.copy()
    LT_weight = LT.copy()

    pseudoObsMeans = K + NIW_PSEUDO_OBS_MEAN
    pseudoObsVar   = K + NIW_PSEUDO_OBS_VAR
    priorSigT_diag = np.ndarray(shape=(K,), dtype=dtype)
    priorSigT_diag.fill (NIW_PSI)

    # Iterate over parameters
    for itr in range(iterations):

        # We start with the M-Step, so the parameters are consistent with our
        # initialisation of the RVs when we do the E-Step

        # Update the mean and covariance of the prior
        topicMean = means.sum(axis = 0) / (D + pseudoObsMeans) \
                  if USE_NIW_PRIOR \
                  else means.mean(axis=0)
        debugFn (itr, topicMean, "topicMean", data, K, topicMean, topicCov, vocab, dtype, means, varcs, A, docLens)

        if USE_NIW_PRIOR:
            diff = means - topicMean[np.newaxis,:]
            topicCov = diff.T.dot(diff) \
                 + pseudoObsVar * np.outer(topicMean, topicMean)
            topicCov += np.diag(varcs.mean(axis=0) + priorSigT_diag)
            topicCov /= (D + pseudoObsVar - K)
        else:
            topicCov = np.cov(means.T) if topicCov.dtype == np.float64 else np.cov(means.T).astype(dtype)
            topicCov += np.diag(varcs.mean(axis=0))

        if diagonalPriorCov:
            diag = np.diag(topicCov)
            topicCov = np.diag(diag)
            itopicCov = np.diag(1./ diag)
        else:
            itopicCov = la.inv(topicCov)

        debugFn (itr, topicCov, "topicCov", data, K, topicMean, topicCov, vocab, dtype, means, varcs, A, docLens)
#        print("                topicCov.det = " + str(la.det(topicCov)))

        # Building Blocks - temporarily replaces means with exp(means)
        expMeansCol = np.exp(means - means.max(axis=0)[np.newaxis, :])
        lse_at_k = np.sum(expMeansCol, axis=0)
        F = 0.5 * means \
          - (1. / (2*D + 2)) * means.sum(axis=0) \
          - expMeansCol / lse_at_k[np.newaxis, :]

        expMeansRow = np.exp(means - means.max(axis=1)[:, np.newaxis])
        W_weight   = sparseScalarQuotientOfDot(W, expMeansRow, vocab, out=W_weight)

        # Update the vocabularies

        vocab *= (W_weight.T.dot(expMeansRow)).T # Awkward order to maintain sparsity (R is sparse, expMeans is dense)
        vocab += VocabPrior
        vocab = normalizerows_ip(vocab)

        docVocab = (expMeansCol / lse_at_k[np.newaxis, :]).T # FIXME Dupes line in definitino of F

        # Recalculate w_top_sums with the new vocab and log vocab improvement
        W_weight = sparseScalarQuotientOfDot(W, expMeansRow, vocab, out=W_weight)
        w_top_sums = W_weight.dot(vocab.T) * expMeansRow

        debugFn (itr, vocab, "vocab", data, K, topicMean, topicCov, vocab, dtype, means, varcs, A, docLens)

        # Now do likewise for the links, do it twice to model in-counts (first) and
        # out-counts (Second). The difference is the transpose
        LT_weight    = sparseScalarQuotientOfDot(LT, expMeansRow, docVocab, out=LT_weight)
        l_intop_sums = LT_weight.dot(docVocab.T) * expMeansRow
        in_counts    = l_intop_sums.sum(axis=0)

        L_weight     = sparseScalarQuotientOfDot(L, expMeansRow, docVocab, out=L_weight)
        l_outtop_sums = L_weight.dot(docVocab.T) * expMeansRow

        # Reset the means and use them to calculate the weighted sum of means
        meanSum = means.sum(axis=0) * in_counts

        # And now this is the E-Step, though itr's followed by updates for the
        # parameters also that handle the log-sum-exp approximation.

        # Update the Variances: var_d = (2 N_d * A + itopicCov)^{-1}
        varcs = np.reciprocal(docLens[:, np.newaxis] * (0.5 - 1./K) + np.diagonal(topicCov))
        debugFn (itr, varcs, "varcs", data, K, topicMean, topicCov, vocab, dtype, means, varcs, A, docLens)

        # Update the Means
        rhs  = w_top_sums.copy()
        rhs += l_intop_sums
        rhs += l_outtop_sums
        rhs += itopicCov.dot(topicMean)
        rhs += emit_counts[:, np.newaxis] * (means.dot(A) - rowwise_softmax(means))
        rhs += in_counts[np.newaxis, :] * F
        if diagonalPriorCov:
            raise ValueError("Not implemented")
        else:
            for d in range(D):
                rhs_         = rhs[d, :] + (1. / (4 * D + 4)) * (meanSum - in_counts * means[d, :])
                means[d, :]  = la.inv(itopicCov + emit_counts[d] * A + np.diag(D * in_counts / (2 * D + 2))).dot(rhs_)
                if np.any(np.isnan(means[d, :])) or np.any (np.isinf(means[d, :])):
                    pass

                if np.any(np.isnan(np.exp(means[d, :] - means[d, :].max()))) or np.any (np.isinf(np.exp(means[d, :] - means[d, :].max()))):
                    pass

        debugFn (itr, means, "means", data, K, topicMean, topicCov, vocab, dtype, means, varcs, A, docLens)

        if True: #logFrequency > 0 and itr % logFrequency == 0:
            modelState = ModelState(K, topicMean, topicCov, vocab, A, dtype, MODEL_NAME)
            queryState = QueryState(means, varcs, docLens)

            boundValues.append(var_bound(data, modelState, queryState))
            likelyValues.append(log_likelihood(data, modelState, queryState))
            boundIters.append(itr)

            print (time.strftime('%X') + " : Iteration %d: bound %f \t Perplexity: %.2f" % (itr, boundValues[-1], perplexity_from_like(likelyValues[-1], docLens.sum())))
            if len(boundValues) > 1:
                if boundValues[-2] > boundValues[-1]:
                    printStderr ("ERROR: bound degradation: %f > %f" % (boundValues[-2], boundValues[-1]))

                # Check to see if the improvement in the bound has fallen below the threshold
                if False and itr > 100 and abs(perplexity_from_like(likelyValues[-1], docLens.sum()) - perplexity_from_like(likelyValues[-2], docLens.sum())) < 1.0:
                    break


    return \
        ModelState(K, topicMean, topicCov, vocab, A, dtype, MODEL_NAME), \
        QueryState(means, varcs, docLens), \
        (np.array(boundIters), np.array(boundValues), np.array(likelyValues))


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
    means, varcs, n = queryState.means, queryState.varcs, queryState.docLens
    K, topicMean, topicCov, vocab, A, dtype = modelState.K, modelState.topicMean, modelState.topicCov, modelState.vocab, modelState.A, modelState.dtype
    
    debugFn = _debug_with_bound if debug else _debug_with_nothing
    W = data.words
    D = W.shape[0]

    expMeansOut = np.exp(means - means.max(axis=1)[:, np.newaxis])
    expMeansIn  = np.exp(means - means.max(axis=0)[np.newaxis, :])
    lse_at_k    = expMeansIn.sum(axis=0)
    
    # Necessary temp variables (notably the count of topic to word assignments
    # per topic per doc)
    itopicCov = la.inv(topicCov)
    
    # Update the Variances
    varcs = 1./((n * (K-1.)/K)[:,np.newaxis] + itopicCov.flat[::K+1])
    debugFn (0, varcs, "varcs", W, K, topicMean, topicCov, vocab, dtype, means, varcs, A, n)    
    
    R = W.copy()
    for itr in range(iterations):
        R = sparseScalarQuotientOfDot(W, expMeansOut, vocab, out=R)
        V = expMeansOut * R.dot(vocab.T)
        
        # Update the Means
        rhs = V.copy()
        rhs += n[:, np.newaxis] * means.dot(A) + itopicCov.dot(topicMean)
        rhs -= n[:, np.newaxis] * rowwise_softmax(means, out=means)
        if diagonalPriorCov:
            means = varcs * rhs
        else:
            for d in range(D):
                means[d, :] = la.inv(itopicCov + n[d] * A).dot(rhs[d, :])
        
        debugFn (itr, means, "means", W, K, topicMean, topicCov, vocab, dtype, means, varcs, A, n)        
        
    
    return modelState, queryState


def log_likelihood (data, modelState, queryState):
    ''' 
    Return the log-likelihood of the given data W according to the model
    and the parameters inferred for the entries in W stored in the 
    queryState object.
    '''
    probs = rowwise_softmax(queryState.means)
    doc_dist = colwise_softmax(queryState.means)

    word_likely = np.sum( \
        sparseScalarProductOfSafeLnDot(\
            data.words, \
            probs, \
            modelState.vocab \
        ).data \
    )

    link_likely = np.sum( \
        sparseScalarProductOfSafeLnDot(\
            data.links, \
            probs, \
            doc_dist \
        ).data \
    )

    return word_likely + link_likely



def var_bound_null(data, modelState, queryState):
    return 0.0


def var_bound(data, modelState, queryState):
    '''
    Determines the variational bounds. Values are mutated in place, but are
    reset afterwards to their initial values. So it's safe to call in a serial
    manner.
    '''
    
    # Unpack the the structs, for ease of access and efficiency
    W, L, X  = data.words, data.links, data.feats
    D,_ = W.shape
    means, varcs, docLens = queryState.means, queryState.varcs, queryState.docLens
    K, topicMean, topicCov, vocab, A = modelState.K, modelState.topicMean, modelState.topicCov, modelState.vocab, modelState.A
    
    # Calculate some implicit  variables
    itopicCov = la.inv(topicCov)
    
    bound = 0

    expMeansOut = np.exp(means - means.max(axis=1)[:, np.newaxis])
    expMeansIn  = np.exp(means - means.max(axis=0)[np.newaxis, :])
    lse_at_k    = expMeansIn.sum(axis=0)
    
    if USE_NIW_PRIOR:
        pseudoObsMeans = K + NIW_PSEUDO_OBS_MEAN
        pseudoObsVar   = K + NIW_PSEUDO_OBS_VAR

        # distribution over topic covariance
        bound -= 0.5 * K * pseudoObsVar * log(NIW_PSI)
        bound -= 0.5 * K * pseudoObsVar * log(2)
        bound -= fns.multigammaln(pseudoObsVar / 2., K)
        bound -= 0.5 * (pseudoObsVar + K - 1) * safe_log_det(topicCov)
        bound += 0.5 * NIW_PSI * np.trace(itopicCov)

        # and its entropy
        # is a constant which we skip
        
        # distribution over means
        bound -= 0.5 * K * log(1./pseudoObsMeans) * safe_log_det(topicCov)
        bound -= 0.5 / pseudoObsMeans * (topicMean).T.dot(itopicCov).dot(topicMean)
        
        # and its entropy
        bound += 0.5 * safe_log_det(topicCov) # +  a constant
        
    
    # Distribution over document topics
    bound -= (D*K)/2. * LN_OF_2_PI
    bound -= D/2. * la.det(topicCov)
    diff   = means - topicMean[np.newaxis,:]
    bound -= 0.5 * np.sum (diff.dot(itopicCov) * diff)
    bound -= 0.5 * np.sum(varcs * np.diag(itopicCov)[np.newaxis,:]) # = -0.5 * sum_d tr(V_d \Sigma^{-1}) when V_d is diagonal only.
       
    # And its entropy
#     bound += 0.5 * D * K * LN_OF_2_PI_E + 0.5 * np.sum(np.log(varcs)) 


    # Distribution over word-topic assignments and words and the formers
    # entropy, and similaarly for out-links. This is somewhat jumbled to
    # avoid repeatedly taking the exp and log of the means
    W_weights  = sparseScalarQuotientOfDot(W, expMeansOut, vocab)  # D x V   [W / TB] is the quotient of the original over the reconstructed doc-term matrix
    w_top_sums = expMeansOut * (W_weights.dot(vocab.T)) # D x K

    L_weights  = sparseScalarQuotientOfNormedDot(L, expMeansOut, expMeansIn, lse_at_k)
    l_top_sums = L_weights.dot(expMeansIn) / lse_at_k[np.newaxis, :] * expMeansOut
    
    bound += np.sum(docLens * np.log(np.sum(expMeansOut, axis=1)))
    bound += np.sum(sparseScalarProductOfSafeLnDot(W, expMeansOut, vocab).data)
    # means = np.log(expMeans, out=expMeans)
    #means = safe_log(expMeansOut, out=means)
    
    bound += np.sum(means * w_top_sums)
    bound += np.sum(2 * ssp.diags(docLens,0) * means.dot(A) * means)
    bound -= 2. * scaledSelfSoftDot(means, docLens)
    bound -= 0.5 * np.sum(docLens[:,np.newaxis] * w_top_sums * (np.diag(A))[np.newaxis,:])
    
    bound -= np.sum(means * w_top_sums)
    
    
    return bound


def softmax(x):
    r  = x.copy()
    r -= r.max()
    r  = np.exp(r)
    r /= np.sum(x)
    return r


def min_link_probs(model, topics, links):
    '''
    For every document, for each of the given links, determine the
    probability of the least likely link (i.e the document-specific
    minimum of probabilities).

    :param model: the model object
    :param topics: the topics that were inferred for each document
        represented by the links matrix
    :param links: a DxD matrix of links for each document (row)
    :return: a D-dimensional vector with the minimum probabilties for each
        link
    '''
    D = topics.means.shape[0]
    col_maxes = topics.means.max(axis=0)
    lse_at_k = np.sum(np.exp(topics.means - col_maxes), axis=0)
    mins = np.empty((D,), dtype=model.dtype)
    for d in range(D):
        topDist = softmax(topics.means[d, :])
        probs = []
        for i in range(len(links[d,:].indices)):
            l = links[d,:].indices[i]
            linkDist  = np.exp(topics.means[l, :] - col_maxes) / lse_at_k
            probs.append(np.dot(topDist, linkDist))
        mins[d] = min(probs)

    return mins

def link_probs(model, topics, min_link_probs):
    '''
    Generate the probability of a link for all possible pairs of documents,
    but only store those probabilities that are bigger than or equal to the
    minimum. This ensures, hopefully, that we don't materialise a complete
    DxD matrix, but rather the minimum needed to determine the mean
    average precisions

    :param model: the trained model
    :param topics: the topics for each of the documents we're generating
        links for
    :param min_link_probs: the minimum link probability for each document
    :return: a (hopefully) sparse DxD matrix of link probabilities
    '''
    # We build the result up as a COO matrix
    rows = []
    cols = []
    vals = []

    # Calculate the softmax transform parameters
    D = topics.means.shape[0]
    linkDist = colwise_softmax(topics.means)

    # Infer the link probabilities
    for d in range(D):
        topDistAtD = softmax(topics.means[d, :])
        probs      = topDistAtD.dot(linkDist)
        relevant   = np.where(probs >= min_link_probs[d])[0]

        rows.extend([d] * len(relevant))
        cols.extend(relevant)
        vals.extend(probs[relevant])

    # Build the COO matrix, then covert it to CSR. Converts lists to numpy
    # arrays to ensure appropriate dtypes
    r = np.array(rows, dtype=np.int32)
    c = np.array(cols, dtype=np.int32)
    v = np.array(vals, dtype=model.dtype)

    return ssp.coo_matrix((v, (r, c)), shape=(D, D)).tocsr()

# ==============================================================
# PUBLIC HELPERS
# ==============================================================


@static_var("old_bound", 0)
def _debug_with_bound (itr, var_value, var_name, data, K, topicMean, topicCov, vocab, dtype, means, varcs, A, n):
    if np.isnan(var_value).any():
        printStderr ("WARNING: " + var_name + " contains NaNs")
    if np.isinf(var_value).any():
        printStderr ("WARNING: " + var_name + " contains INFs")
    if var_value.dtype != dtype:
        printStderr ("WARNING: dtype(" + var_name + ") = " + str(var_value.dtype))

    model = ModelState(K, topicMean, topicCov, vocab, A, dtype, MODEL_NAME)
    query = QueryState(means, varcs, n)

    old_bound = _debug_with_bound.old_bound
    bound     = var_bound(data, model, query)
    diff = "" if old_bound == 0 else "%15.4f" % (bound - old_bound)
    _debug_with_bound.old_bound = bound
    
    addendum = ""
    if var_name == "topicCov":
        try:
            addendum = "det(topicCov) = %g" % (la.det(topicCov))
        except:
            addendum = "det(topicCov) = <undefined>"
    
    if isnan(bound):
        printStderr ("Bound is NaN")
    else:
        perp = perplexity_from_like(log_likelihood(data, model, query), data.word_count)
        if int(bound - old_bound) < 0:
            printStderr ("Iter %3d Update %-15s Bound %22f (%15s) (%5.0f)     %s" % (itr, var_name, bound, diff, perp, addendum))
        else:
            print ("Iter %3d Update %-15s Bound %22f (%15s) (%5.0f)  %s" % (itr, var_name, bound, diff, perp, addendum))

def _debug_with_nothing (itr, var_value, var_name, W, K, topicMean, topicCov, vocab, dtype, means, varcs, A, n):
    pass