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
import numpy.random as rd

from sidetopics.util.array_utils import normalizerows_ip
from sidetopics.util.sigmoid_utils import rowwise_softmax, scaledSelfSoftDot, \
    colwise_softmax
from sidetopics.util.sparse_elementwise import sparseScalarQuotientOfDot, \
    sparseScalarQuotientOfNormedDot, sparseScalarProductOfSafeLnDot, \
    sparseScalarProductOfDot
from sidetopics.util.misc import printStderr, static_var
from sidetopics.util.overflow_safe import safe_log_det, safe_log
from sidetopics.model.evals import perplexity_from_like

from math import isnan

import sidetopics.model.lda_gibbs as lda_gibbs
import sidetopics.model.lda_vb_python as lda_vb

    
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

VocabPrior = 0.01

DEBUG=False

MODEL_NAME="mtm2/vb"

INIT_WITH_CTM=False

MinItersBeforeEarlyStop=50

IGAMMA_A = 10
IGAMMA_B = 10
IWISH_S_SCALE = 1
IWISH_DENOM   = 0.1

# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')                            

QueryState = namedtuple ( \
    'QueryState', \
    'outMeans outVarcs inMeans inVarcs inDocCov docLens'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K topicMean topicCov outDocCov vocab A trained dtype name'
)

# ==============================================================
# PUBLIC API
# ==============================================================


def wordDists(model):
    return model.vocab

def is_undirected_link_predictor():
    return False

def topicDists(queryState):
    result  = np.exp(queryState.topicMean - queryState.topicMean.sum(axis=1))
    result /= result.sum(axis=1)
    return result

def newModelFromExisting(model, withLdaModel=None):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState(
        model.K,
        model.topicMean.copy(),
        model.topicCov.copy(),
        model.outDocCov,
        model.vocab.copy() if withLdaModel is None else lda_vocab(withLdaModel),
        model.A.copy(),
        model.trained,
        model.dtype,
        model.name
    )


def newModelAtRandom(data, K, outDocCov=0.001, dtype=DTYPE, withLdaModel=None):
    '''
    Creates a new ModelState for the given training set and
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
    
    D,T = data.words.shape

    # Pick some random documents as the vocabulary
    if withLdaModel is not None:
        vocab = lda_vocab(withLdaModel)
    else:
        vocab = np.ones((K, T), dtype=dtype)
        doc_ids = rd.randint(0, data.doc_count, size=K)
        for k in range(K):
            sample_doc = data.words[doc_ids[k], :]
            vocab[k, sample_doc.indices] += sample_doc.data # use plus equals in case we
            vocab[k, :] /= vocab[k, :].sum()                # later use multiple docs per
                                                            # vocab component
    topicMean = rd.random((K,)).astype(dtype)
    topicMean /= np.sum(topicMean)

    topicCov  = np.eye(K, dtype=dtype)
    
    A = np.eye(K, dtype=dtype) - 1./K
    
    return ModelState(K, topicMean, topicCov, outDocCov, vocab, A, False, dtype, MODEL_NAME)

def lda_vocab(ldaModel):
    if ldaModel.name == lda_gibbs.MODEL_NAME:
        return lda_gibbs.wordDists(ldaModel)
    elif ldaModel.name == lda_vb.MODEL_NAME:
        return lda_vb.wordDists(ldaModel)
    else:
        raise ValueError("Unknown LDA implementation")

def lda_topics(ldaQuery):
    if "numSamples" in dir(ldaQuery):
        return lda_gibbs.topicDists(ldaQuery)
    else:
        return lda_vb.topicDists(ldaQuery)


def _newQueryStateFromCtm(data, model):
    import model.ctm_bohning as ctm

    ctm_model = ctm.newModelAtRandom(data, model.K, VocabPrior, model.dtype)
    ctm_query = ctm.newQueryState(data, model)
    ctm_plan  = ctm.newTrainPlan(200, epsilon=1, logFrequency=100, debug=False)

    ctm_model, ctm_query, (_, _, _) = ctm.train(data, ctm_model, ctm_query, ctm_plan)

    model.vocab[:,:]    = ctm_model.vocab
    model.topicCov[:,:] = ctm_model.sigT
    model.topicMean[:]  = ctm_model.topicMean

    K, vocab, dtype =  model.K, model.vocab, model.dtype

    D,T = data.words.shape
    assert T == vocab.shape[1], "The number of terms in the document-term matrix (" + str(T) + ") differs from that in the model-states vocabulary parameter " + str(vocab.shape[1])
    docLens = np.squeeze(np.asarray(data.words.sum(axis=1)))

    outMeans = ctm_query.means
    outVarcs = np.ones((D,K), dtype=dtype)

    inMeans = np.ndarray(shape=(D,K), dtype=dtype)
    for d in range(D):
        inMeans[d,:] = rd.multivariate_normal(outMeans[d,:], model.topicCov)
    inVarcs = np.ones((D,K), dtype=dtype)

    inDocCov  = np.ones((D,), dtype=dtype)

    return QueryState(outMeans, outVarcs, inMeans, inVarcs, inDocCov, docLens)


def _newQueryStateFromLda(data, model, ldaQueryState):
    '''
    Note this MODIFIES the model.

    Create a QueryState object intitalised with the given topics from an
    LDA run.
    :param data: the data we'll be training on
    :param model: the MTM model state
    :param ldaQueryState: the lda topics, an Lda QueryState object (Gibbs or VB)
    :return: an MTM Query State
    '''
    K, vocab, dtype =  model.K, model.vocab, model.dtype

    D,T = data.words.shape
    assert T == vocab.shape[1], "The number of terms in the document-term matrix (" + str(T) + ") differs from that in the model-states vocabulary parameter " + str(vocab.shape[1])
    docLens = np.squeeze(np.asarray(data.words.sum(axis=1)))

    outMeans = np.log(lda_topics(ldaQueryState))
    m = outMeans.mean()
    outMeans -= m
    outVarcs = np.abs(np.ones((D,K), dtype=dtype) * m/10)

    inMeans = outMeans + (m/10) * rd.random((D,K)).astype(dtype)
    inVarcs = outVarcs.copy()

    inDocCov  = np.ones((D,), dtype=dtype)

    model.topicMean[:]  = outMeans.mean(axis=0)
    model.topicCov[:,:] = np.cov(outMeans.T)

    return QueryState(outMeans, outVarcs, inMeans, inVarcs, inDocCov, docLens)


def newQueryState(data, modelState, withLdaTopics=None):
    '''
    Creates a new CTM Query state object. This contains all
    parameters and random variables tied to individual
    datapoints.
    
    Param:
    data - the dataset of words, features and links of which only words are used in this model
    modelState - the model state object
    withLdaQuery - if not null, this is used to instantiate the
    initial topics. IT IS ALSO USED TO MUTATE THE MODEL
    
    REturn:
    A CtmQueryState object
    '''
    if INIT_WITH_CTM:
        return _newQueryStateFromCtm(data, modelState)
    elif withLdaTopics is not None:
        return _newQueryStateFromLda(data, modelState, withLdaTopics)

    K, vocab, dtype =  modelState.K, modelState.vocab, modelState.dtype
    
    D,T = data.words.shape
    assert T == vocab.shape[1], "The number of terms in the document-term matrix (" + str(T) + ") differs from that in the model-states vocabulary parameter " + str(vocab.shape[1])
    docLens = np.squeeze(np.asarray(data.words.sum(axis=1)))
    
    outMeans = normalizerows_ip(rd.random((D,K)).astype(dtype))
    outVarcs = np.ones((D,K), dtype=dtype)

    inMeans = normalizerows_ip(outMeans + 0.1 * rd.random((D,K)).astype(dtype))
    inVarcs = np.ones((D,K), dtype=dtype)

    inDocCov  = np.ones((D,), dtype=dtype)
    
    return QueryState(outMeans, outVarcs, inMeans, inVarcs, inDocCov, docLens)


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
    outMeans, outVarcs, inMeans, inVarcs, inDocCov, docLens = queryState.outMeans, queryState.outVarcs, queryState.inMeans, queryState.inVarcs, queryState.inDocCov, queryState.docLens
    K, topicMean, topicCov, outDocCov, vocab, A, dtype = modelState.K, modelState.topicMean, modelState.topicCov, modelState.outDocCov, modelState.vocab, modelState.A, modelState.dtype

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

    inDocCov,  inDocPre  = np.ones((D,)), np.ones((D,))

    # Interestingly, outDocCov trades off good perplexity fits
    # with good ranking fits. > 10 gives better perplexity and
    # worse ranking. At 10 both are good. Below 10 both get
    # worse. Below 0.5, convergence stalls after the first iter.
    outDocCov, outDocPre = 10, 1./10

    # Iterate over parameters
    for itr in range(iterations):
        # We start with the M-Step, so the parameters are consistent with our
        # initialisation of the RVs when we do the E-Step

        # Update the mean and covariance of the prior over out-topics
        topicMean = outMeans.mean(axis=0)
        debugFn (itr, topicMean, "topicMean", data, K, topicMean, topicCov, outDocCov, inDocCov, vocab, dtype, outMeans, outVarcs, inMeans, inVarcs, A, docLens)

        outDiff = outMeans - topicMean[np.newaxis, :]
        inDiff =  inMeans - outMeans

        for _ in range(5): # It typically takes three iterations for the three dependant covariances -
                           # outDocCov, inDocCov and topicCov - to become consistent w.r.t each other
            topicCov  = (outDocPre * outDiff).T.dot(outDiff)
            topicCov += (inDocPre[:,np.newaxis] * inDiff).T.dot(inDiff)

            topicCov += np.diag(outVarcs.sum(axis=0))
            topicCov += np.diag(inVarcs.sum(axis=0))

            topicCov += IWISH_S_SCALE * np.eye(K)
            topicCov /= (2 * D + IWISH_DENOM)
            itopicCov = la.inv(topicCov)

            debugFn (itr, topicMean, "topicCov", data, K, topicMean, topicCov, outDocCov, inDocCov, vocab, dtype, outMeans, outVarcs, inMeans, inVarcs, A, docLens)

            diffSig   = inDiff.dot(itopicCov)
            diffSig  *= inDiff

            inDocCov  = diffSig.sum(axis=1)
            inDocCov += (outVarcs * np.diagonal(itopicCov)[np.newaxis, :]).sum(axis=1)
            inDocCov += (inVarcs  * np.diagonal(itopicCov)[np.newaxis, :]).sum(axis=1)
            inDocCov += IGAMMA_B
            inDocCov /= (IGAMMA_A - 1 + K)
            inDocPre  = np.reciprocal(inDocCov)

            debugFn (itr, inDocCov, "inDocCov", data, K, topicMean, topicCov, outDocCov, inDocCov, vocab, dtype, outMeans, outVarcs, inMeans, inVarcs, A, docLens)

            diffSig   = outDiff.dot(itopicCov)
            diffSig  *= outDiff
            # outDocCov = (IGAMMA_B + diffSig.sum() + (np.diagonal(itopicCov) * outVarcs).sum()) / (IGAMMA_A - 1 + (D * K))
            # outDocPre = 1./outDocCov

            debugFn (itr, outDocCov, "outDocCov", data, K, topicMean, topicCov, outDocCov, inDocCov, vocab, dtype, outMeans, outVarcs, inMeans, inVarcs, A, docLens)


        # Apply the exp function to get the (unnormalised) softmaxes in both directions.
        expMeansCol = np.exp(inMeans - inMeans.max(axis=0)[np.newaxis, :])
        lse_at_k = np.sum(expMeansCol, axis=0)
        F = 0.5 * inMeans \
          - (0.5/ D) * inMeans.sum(axis=0) \
          - expMeansCol / lse_at_k[np.newaxis, :]

        expMeansRow = np.exp(outMeans - outMeans.max(axis=1)[:, np.newaxis])
        W_weight   = sparseScalarQuotientOfDot(W, expMeansRow, vocab, out=W_weight)

        # Update the vocabularies

        vocab *= (W_weight.T.dot(expMeansRow)).T # Awkward order to maintain sparsity (R is sparse, expMeans is dense)
        vocab += VocabPrior
        vocab = normalizerows_ip(vocab)

        docVocab = (expMeansCol / lse_at_k[np.newaxis, :]).T.copy() # FIXME Dupes line in definition of F

        # Recalculate w_top_sums with the new vocab and log vocab improvement
        W_weight = sparseScalarQuotientOfDot(W, expMeansRow, vocab, out=W_weight)
        w_top_sums = W_weight.dot(vocab.T) * expMeansRow

        debugFn (itr, vocab, "vocab", data, K, topicMean, topicCov, outDocCov, inDocCov, vocab, dtype, outMeans, outVarcs, inMeans, inVarcs, A, docLens)

        # Now do likewise for the links, do it twice to model in-counts (first) and
        # out-counts (Second). The difference is the transpose
        LT_weight    = sparseScalarQuotientOfDot(LT, expMeansRow, docVocab, out=LT_weight)
        l_intop_sums = LT_weight.dot(docVocab.T) * expMeansRow
        in_counts    = l_intop_sums.sum(axis=0)

        L_weight     = sparseScalarQuotientOfDot(L, expMeansRow, docVocab, out=L_weight)
        l_outtop_sums = L_weight.dot(docVocab.T) * expMeansRow


        # Update the posterior variances
        outVarcs = np.reciprocal(emit_counts[:, np.newaxis] * (K-1)/(2*K) + (outDocPre + inDocPre[:,np.newaxis]) * np.diagonal(itopicCov)[np.newaxis,:])
        debugFn (itr, outVarcs, "outVarcs", data, K, topicMean, topicCov, outDocCov, inDocCov, vocab, dtype, outMeans, outVarcs, inMeans, inVarcs, A, docLens)

        inVarcs = np.reciprocal(in_counts[np.newaxis,:] * (D-1)/(2*D) + inDocPre[:,np.newaxis] * np.diagonal(itopicCov)[np.newaxis,:])
        debugFn (itr, inVarcs, "inVarcs", data, K, topicMean, topicCov, outDocCov, inDocCov, vocab, dtype, outMeans, outVarcs, inMeans, inVarcs, A, docLens)

        # Update the out-means and in-means
        out_rhs  = w_top_sums.copy()
        out_rhs += l_outtop_sums
        out_rhs += itopicCov.dot(topicMean) / outDocCov
        out_rhs += inMeans.dot(itopicCov) / inDocCov[:,np.newaxis]
        out_rhs += emit_counts[:, np.newaxis] * (outMeans.dot(A) - rowwise_softmax(outMeans))

        scaled_n_in = ((D-1.)/(2*D)) * ssp.diags(in_counts, 0)
        in_rhs = (inDocPre[:, np.newaxis] * outMeans).dot(itopicCov)
        in_rhs += ((-inMeans.sum(axis=0) * in_counts) / (4*D))[np.newaxis,:]
        in_rhs += l_intop_sums
        in_rhs += in_counts[np.newaxis, :] * F
        for d in range(D):
            in_rhs[d, :]  += in_counts * inMeans[d, :] / (4*D)
            inMeans[d, :]  = la.inv(inDocPre[d] * itopicCov + scaled_n_in).dot(in_rhs[d, :])
            in_rhs[d,:]   -= in_counts * inMeans[d, :] / (4*D)

            try:
                outCov          = la.inv((outDocPre + inDocPre[d]) * itopicCov + emit_counts[d] * A)
                outMeans[d, :]  = outCov.dot(out_rhs[d,:])
            except la.LinAlgError as err:
                print ("ABORTING: " + str(err))
                return \
                    ModelState(K, topicMean, topicCov, outDocCov, vocab, A, True, dtype, MODEL_NAME), \
                    QueryState(outMeans, outVarcs, inMeans, inVarcs, inDocCov, docLens), \
                    (np.array(boundIters), np.array(boundValues), np.array(likelyValues))


        debugFn (itr, outMeans, "inMeans/outMeans", data, K, topicMean, topicCov, outDocCov, inDocCov, vocab, dtype, outMeans, outVarcs, inMeans, inVarcs, A, docLens)
        # debugFn (itr, inMeans,  "inMeans",  data, K, topicMean, topicCov, outDocCov, inDocCov, vocab, dtype, outMeans, outVarcs, inMeans, inVarcs, A, docLens)

        if logFrequency > 0 and itr % logFrequency == 0:
            modelState = ModelState(K, topicMean, topicCov, outDocCov, vocab, A, True, dtype, MODEL_NAME)
            queryState = QueryState(outMeans, outVarcs, inMeans, inVarcs, inDocCov, docLens)

            boundValues.append(var_bound(data, modelState, queryState))
            likelyValues.append(log_likelihood(data, modelState, queryState))
            boundIters.append(itr)

            print (time.strftime('%X') + " : Iteration %d: bound %f \t Perplexity: %.2f" % (itr, boundValues[-1], perplexity_from_like(likelyValues[-1], docLens.sum())))
            if len(boundValues) > 1:
                if boundValues[-2] > boundValues[-1]:
                    printStderr ("ERROR: bound degradation: %f > %f" % (boundValues[-2], boundValues[-1]))

                # Check to see if the improvement in the bound has fallen below the threshold
                if itr > MinItersBeforeEarlyStop and abs(perplexity_from_like(likelyValues[-1], docLens.sum()) - perplexity_from_like(likelyValues[-2], docLens.sum())) < 1.0:
                    break

        # if True or debug or itr % logFrequency == 0:
        #     print("   Sigma     %6.1f  \t %9.3g, %9.3g, %9.3g" % (np.log(la.det(topicCov)), topicCov.min(), topicCov.mean(), topicCov.max()), end="  |")
        #     print("   rho       %6.1f  \t %9.3g, %9.3g, %9.3g" % (sum(log(inDocCov[d]) for d in range(D)), inDocCov.min(), inDocCov.mean(), inDocCov.max()), end="  |")
        #     print("   alpha     %6.1f  \t %9.3g" % (np.log(la.det(np.eye(K,) * outDocCov)), outDocCov), end="  |")
        #     print("   inMeans   %9.3g, %9.3g, %9.3g" % (inMeans.min(),  inMeans.mean(),  inMeans.max()), end="  |")
        #     print("   outMeans  %9.3g, %9.3g, %9.3g" % (outMeans.min(), outMeans.mean(), outMeans.max()), end="  |")
        #     print("   inVarcs   %6.1f  \t %9.3g, %9.3g, %9.3g" % (sum(safe_log_det(np.diag(inVarcs[d]))  for d in range(D)) / D, inVarcs.min(),  inVarcs.mean(),  inVarcs.max()), end="  |")
        #     print("   outVarcs  %6.1f  \t %9.3g, %9.3g, %9.3g" % (sum(safe_log_det(np.diag(outVarcs[d])) for d in range(D)) / D, outVarcs.min(), outVarcs.mean(), outVarcs.max()))

    return \
        ModelState(K, topicMean, topicCov, outDocCov, vocab, A, True, dtype, MODEL_NAME), \
        QueryState(outMeans, outVarcs, inMeans, inVarcs, inDocCov, docLens), \
        (np.array(boundIters), np.array(boundValues), np.array(likelyValues))


def query(data, modelState, queryState, queryPlan):
    '''
    Given a _trained_ model, attempts to predict the topics for each of
    the inputs. The assumption is that there are no out-links associated
    with the documents, and that no documents in the training set link
    to any of these documents in the query set.

    The word and link vocabularies are kept fixed. Due to the assumption
    of no in-links, we don't learn the prior in-document covariance, nor
    the posterior distribution over in-links. Also, we don't modify

    
    Params:
    data - the dataset of words, features and links of which only words are used in this model
    modelState - the _trained_ model
    queryState - the query state generated for the query dataset
    queryPlan  - used in this case as we need to tighten up the approx
    
    Returns:
    The model state and query state, in that order. The model state is
    unchanged, the query is.
    '''
    W, L, LT, X = data.words, data.links, ssp.csr_matrix(data.links.T), data.feats
    D,_ = W.shape
    out_links = np.squeeze(np.asarray(data.links.sum(axis=1)))

    # Book-keeping for logs
    boundIters, boundValues, likelyValues = [], [], []

    # Unpack the the structs, for ease of access and efficiency
    iterations, epsilon, logFrequency, diagonalPriorCov, debug = queryPlan.iterations, queryPlan.epsilon, queryPlan.logFrequency, queryPlan.fastButInaccurate, queryPlan.debug
    outMeans, outVarcs, inMeans, inVarcs, inDocCov, docLens = queryState.outMeans, queryState.outVarcs, queryState.inMeans, queryState.inVarcs, queryState.inDocCov, queryState.docLens
    K, topicMean, topicCov, outDocCov, vocab, A, dtype = modelState.K, modelState.topicMean, modelState.topicCov, modelState.outDocCov, modelState.vocab, modelState.A, modelState.dtype

    emit_counts = docLens + out_links

    # Initialize some working variables
    W_weight  = W.copy()

    outDocPre = 1./outDocCov
    inDocPre  = np.reciprocal(inDocCov)
    itopicCov = la.inv(topicCov)

    # Iterate over parameters
    for itr in range(iterations):
        # We start with the M-Step, so the parameters are consistent with our
        # initialisation of the RVs when we do the E-Step

        expMeansRow = np.exp(outMeans - outMeans.max(axis=1)[:, np.newaxis])
        W_weight   = sparseScalarQuotientOfDot(W, expMeansRow, vocab, out=W_weight)
        w_top_sums = W_weight.dot(vocab.T) * expMeansRow

        # Update the posterior variances
        outVarcs = np.reciprocal(emit_counts[:, np.newaxis] * (K-1)/(2*K) + (outDocPre + inDocPre[:,np.newaxis]) * np.diagonal(itopicCov)[np.newaxis,:])

        # Update the out-means and in-means
        out_rhs  = w_top_sums.copy()
        # No link outputs to model.
        out_rhs += itopicCov.dot(topicMean) / outDocCov
        out_rhs += emit_counts[:, np.newaxis] * (outMeans.dot(A) - rowwise_softmax(outMeans))

        for d in range(D):
            outCov          = la.inv(outDocPre * itopicCov + emit_counts[d] * A)
            outMeans[d, :]  = outCov.dot(out_rhs[d,:])

        if logFrequency > 0 and itr % logFrequency == 0:
            modelState = ModelState(K, topicMean, topicCov, outDocCov, vocab, A, True, dtype, MODEL_NAME)
            queryState = QueryState(outMeans, outVarcs, inMeans, inVarcs, inDocCov, docLens)

            boundValues.append(0)
            likelyValues.append(log_likelihood(data, modelState, queryState))
            boundIters.append(itr)

            print (time.strftime('%X') + " : Iteration %d: bound %f \t Perplexity: %.2f" % (itr, boundValues[-1], perplexity_from_like(likelyValues[-1], docLens.sum())))
            if len(boundValues) > 1:
                # Check to see if the improvement in the bound has fallen below the threshold
                if itr > MinItersBeforeEarlyStop and abs(perplexity_from_like(likelyValues[-1], docLens.sum()) - perplexity_from_like(likelyValues[-2], docLens.sum())) < 1.0:
                    break

    return \
        ModelState(K, topicMean, topicCov, outDocCov, vocab, A, True, dtype, MODEL_NAME), \
        QueryState(outMeans, outVarcs, inMeans, inVarcs, inDocCov, docLens)


def log_likelihood (data, modelState, queryState):
    ''' 
    Return the log-likelihood of the given data W according to the model
    and the parameters inferred for the entries in W stored in the 
    queryState object.
    '''
    probs = rowwise_softmax(queryState.outMeans)
    doc_dist = colwise_softmax(queryState.inMeans)

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
    outMeans, outVarcs, inMeans, inVarcs, inDocCov, docLens = queryState.outMeans, queryState.outVarcs, queryState.inMeans, queryState.inVarcs, queryState.inDocCov, queryState.docLens
    K, topicMean, topicCov, outDocCov, vocab, A, dtype = modelState.K, modelState.topicMean, modelState.topicCov, modelState.outDocCov, modelState.vocab, modelState.A, modelState.dtype

    # Calculate some implicit  variables
    itopicCov = la.inv(topicCov)
    
    bound = 0

    expMeansOut = np.exp(outMeans - outMeans.max(axis=1)[:, np.newaxis])
    expMeansIn  = np.exp(inMeans - inMeans.max(axis=0)[np.newaxis, :])
    lse_at_k    = expMeansIn.sum(axis=0)

    # Distribution over document topics
    bound -= (D*K)/2. * LN_OF_2_PI
    bound -= D/2. * safe_log_det(outDocCov * topicCov)
    diff   = outMeans - topicMean[np.newaxis,:]
    bound -= 0.5 * np.sum (diff.dot(itopicCov) * diff * 1./outDocCov)
    bound -= (0.5 / outDocCov) * np.sum(outVarcs * np.diag(itopicCov)[np.newaxis,:]) # = -0.5 * sum_d tr(V_d \Sigma^{-1}) when V_d is diagonal only.

    # And its entropy
    bound += 0.5 * D * K * LN_OF_2_PI_E + 0.5 * np.log(outVarcs).sum()

    # Distribution over document in-links
    inDocPre = np.reciprocal(inDocCov)
    bound -= (D*K)/2. * LN_OF_2_PI
    bound -= D/2. * safe_log_det(topicCov)
    bound -= K/2 * safe_log(inDocCov).sum()
    diff   = inMeans - outMeans
    bound -= 0.5 * np.sum (diff.dot(itopicCov) * diff * inDocPre[:,np.newaxis])
    bound -= 0.5 * np.sum((inVarcs * inDocPre[:,np.newaxis]) * np.diag(itopicCov)[np.newaxis,:]) # = -0.5 * sum_d tr(V_d \Sigma^{-1}) when V_d is diagonal only.

    # And its entropy
    bound += 0.5 * D * K * LN_OF_2_PI_E + 0.5 * np.log(inVarcs).sum()

    # Distribution over topic assignments E[p(Z)] and E[p(Y)]
    W_weights  = sparseScalarQuotientOfDot(W, expMeansOut, vocab)  # D x V   [W / TB] is the quotient of the original over the reconstructed doc-term matrix
    top_sums   = expMeansOut * (W_weights.dot(vocab.T)) # D x K

    L_weights  = sparseScalarQuotientOfNormedDot(L, expMeansOut, expMeansIn, lse_at_k)
    top_sums  += expMeansOut * (L_weights.dot(expMeansIn) / lse_at_k[np.newaxis, :])

    # E[p(Z,Y)]
    linkLens = np.squeeze(np.array(L.sum(axis=1)))
    bound += np.sum(outMeans * top_sums)
    bound -= np.sum((docLens + linkLens) * np.log(np.sum(expMeansOut, axis=1)))

    # H[Z]
    bound += ((W_weights.dot(vocab.T)) * expMeansOut * outMeans).sum() \
           + ((W_weights.dot((np.log(vocab) * vocab).T)) * expMeansOut).sum() \
           - np.trace(sparseScalarProductOfSafeLnDot(W_weights, expMeansOut, vocab).dot(vocab.T).dot(expMeansOut.T))

    # H[Y]
    docVocab = (expMeansIn / lse_at_k[np.newaxis,:]).T.copy()
    bound += ((L_weights.dot(docVocab.T)) * expMeansOut * outMeans).sum() \
           + ((L_weights.dot((np.log(docVocab) * docVocab).T)) * expMeansOut).sum() \
           - np.trace(sparseScalarProductOfSafeLnDot(L_weights, expMeansOut, docVocab).dot(docVocab.T).dot(expMeansOut.T))

    # E[p(W)]
    vlv = np.log(vocab) * vocab
    bound += np.trace(expMeansOut.T.dot(W_weights.dot(vlv.T)))

    # E[p(L)
    dld = np.log(docVocab) * docVocab
    bound += np.trace(expMeansOut.T.dot(L_weights.dot(dld.T)))

    return bound


def softmax(x):
    r  = x.copy()
    r -= r.max()
    np.exp(r, out = r)
    r /= np.sum(r)
    return r


def min_link_probs(model, train_tops, query_tops, links, docSubset=None):
    '''
    For every document, for each of the given links, determine the
    probability of the least likely link (i.e the document-specific
    minimum of probabilities).

    :param model: the model object
    :param train_tops: the representations of the link-target documents
    :param query_tops: the representations of the link-origin documents
    :param links: a DxD matrix of links for each document (row)
    :param docSubset: a list of documents to consider for evaluation. If
    none all documents are considered.
    :return: a vector with the minimum out-link probabilities for each
        document in the subset
    '''
    if docSubset is None:
        docSubset = [q for q in range(query_tops.outMeans.shape[0])]
    Q = len(docSubset)

    col_maxes = train_tops.inMeans.max(axis=0)
    lse_at_k = np.sum(np.exp(train_tops.inMeans - col_maxes), axis=0)
    mins = np.empty((Q,), dtype=model.dtype)

    outRow = -1
    for d in docSubset:
        outRow += 1
        srcTopDist = softmax(query_tops.outMeans[d, :])
        probs = []

        for i in range(len(links[d,:].indices)):
            dst = links[d,:].indices[i]
            linkDist  = np.exp(train_tops.inMeans[dst, :] - col_maxes) / lse_at_k
            probs.append(np.dot(srcTopDist, linkDist))
        mins[outRow] = min(probs) if len(probs) > 0 else -1

    return mins


def link_probs(model, train_tops, query_tops, min_link_probs, docSubset=None):
    '''
    Generate the probability of a link for all possible pairs of documents,
    but only store those probabilities that are bigger than or equal to the
    minimum. This ensures, hopefully, that we don't materialise a complete
    DxD matrix, but rather the minimum needed to determine the mean
    average precisions

    :param model: the trained model
    :param train_tops: the representations of the link-target documents
    :param query_tops: the representations of the link-origin documents
    :param min_link_probs: the minimum link probability for each document
    in the subset
    :param docSubset: a list of documents to consider for evaluation. If
    none all documents are considered.
    :return: a (hopefully) sparse len(docSubset)xD matrix of link probabilities
    '''
    # We build the result up as a COO matrix
    rows = []
    cols = []
    vals = []

    # Determine the size of the output
    D = train_tops.outMeans.shape[0]
    if docSubset is None:
        docSubset = [q for q in range(query_tops.outMeans.shape[0])]
    Q = len(docSubset)

    # Calculate the softmax transform parameters
    dstTopDists = colwise_softmax(train_tops.inMeans)

    # Infer the link probabilities
    outRow = -1
    for src in docSubset:
        outRow += 1

        srcTopDist = softmax(query_tops.outMeans[src, :])
        probs      = dstTopDists.dot(srcTopDist)
        relevant   = np.where(probs >= min_link_probs[outRow] - 1E-9)[0]

        rows.extend([outRow] * len(relevant))
        cols.extend(relevant)
        vals.extend(probs[relevant])

    # Build the COO matrix, then covert it to CSR. Converts lists to numpy
    # arrays to ensure appropriate dtypes
    r = np.array(rows, dtype=np.int32)
    c = np.array(cols, dtype=np.int32)
    v = np.array(vals, dtype=model.dtype)

    return ssp.coo_matrix((v, (r, c)), shape=(Q, D)).tocsr()

# ==============================================================
# PUBLIC HELPERS
# ==============================================================


@static_var("old_bound", 0)
def _debug_with_bound (itr, var_value, var_name, data, K, topicMean, topicCov, outDocCov, inDocCov, vocab, dtype, outMeans, outVarcs, inMeans, inVarcs, A, n):
    if np.isnan(var_value).any():
        printStderr ("WARNING: " + var_name + " contains NaNs")
    if np.isinf(var_value).any():
        printStderr ("WARNING: " + var_name + " contains INFs")
    if "dtype" in dir(var_value) and var_value.dtype != dtype:
        printStderr ("WARNING: dtype(" + var_name + ") = " + str(var_value.dtype))

    model = ModelState(K, topicMean, topicCov, outDocCov, vocab, A, False, dtype, MODEL_NAME)
    query = QueryState(outMeans, outVarcs, inMeans, inVarcs, inDocCov, n)

    old_bound = _debug_with_bound.old_bound
    bound     = var_bound(data, model, query)
    diff = "" if old_bound == 0 else "%15.4f" % (bound - old_bound)
    _debug_with_bound.old_bound = bound
    
    addendum = ""
    if var_name == "topicCov":
        try:
            addendum = "log det(topicCov) = %g" % (np.log(la.det(topicCov)))
        except:
            addendum = "log det(topicCov) = <undefined>"
    
    if isnan(bound):
        printStderr ("Bound is NaN")
    else:
        perp = perplexity_from_like(log_likelihood(data, model, query), data.word_count)
        if int(bound - old_bound) < 0:
            printStderr ("Iter %3d Update %-15s Bound %22f (%15s) (%5.0f)     %s" % (itr, var_name, bound, diff, perp, addendum))
        else:
            print ("Iter %3d Update %-15s Bound %22f (%15s) (%5.0f)  %s" % (itr, var_name, bound, diff, perp, addendum))

def _debug_with_nothing (itr, var_value, var_name, data, K, topicMean, topicCov, outDocCov, inDocCov, vocab, dtype, outMeans, outVarcs, inMeans, inVarcs, A, n):
    pass