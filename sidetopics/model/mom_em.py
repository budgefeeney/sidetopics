'''
Mixture of Multinomials implement using a MAP EM algorithm.

@author: bryanfeeney
'''
from numbers import Number

import numpy as np
import numpy.random as rd
import scipy.special as fns
from typing import Sized, Tuple
import logging


from sidetopics.util.misc import constantArray

from sidetopics.model.evals import perplexity_from_like
from sidetopics.util.sparse_elementwise import sparseScalarProductOfSafeLnDot, sparseScalarProductOfDot
from sidetopics.util.overflow_safe import safe_log
from sidetopics.model.common import DataSet

from collections import namedtuple

CHANGE_TOLERANCE = 0.001

MODEL_NAME = "mom/em"
DTYPE = np.float64
VOCAB_PRIOR = 1.1

# After how many training iterations should we stop to update the hyperparameters
HyperParamUpdateInterval = 5
HyperUpdateEnabled = False

TrainPlan = namedtuple( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')

QueryState = namedtuple( \
    'QueryState', \
    'docLens topicDists processed' \
    )

ModelState = namedtuple( \
    'ModelState', \
    'K topicPrior vocabPrior wordDistParam corpusTopicDistParam processed dtype name'
)


def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState( \
        model.K, \
        model.topicPrior, \
        model.vocabPrior, \
        None if model.wordDistParam is None else model.wordDistParam.copy(),
        None if model.corpusTopicDistParam is None else model.corpusTopicDistParam.copy(),
        model.processed, \
        model.dtype, \
        model.name)


def newModelAtRandom(data, K, topicPrior=None, vocabPrior=VOCAB_PRIOR, dtype=DTYPE):
    '''
    Creates a new LDA ModelState for the given training set and
    the given number of topics. Everything is instantiated purely
    at random. This contains all parameters independent of of
    the dataset (e.g. learnt priors)

    Param:
    W - the DxT document-term matrix of T terms in D documents
        which will be used for training.
    X - the DxD matrix of document-document links
    K - the number of topics
    topicPrior - the prior over topics, either a scalar or a K-dimensional vector
    vocabPrior - the prior over vocabs, either a scalar or a T-dimensional vector
    dtype      - the datatype to be used throughout.

    Return:
    A ModelState object
    '''
    assert K > 1, "There must be at least two topics"
    assert K < 255, "There can be no more than 255 topics"
    T = data.words.shape[1]

    if topicPrior is None:
        topicPrior = constantArray((K,), 5.0 / K + 0.5, dtype)  # From Griffiths and Steyvers 2004
    elif type(topicPrior) is float:
        topicPrior = constantArray((K,), topicPrior, dtype)
    if vocabPrior is None:
        vocabPrior = VOCAB_PRIOR
    if issubclass(type(vocabPrior), Number):
        fill_value = vocabPrior
        vocabPrior = np.ndarray((T,), dtype=dtype)
        vocabPrior[:] = fill_value

    wordDistParam = np.ones((K, T), dtype=dtype)
    for k in range(K):
        docLenSum = 0
        while docLenSum < 1000:
            randomDoc = rd.randint(0, data.doc_count, size=1)
            sample_doc = data.words[randomDoc, :]
            wordDistParam[k, sample_doc.indices] += sample_doc.data
            docLenSum += sample_doc.sum()
        # wordDists[k, :] /= wordDists[k, :].sum()

    corpusTopicDistParam = topicPrior.copy()

    return ModelState(K, topicPrior, vocabPrior, wordDistParam, corpusTopicDistParam, False, dtype, MODEL_NAME)


def newQueryState(data, modelState, debug):
    '''
    Creates a new LDA QueryState object. This contains all
    parameters and random variables tied to individual
    datapoints.

    Param:
    W - the DxT document-term matrix used for training or
        querying.
    modelState - the model state object

    Return:
    A QueryState object
    '''
    if debug:
        print("Ignoring setting of debug to True")

    docLens = np.squeeze(np.asarray(data.words.sum(axis=1)))

    # Initialise the per-token assignments at random according to the dirichlet hyper
    # This is super-slow
    dist = modelState.topicPrior.copy()
    dist /= dist.sum()

    topicDists = rd.dirichlet(dist, size=data.doc_count).astype(modelState.dtype)

    return QueryState(docLens, topicDists, False)


def newTrainPlan(iterations=100, epsilon=2, logFrequency=10, fastButInaccurate=False, debug=False):
    '''
    Create a training plan determining how many iterations we
    process, how often we plot the results, how often we log
    the variational bound, etc.

    epsilon is oddly measured, we just evaluate the angle of the line segment between
    the last value of the bound and the current, and if it's less than the given angle,
    then stop.
    '''
    return TrainPlan(iterations, epsilon, logFrequency, fastButInaccurate, debug)


def corpusTopicDistDirichletParam(model: ModelState) -> np.ndarray:
    return model.corpusTopicDistParam


def corpusTopicDist(model: ModelState) -> np.ndarray:
    return model.corpusTopicDistParam / model.corpusTopicDistParam.sum()

def wordDistsDirichletParam(modelState):
    return modelState.wordDistParam

def wordDists(modelState):
    '''
    The K x T matrix of  word distributions inferred for the K topics
    '''
    result = modelState.wordDistParam.copy()
    norm = result.sum(axis=1)
    result /= norm[:, np.newaxis]

    return result


def topicDists(queryState: QueryState) -> np.ndarray:
    '''
    The D x K matrix of distributions over K topics for each of the
    D documents
    '''
    return queryState.topicDists


def doc_count(query: QueryState) -> int:
    return len(query.docLens)


def train(data, model: ModelState, query: QueryState, plan: TrainPlan, updateVocab=True):
    """
    Infers the topic distributions in general, and specifically for
    each individual datapoint,

    Params:
    data - the training data, we just use the DxT document-term matrix
    model - the initial model configuration. This is MUTATED IN-PLACE
    query - the query results - essentially all the "local" variables
            matched to the given observations. Also MUTATED IN-PLACE
    plan  - how to execute the training process (e.g. iterations,
            log-interval etc.)

    Return:
    The updated model object (note parameters are updated in place, so make a
    defensive copy if you want it)
    The query object with the update query parameters
    """
    iterations, epsilon, logFrequency, fastButInaccurate, debug = \
        plan.iterations, plan.epsilon, plan.logFrequency, plan.fastButInaccurate, plan.debug
    docLens = query.docLens
    topicDist = topicDists(query)
    K, topicPrior, vocabPrior, wordDistParam, corpusTopicDistParam, dtype = \
        model.K, model.topicPrior, model.vocabPrior, wordDistsDirichletParam(model), corpusTopicDistDirichletParam(model), model.dtype

    W = data.words

    iters, bnds, likes = [], [], []

    # Quick sanity check
    if np.any(docLens < 1):
        raise ValueError("Input document-term matrix contains at least one document with no words")
    assert dtype == np.float64, "Only implemented for 64-bit floats"

    lnCorpusTopicDist = fns.digamma(corpusTopicDistParam) - fns.digamma(corpusTopicDistParam.sum())
    lnWordDist = fns.digamma(wordDistParam) - fns.digamma(wordDistParam.sum(axis=1))[:, np.newaxis]
    oldTopicDist = np.ndarray(topicDist.shape, dtype=topicDist.dtype)

    for itr in range(iterations):
        oldTopicDist[:, :] = topicDist[:, :]

        topicDist[:, :] = (data.words @ lnWordDist.T)
        topicDist[:, :] += lnCorpusTopicDist[np.newaxis, :]
        topicDist -= topicDist.max(axis=1)[:, np.newaxis]
        np.exp(topicDist, out=topicDist)
        topicDist /= topicDist.sum(axis=1)[:, np.newaxis]

        if np.abs(oldTopicDist - topicDist).sum() < CHANGE_TOLERANCE * topicDist.shape[0] * topicDist.shape[1]:
            logging.info(f"Stopping train after {itr + 1} iterations as change in topic distibution is minimal")
            break

        corpusTopicDistParam = topicDist.sum(axis=0) + model.topicPrior
        fns.digamma(corpusTopicDistParam, out=lnCorpusTopicDist)
        lnCorpusTopicDist -= fns.digamma(corpusTopicDistParam.sum())

        # Derive new parameter estimates
        wordDistParam = (data.words.T @ topicDist).T \
                      + model.vocabPrior[np.newaxis, :]
        fns.digamma(wordDistParam, out=lnWordDist)
        lnWordDist -= fns.digamma(wordDistParam.sum(axis=1))[:, np.newaxis]

        if debug or (logFrequency > 0 and itr % logFrequency == 0):
            m = ModelState(K, topicPrior, vocabPrior, wordDistParam, corpusTopicDistParam, True, dtype, model.name)
            q = QueryState(query.docLens, topicDist, True)

            iters.append(itr)
            bnds.append(var_bound(data, m, q))
            likes.append(log_likelihood_point(data, m, q))

            perp = perplexity_from_like(likes[-1], W.sum())
            print("Iteration %d : Train Perp = %4.0f  Bound = %.3f" % (itr, perp, bnds[-1]))

            if len(iters) > 2 and iters[-1] > 50:
                lastPerp = perplexity_from_like(likes[-2], W.sum())
                if lastPerp - perp < 1:
                    break

    return ModelState(K, topicPrior, vocabPrior, wordDistParam, corpusTopicDistParam, True, dtype, model.name), \
           QueryState(query.docLens, topicDist, True), \
           (np.array(iters, dtype=np.int32), np.array(bnds), np.array(likes))


def query(data: DataSet, model: ModelState, queryState: QueryState, plan: TrainPlan) -> Tuple[ModelState, QueryState]:
    '''
    Infers the topic distributions in general, and specifically for
    each individual datapoint, without altering the model

    Params:
    W - the DxT document-term matrix
    X - The DxD document-document matrix
    model - the initial model configuration. This is MUTATED IN-PLACE
    qyery - the query results - essentially all the "local" variables
            matched to the given observations. Also MUTATED IN-PLACE
    plan  - how to execute the training process (e.g. iterations,
            log-interval etc.)

    Return:
    The updated model object (note parameters are updated in place, so make a
    defensive copy if you want it)
    The query object with the update query parameters
    '''
    lnCorpusTopicDist = fns.digamma(corpusTopicDistDirichletParam(model))
    lnCorpusTopicDist -= fns.digamma(corpusTopicDistDirichletParam(model).sum())

    lnWordDist = fns.digamma(wordDistsDirichletParam(model))
    lnWordDist -= fns.digamma(wordDistsDirichletParam(model).sum(axis=1))[:, np.newaxis]

    topicDist = topicDists(queryState)
    oldTopicDist = np.ndarray(topicDist.shape, dtype=topicDist.dtype)
    for itr in range(plan.iterations):
        oldTopicDist[:, :] = topicDist
        topicDist = (data.words @ lnWordDist.T) + lnCorpusTopicDist[np.newaxis, :]
        topicDist -= topicDist.max(axis=1)[:, np.newaxis]
        np.exp(topicDist, out=topicDist)
        topicDist /= topicDist.sum(axis=1)[:, np.newaxis]

        delta = np.abs(topicDist - oldTopicDist).sum()
        if delta < topicDist.shape[0] * topicDist.shape[1] * CHANGE_TOLERANCE:
            logging.info(f"Stopping query after {itr + 1} iterations as change in topic distibution is minimal")
            break

    logging.info(f"Returning query results after {itr + 1} iterations")
    return model, QueryState(queryState.docLens, topicDist, True)


def log_likelihood_expected(data: DataSet, model: ModelState, query: QueryState) -> float:
    lls = np.zeros(shape=(len(data), model.K), dtype=model.dtype)

    # p(z=k | alpha)
    if query is None:  # use prior, the standard likelihood formula
        logging.info("Returning likelihood based on prior over topic assignments")
        alpha = np.ndarray(shape=(len(data), model.K), dtype=model.dtype)
        alpha[:, :] = corpusTopicDistDirichletParam(model)[np.newaxis, :]
        alpha_sum = np.ndarray(shape=(len(data),), dtype=model.dtype)
        alpha_sum[:] = corpusTopicDistDirichletParam(model).sum()
    else:  # Use a different distribution, e.g. the posterior from a doc-completion split
        logging.info("Returning likelihood based on give (e.g. posterior) assignments")
        alpha = topicDistsDirichletParam(query)  # DxK
        alpha_sum = alpha.sum(axis=1)            # Dx1

    lls = np.zeros(shape=(len(data), model.K), dtype=model.dtype)
    lls += fns.loggamma(alpha_sum)[:, np.newaxis]
    lls -= fns.loggamma(alpha_sum + 1)[:, np.newaxis]
    # FIXME this memory explosion needs to be contained
    lls += np.sum(fns.loggamma(np.eye(model.K)[np.newaxis, :, :] + alpha[:, :, np.newaxis]), axis=1)
    lls -= np.sum(fns.loggamma(alpha), axis=1)[:, np.newaxis]

    # # Try at this point using the point estimate of the prior and see what happens to isolate.
    # if query is not None:
    #     lls += np.log(query.topicDists)
    # else:
    #     lls += np.log(corpusTopicDist(model))[np.newaxis, :]

    # p(X = x|z=k, beta)
    doc_lens = np.squeeze(np.array(data.words.sum(axis=1)))

    beta = wordDistsDirichletParam(model)  # K x T
    beta_sum = beta.sum(axis=1)  # K x 1

    lls += fns.loggamma(beta_sum)[np.newaxis, :]
    lls -= fns.loggamma(doc_lens[:, np.newaxis] + beta_sum[np.newaxis, :])
    for k in range(model.K):
        lls[:, k] += np.squeeze(np.array(
            fns.loggamma(data.words + (beta[k, :])[np.newaxis, :]).sum(axis=1)
        ))
    lls -= np.sum(fns.loggamma(beta), axis=1)[np.newaxis, :]

    max_lls = lls.max(axis=1)
    lls -= max_lls[:, np.newaxis]
    np.exp(lls, out=lls)

    lls = max_lls + np.log(lls.sum(axis=1))

    return lls.sum()


def log_likelihood_point(data: DataSet, model: ModelState, query: QueryState = None) -> float:
    wordDist = wordDists((model))

    # ln p(x|topic=k, word_dist) + ln p(topic=k) for all documents, for all k
    lls = data.words @ np.log(wordDist.T)
    if query is not None:
        topicDist = topicDists(query)
        lls += safe_log(topicDist)
    else:
        lls += safe_log(corpusTopicDist(model))[np.newaxis, :]

    # Safe Log-sum-exp (of topic-specific log likelihoods)
    max_lls = lls.max(axis=1)
    lls -= max_lls[:, np.newaxis]
    np.exp(lls, out=lls)

    lls = max_lls + np.log(lls.sum(axis=1))

    # Return corpus-total log likelihood
    return lls.sum()



def var_bound(data, model, query, topicDistOverride=None):
    '''
    Determines the variational bounds.
    '''
    bound = 0

    # Unpack the the structs, for ease of access and efficiency
    docLens, topicMeans = \
        query.docLens, query.topicDists
    K, topicPrior, vocabPrior, wordDists, corpusTopicDist, dtype = \
        model.K, model.topicPrior, model.vocabPrior, model.wordDists, model.corpusTopicDist, model.dtype

    tops = topicDistOverride \
        if topicDistOverride is not None \
        else topicDists(query)

    # Initialize z matrix if necessary
    W = data.words
    D, T = W.shape

    wordLikely = sparseScalarProductOfSafeLnDot(data.words, tops, wordDists(model)).sum()
    topicLikely = topicMeans.dot(fns.digamma(corpusTopicDist) - fns.digamma(corpusTopicDist.sum()))


    # Expected joint
    like = W.dot(safe_log(wordDists).T) # D*K
    like += corpusTopicDist[np.newaxis,:]
    like *= safe_log(topicMeans)

    # Entropy
    ent = (-topicMeans * safe_log(topicMeans)).sum()

    return like.sum() + ent





