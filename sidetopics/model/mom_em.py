'''
Mixture of Multinomials implement using a MAP EM algorithm.

@author: bryanfeeney
'''
import numpy as np
import numpy.random as rd
import scipy.misc as fns


from sidetopics.util.misc import constantArray

from sidetopics.model.evals import perplexity_from_like
from sidetopics.util.sparse_elementwise import sparseScalarProductOfSafeLnDot
from sidetopics.util.overflow_safe import safe_log

from collections import namedtuple

MODEL_NAME = "mom/em"
DTYPE = np.float64
VocabPrior = 1.1

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
    'K topicPrior vocabPrior wordDists corpusTopicDist processed dtype name'
)


def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState( \
        model.K, \
        model.topicPrior, \
        model.vocabPrior, \
        None if model.wordDists is None else model.wordDists.copy(),
        None if model.corpusTopicDist is None else model.corpusTopicDist.copy(),
        model.processed, \
        model.dtype, \
        model.name)


def newModelAtRandom(data, K, topicPrior=None, vocabPrior=VocabPrior, dtype=DTYPE):
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
        topicPrior = constantArray((K,), topicPrior, dtype)  # From Griffiths and Steyvers 2004
    if vocabPrior is None:
        vocabPrior = VocabPrior

    wordDists = np.ones((K, T), dtype=dtype)
    for k in range(K):
        docLenSum = 0
        while docLenSum < 1000:
            randomDoc = rd.randint(0, data.doc_count, size=1)
            sample_doc = data.words[randomDoc, :]
            wordDists[k, sample_doc.indices] += sample_doc.data
            docLenSum += sample_doc.sum()
        wordDists[k, :] /= wordDists[k, :].sum()

    corpusTopicDist = np.array([1./K] * K, dtype=dtype)

    return ModelState(K, topicPrior, vocabPrior, wordDists, corpusTopicDist, False, dtype, MODEL_NAME)


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
    topicDists *= docLens[:, np.newaxis]
    topicDists += modelState.topicPrior[np.newaxis, :]

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


def wordDists(modelState):
    '''
    The K x T matrix of  word distributions inferred for the K topics
    '''
    result = modelState.wordDists.copy()
    norm = result.sum(axis=1)
    result /= norm[:, np.newaxis]

    return result


def topicDists(queryState):
    '''
    The D x T matrix of distributions over K topics for each of the
    D documents
    '''
    return queryState.topicDists


# TODO Verify this is correct
def log_likelihood(data, model, query, topicDistOverride=None):
    '''
    Return the log-likelihood of the given data according to the model
    and the parameters inferred for datapoints in the query-state object

    Actually returns a vector of D document specific log likelihoods
    '''
    tops = topicDistOverride \
        if topicDistOverride is not None \
        else topicDists(query)
    wordLikely = sparseScalarProductOfSafeLnDot(data.words, tops, wordDists(model)).sum()
    return wordLikely
    # return (data.words.dot(wordDists(model).T) * model.corpusTopicDist).sum()


def train(data, model, query, plan, updateVocab=True):
    '''
    Infers the topic distributions in general, and specifically for
    each individual datapoint,

    Params:
    data - the training data, we just use the DxT document-term matrix
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
    iterations, epsilon, logFrequency, fastButInaccurate, debug = \
        plan.iterations, plan.epsilon, plan.logFrequency, plan.fastButInaccurate, plan.debug
    docLens, topicMeans = \
        query.docLens, query.topicDists
    K, topicPrior, vocabPrior, wordDists, corpusTopicDist, dtype = \
        model.K, model.topicPrior, model.vocabPrior, model.wordDists, model.corpusTopicDist, model.dtype

    W = data.words

    iters, bnds, likes = [], [], []

    # Quick sanity check
    if np.any(docLens < 1):
        raise ValueError("Input document-term matrix contains at least one document with no words")
    assert dtype == np.float64, "Only implemented for 64-bit floats"

    for itr in range(iterations):
        # E-Step
        safe_log(wordDists, out=wordDists)
        safe_log(corpusTopicDist, out=corpusTopicDist)

        topicDists = W.dot(wordDists.T) + corpusTopicDist[np.newaxis, :]
        #topicDists -= topicDists.max(axis=1)[:, np.newaxis] # TODO Ensure this is okay
        norms = fns.logsumexp(topicDists, axis=1)
        topicDists -= norms[:, np.newaxis]

        np.exp(topicDists, out=topicDists)

        # M-Step
        wordDists = (W.T.dot(topicDists)).T
        wordDists += vocabPrior
        wordDists /= wordDists.sum(axis=1)[:, np.newaxis]

        corpusTopicDist     = topicDists.sum(axis=0)
        corpusTopicDist[:] += topicPrior
        corpusTopicDist    /= corpusTopicDist.sum()

        if itr % logFrequency == 0 or debug:
            m = ModelState(K, topicPrior, vocabPrior, wordDists, corpusTopicDist, True, dtype, model.name)
            q = QueryState(query.docLens, topicDists, True)

            iters.append(itr)
            bnds.append(var_bound(data, m, q))
            likes.append(log_likelihood(data, m, q))

            perp = perplexity_from_like(likes[-1], W.sum())
            print("Iteration %d : Train Perp = %4.0f  Bound = %.3f" % (itr, perp, bnds[-1]))

            if len(iters) > 2 and iters[-1] > 50:
                lastPerp = perplexity_from_like(likes[-2], W.sum())
                if lastPerp - perp < 1:
                    break;

    return ModelState(K, topicPrior, vocabPrior, wordDists, corpusTopicDist, True, dtype, model.name), \
           QueryState(query.docLens, topicDists, True), \
           (np.array(iters, dtype=np.int32), np.array(bnds), np.array(likes))


def query(data, model, queryState, _):
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
    K, wordDists, corpusTopicDist, topicPrior, vocabPrior = \
        model.K, model.wordDists, model.corpusTopicDist, model.topicPrior, model.vocabPrior
    topicDists = queryState.topicDists

    W = data.words

    wordDists = safe_log(wordDists)
    corpusTopicDist = safe_log(corpusTopicDist)

    topicDists = W.dot(wordDists.T) + corpusTopicDist[np.newaxis, :]
    norms = fns.logsumexp(topicDists, axis=1)
    topicDists -= norms[:, np.newaxis]

    return model, QueryState(queryState.docLens, np.exp(topicDists), True)


def var_bound(data, model, query):
    '''
    Determines the variational bounds.
    '''
    bound = 0

    # Unpack the the structs, for ease of access and efficiency
    docLens, topicMeans = \
        query.docLens, query.topicDists
    K, topicPrior, vocabPrior, wordDists, corpusTopicDist, dtype = \
        model.K, model.topicPrior, model.vocabPrior, model.wordDists, model.corpusTopicDist, model.dtype

    # Initialize z matrix if necessary
    W = data.words
    D, T = W.shape

    #  ln p(x,z) >= sum_k p(z=k|x) * (ln p(x|z=k, phi) + p(z=k)) + H[q]

    # Expected joint
    like = W.dot(safe_log(wordDists).T) # D*K
    like += corpusTopicDist[np.newaxis,:]
    like *= safe_log(topicMeans)

    # Entropy
    ent = (-topicMeans * safe_log(topicMeans)).sum()

    return like.sum() + bound





