'''
Created on 15 Apr 2015

@author: bryanfeeney
'''
import sys
import numpy as np
import numpy.random as rd
import scipy.linalg as la
import scipy.special as fns

from math import exp
#import numba as nb


from sidetopics.util.sparse_elementwise import sparseScalarProductOfSafeLnDot
from sidetopics.util.overflow_safe import safe_log
from sidetopics.util.misc import constantArray, converged

from sidetopics.model.evals import perplexity_from_like

from collections import namedtuple

MODEL_NAME = "lda/vbp"
DTYPE      = np.float64
VocabPrior = 1.1

# After how many training iterations should we stop to update the hyperparameters
HyperParamUpdateInterval = 5
HyperUpdateEnabled = False

RateAlgorBatch="batch"
RateAlgorTimeKappa="kappa"
RateAlgorAmaria="amari"
RateAlgorVariance="variance"
RateAlgors=[RateAlgorTimeKappa, RateAlgorAmaria, RateAlgorVariance]
DefaultNonBatchRateAlgor=RateAlgorTimeKappa

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug batchSize rate_delay forgetting_rate rate_a rate_b rate_algor')

QueryState = namedtuple ( \
    'QueryState', \
    'docLens topicDists processed'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K topicPrior vocabPrior wordDists processed dtype name'
)

def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState(\
        model.K, \
        model.topicPrior.copy(), \
        model.vocabPrior, \
        None if model.wordDists is None else model.wordDists.copy(),
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
    assert K > 1,   "There must be at least two topics"
    assert K < 255, "There can be no more than 255 topics"
    T = data.words.shape[1]

    if topicPrior is None:
        topicPrior = constantArray((K,), 5.0 / K + 0.5, dtype) # From Griffiths and Steyvers 2004
    elif type(topicPrior) is float:
        topicPrior = constantArray((K,), topicPrior, dtype) # From Griffiths and Steyvers 2004
    if vocabPrior is None:
        vocabPrior = VocabPrior

    wordDists = np.ones((K,T), dtype=dtype)
    for k in range(K):
        docLenSum = 0
        while docLenSum < 1000:
            randomDoc  = rd.randint(0, data.doc_count, size=1)
            sample_doc = data.words[randomDoc, :]
            wordDists[k, sample_doc.indices] += sample_doc.data
            docLenSum += sample_doc.sum()
        wordDists[k,:] /= wordDists[k,:].sum()

    print(f'wordDists[0,:10] = {wordDists[0,:10]}')
    return ModelState(K, topicPrior, vocabPrior, wordDists, False, dtype, MODEL_NAME)


def newQueryState(data, modelState, debug=False):
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
    docLens = np.squeeze(np.asarray(data.words.sum(axis=1)))

    # Initialise the per-token assignments at random according to the dirichlet hyper
    # This is super-slow
    dist = modelState.topicPrior.copy()
    dist /= dist.sum()

    topicDists  = rd.dirichlet(dist, size=data.doc_count).astype(modelState.dtype)
    topicDists *= docLens[:, np.newaxis]
    topicDists += modelState.topicPrior[np.newaxis, :]

    return QueryState(docLens, topicDists, debug)


def pruneQueryState (query, indices):
    '''
    Returns a query state corresponding to the given indices only
    '''
    return QueryState(query.docLens[indices], query.topicDists[indices,:], query.processed)


def newTrainPlan(iterations=100, epsilon=2, logFrequency=10, fastButInaccurate=False, debug=False, batchSize=0,
                 rate_delay=0.6, forgetting_rate=0.6, rate_a=2, rate_b=0.5, rate_algor=RateAlgorBatch):
    '''
    Create a training plan determining how many iterations we
    process, how often we plot the results, how often we log
    the variational bound, etc.

    epsilon is oddly measured, we just evaluate the angle of the line segment between
    the last value of the bound and the current, and if it's less than the given angle,
    then stop.
    '''
    if batchSize > 0:
        rate_algor = DefaultNonBatchRateAlgor

    return TrainPlan(iterations, epsilon, logFrequency, fastButInaccurate, debug, batchSize, rate_delay,
                     forgetting_rate, rate_a, rate_b, rate_algor)


def wordDists (modelState):
    '''
    The K x T matrix of  word distributions inferred for the K topics
    '''
    result = modelState.wordDists.copy()
    norm   = result.sum(axis=1)
    result /= norm[:,np.newaxis]

    return result



def topicDists (queryState):
    '''
    The D x K matrix of topics distributions inferred for the K topics
    across all D documents
    '''
    result  = queryState.topicDists.copy()
    norm    = np.sum(result, axis=1)
    result /= norm[:, np.newaxis]

    return result


def _log_likelihood_internal(data, model, query):
    _convertMeansToDirichletParam(query.docLens, query.topicDists, model.topicPrior)
    result = log_likelihood(data, model, query)
    _convertDirichletParamToMeans(query.docLens, query.topicDists, model.topicPrior)

    return result


def log_likelihood (data, model, query, topicDistOverride=None):
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


def _convertDirichletParamToMeans(docLens, topicMeans, topicPrior):
    '''
    Convert the Dirichlet parameter to a mean of per-token topic assignments
    in-place
    '''
    topicMeans -= topicPrior[np.newaxis, :]
    topicMeans /= docLens[:, np.newaxis]
    return topicMeans


def _convertMeansToDirichletParam(docLens, topicMeans, topicPrior):
    '''
    Convert the means of per-token topic assignments to a Dirichlet parameter
    in-place
    '''
    topicMeans *= docLens[:, np.newaxis]
    topicMeans += topicPrior[np.newaxis, :]
    return topicMeans


def is_not_all_real(z):
    return np.any(np.isnan(z)) or np.any(np.isinf(z))


#@nb.jit
def _inplace_softmax_colwise(z):
    '''
    Softmax transform of the given vector of scores into a vector of
    probabilities. Safe against overflow.

    Transform happens in-place

    :param z: a KxN matrix representing N unnormalised distributions over K
    possibilities, and returns N normalized distributions
    '''

    if is_not_all_real(z):
        print("Yoinks, Scoob!")

    z_max = z.max(axis=0)
    z -= z_max[np.newaxis, :]

    np.exp(z, out=z)

    z_sum = z.sum(axis=0)
    z /= z_sum[np.newaxis, :]

#@nb.jit
def _inplace_softmax_rowwise(z):
    '''
    Softmax transform of the given vector of scores into a vector of
    probabilities. Safe against overflow.

    Transform happens in-place

    :param z: a NxK matrix representing N unnormalised distributions over K
    possibilities, and returns N normalized distributions
    '''
    z_max = z.max(axis=1)
    z -= z_max[:, np.newaxis]

    np.exp(z, out=z)

    z_sum = z.sum(axis=1)
    z /= z_sum[:, np.newaxis]


#@nb.jit
def _update_topics_at_d(d, data, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums):
    '''
    Infers the topic assignments for all present words in the given document at
    index d as stored in the sparse CSR matrix W. This are used to update the
    topicMeans matrix in-place! The indices of the non-zero words, and their
    probabilities, are returned.
    :param d:  the document for which we update the topic distribution
    :param W:  the DxT document-term matrix, a sparse CSR matrix.
    :param docLens:  the length of each document
    :param topicMeans: the DxK matrix, where the d-th row contains the mean of all
                        per-token topic-assignments.
    :param topicPrior: the prior over topics
    :param diWordDists: the KxT matrix of word distributions, after the digamma funciton
                        has been applied
    :param diWordDistSums: the K-vector of the digamma of the sums of the Dirichlet
                            parameter vectors for each per-topic word-distribution
    :return: the indices of the non-zero words in document d, and the KxV matrix of
            topic assignments for each of the V non-zero words.
    '''
    wordIdx, z = _infer_topics_at_d(d, data, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums)
    topicMeans[d, :] = np.dot(z, data.words[d, :].data) / docLens[d]
    return wordIdx, z

#@nb.jit
def _infer_topics_at_d(d, data, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums):
    '''
    Infers the topic assignments for all present words in the given document at
    index d as stored in the sparse CSR matrix W. This does not affect topicMeans.
    The indices of the non-zero words, and their probabilities, are returned.

    :param d:  the document for which we update the topic distribution
    :param W:  the DxT document-term matrix, a sparse CSR matrix.
    :param docLens:  the length of each document
    :param topicMeans: the DxK matrix, where the d-th row contains the mean of all
                        per-token topic-assignments.
    :param topicPrior: the prior over topics
    :param diWordDists: the KxT matrix of word distributions, after the digamma funciton
                        has been applied
    :param diWordDistSums: the K-vector of the digamma of the sums of the Dirichlet
                            parameter vectors for each per-topic word-distribution
    :return: the indices of the non-zero words in document d, and the KxV matrix of
            topic assignments for each of the V non-zero words.
    '''
    wordIdx = data.words[d, :].indices

    z  = diWordDists[:, wordIdx]
    z -= diWordDistSums[:, np.newaxis]

    distAtD = (topicPrior + docLens[d] * topicMeans[d, :])[:, np.newaxis]

    z += fns.digamma(distAtD)
    # z -= fns.digamma(distAtD.sum())

    _inplace_softmax_colwise(z)

    return wordIdx, z


def train(data, model, query, plan, updateVocab=True):
    '''
    Infers the topic distributions in general, and specifically for
    each individual datapoint,

    Params:
    data - the training data, we just use the DxT document-term matrix
    model - the initial model configuration. This is MUTATED IN-PLACE
    qy=uery - the query results - essentially all the "local" variables
            matched to the given observations. Also MUTATED IN-PLACE
    plan  - how to execute the training process (e.g. iterations,
            log-interval etc.)

    Return:
    The updated model object (note parameters are updated in place, so make a
    defensive copy if you want it)
    The query object with the update query parameters
    '''
    iterations, epsilon, logFrequency, fastButInaccurate, debug, batchSize, rateAlgor = \
        plan.iterations, plan.epsilon, plan.logFrequency, plan.fastButInaccurate, plan.debug, plan.batchSize, plan.rate_algor
    docLens, topicMeans = \
        query.docLens, query.topicDists
    K, topicPrior, vocabPrior, wordDists, dtype = \
        model.K, model.topicPrior, model.vocabPrior, model.wordDists, model.dtype

    # Quick sanity check
    if np.any(docLens < 1):
        raise ValueError("Input document-term matrix contains at least one document with no words")
    assert model.dtype == np.float64, "Only implemented for 64-bit floats"

    # Prepare the data for inference
    topicMeans = _convertDirichletParamToMeans(docLens, topicMeans, topicPrior)

    W = data.words
    D, T = W.shape

    iters, bnds, likes = [], [], []

    # A few parameters for handling adaptive step-sizes in SGD
    if plan.rate_algor == RateAlgorBatch:
        batchSize  = D
        batchCount = 1
    else:
        batchSize  = plan.batchSize
        batchCount = D // batchSize + 1

    gradStep = constantArray((K,), 1./float(batchSize), dtype=dtype)
    grad     = np.zeros((K,T), dtype=dtype)
    ex_grad  = grad.copy()
    exp_gtg  = np.zeros((K,), dtype=dtype)
    stepSize = np.ones((K,), dtype=dtype)

    # The digamma terms for the vocabularly
    diWordDists = fns.digamma(wordDists)
    diWordDistSums = np.sum(wordDists, axis=1)
    fns.digamma(diWordDistSums, out=diWordDistSums)

    # Amend the name to incorporate training information
    rateAlgor = plan.rate_algor
    modelName = "lda/svbp/%s" % _sgd_desc(plan)
    print(modelName)

    # Start traininng
    d = -1
    for b in range(batchCount * iterations):
        grad.fill(vocabPrior)
        # firstD = d
        for s in range(batchSize):
            d = d + 1 if (d + 1) < D else 0

            wordIdx, z = _update_topics_at_d(d, data, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums)
            grad[:, wordIdx] += W[d, :].data[np.newaxis, :] * z

        if rateAlgor == RateAlgorBatch:
            wordDists[:, :] = grad[:, :]
        else:
            if rateAlgor == RateAlgorTimeKappa:
                stepSize[:] = (b + plan.rate_delay)**(-plan.forgetting_rate)
            elif rateAlgor == RateAlgorVariance:
                update_inplace_v(gradStep, ex_grad, change=grad)
                gtg = stepSize.copy()
                for k in range(K):
                    stepSize[k] = np.dot(ex_grad[k,:], ex_grad[k,:])
                    gtg[k] = np.dot(grad[k,:], grad[k,:])
                update_inplace_s(gradStep, old=exp_gtg, change=gtg)
                stepSize /= exp_gtg
                gradStep  = gradStep * (1 - stepSize) + 1
            elif rateAlgor == RateAlgorAmaria:
                topicMeans = _convertMeansToDirichletParam(docLens, topicMeans, topicPrior)
                # doc_indices = np.linspace(firstD, firstD + batchSize -1, batchSize) % D
                log_likely = var_bound(
                    data, # data._reorder(doc_indices),
                    ModelState(K, topicPrior, vocabPrior, wordDists, True, dtype, modelName),
                    QueryState(docLens, topicMeans, True)
                )
                p     = stepSize[0]
                a, b  = plan.rate_a, plan.rate_b
                p    *= exp(a * (b * -log_likely - p))
                stepSize[:] = p
                topicMeans = _convertMeansToDirichletParam(docLens, topicMeans, topicPrior)
            else:
                raise ValueError("No code to support the '" + str(plan.rate_algor) + "' learning rate adaptation algorithm")

            update_inplace_v(stepSize, old=wordDists, change=grad)

        if debug:
            print ("%s : t=%d : step=%s" % (rateAlgor, b, str(stepSize)))

        if is_not_all_real(wordDists):
            print("Worddists nan")
        fns.digamma(wordDists, out=diWordDists)
        if is_not_all_real(diWordDists):
            print("Digamma worddists nan")
        np.sum(wordDists, axis=1, out=diWordDistSums)
        fns.digamma(diWordDistSums, out=diWordDistSums)

    topicMeans = _convertMeansToDirichletParam(docLens, topicMeans, topicPrior)

    return ModelState(K, topicPrior, vocabPrior, wordDists, True, dtype, modelName), \
           QueryState(docLens, topicMeans, True), \
           (np.array(iters, dtype=np.int32), np.array(bnds), np.array(likes))

def update_inplace_s(stepSizeScalar, old, change):
    old    *= (1 - stepSizeScalar)
    change *= stepSizeScalar
    old    += change

def update_inplace_v(stepSizeVector, old, change):
    old    *= (np.ones((stepSizeVector.shape[0],)) - stepSizeVector)[:, np.newaxis]
    change *= stepSizeVector[:, np.newaxis]
    old    += change

##@nb.jit
def _old_train(data, model, query, plan, updateVocab=True):
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
    iterations, epsilon, logFrequency, fastButInaccurate, debug, batchSize = \
        plan.iterations, plan.epsilon, plan.logFrequency, plan.fastButInaccurate, plan.debug, plan.batchSize
    docLens, topicMeans = \
        query.docLens, query.topicDists
    K, topicPrior, vocabPrior, wordDists ,dtype = \
        model.K, model.topicPrior, model.vocabPrior, model.wordDists, model.dtype

    # Quick sanity check
    if np.any(docLens < 1):
        raise ValueError ("Input document-term matrix contains at least one document with no words")
    assert model.dtype == np.float64, "Only implemented for 64-bit floats"

    # Prepare the data for inference
    topicMeans = _convertDirichletParamToMeans(docLens, topicMeans, topicPrior)

    W   = data.words
    D,T = W.shape

    iters, bnds, likes = [], [], []

    # A few parameters for handling adaptive step-sizes in SGD
    grad = 0
    grad_inner = 0
    grad_rate = 1
    log_likely = 0 # complete dataset likelihood for gradient adjustments
    stepSize = np.array([1.] * K, dtype=model.dtype)

    # Instead of storing the full topic assignments for every individual word, we
    # re-estimate from scratch. I.e for the memberships z which is DxNxT in dimension,
    # we only store a 1xNxT = NxT part.
    diWordDistSums = np.empty((K,), dtype=dtype)
    diWordDists    = np.empty(wordDists.shape, dtype=dtype)
    wordUpdates    = wordDists.copy() if batchSize > 0 else None
    batchProcessCount = 0

    # Amend the name if batchSize == 0 implying we're using SGD
    modelName = "lda/svbp/%s" % _sgd_desc(plan) \
                if batchSize > 0 else model.name
    print (modelName)

    for itr in range(iterations):
        diWordDistSums[:] = wordDists.sum(axis=1)
        fns.digamma(diWordDistSums, out=diWordDistSums)
        fns.digamma(wordDists,      out=diWordDists)

        if updateVocab:
            # Perform inference, updating the vocab
            if batchSize == 0:
                wordDists[:, :] = vocabPrior
            else:
                wordUpdates[:,:] = 0

            for d in range(D):
                batchProcessCount += 1
                #if debug and d % 100 == 0: printAndFlushNoNewLine(".")
                wordIdx, z = _update_topics_at_d(d, data, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums)
                wordDists[:, wordIdx] += W[d, :].data[np.newaxis, :] * z

                if plan.rate_algor == RateAlgorAmaria:
                    log_likely += 0
                elif plan.rate_algor == RateAlgorVariance:
                    g = wordDists.mean(axis=0) + vocabPrior
                    grad *= (1 - grad_rate)
                    grad += grad_rate * wordDists
                    grad += grad_rate * vocabPrior
                    gg += 0
                elif plan.rate_algor != RateAlgorTimeKappa:
                    raise ValueError("Unknown rate algorithm " + str(plan.rate_algor))

                if batchSize > 0 and batchProcessCount == batchSize:
                    batch_index    = (itr * D + d) / batchSize #TODO  Will not be right if batchSize is not a multiple of D
                    stepSize       = _step_sizes(stepSize, batch_index, g, gg, log_likely, plan)
                    wordDists     *= (1 - stepSize)
                    wordDists     += stepSize * vocabPrior

                    stepSize      *= float(D) / batchSize
                    wordUpdates   *= stepSize
                    wordDists     += wordUpdates

                    diWordDistSums[:] = wordDists.sum(axis=1)
                    fns.digamma(diWordDistSums, out=diWordDistSums)
                    fns.digamma(wordDists, out=diWordDists)

                    wordUpdates[:,:] = 0
                    batchProcessCount = 0
                    log_likely = 0

                    if debug:
                        bnds.append(_var_bound_internal(data, model, query))
                        likes.append(_log_likelihood_internal(data, model, query))

                        perp = perplexity_from_like(likes[-1], W.sum())
                        print("Iteration %d, after %d docs: Train Perp = %4.0f  Bound = %.3f" % (itr, batchSize, perp, bnds[-1]))
                        sys.stdout.flush()


            # Log bound and the determine if we can stop early
            if itr % logFrequency == 0 or debug:
                iters.append(itr)
                bnds.append(_var_bound_internal(data, model, query))
                likes.append(_log_likelihood_internal(data, model, query))

                perp = perplexity_from_like(likes[-1], W.sum())
                print ("Iteration %d : Train Perp = %4.0f  Bound = %.3f" % (itr, perp, bnds[-1]))

                if len(iters) > 2 and (iters[-1] > 20 or (iters[-1] > 2 and batchSize > 0)):
                    lastPerp = perplexity_from_like(likes[-2], W.sum())
                    if lastPerp - perp < 1:
                        print ("Converged, existing early")
                        break;

            # Update hyperparameters (do this after bound, to make sure bound
            # calculation is internally consistent)
            if HyperUpdateEnabled and itr > 0 and itr % HyperParamUpdateInterval == 0:
                if debug: print("Topic Prior was " + str(topicPrior))
                _updateTopicHyperParamsFromMeans(model, query)
                if debug: print("Topic Prior is now " + str(topicPrior))
        else:
            for d in range(D):
                _ = _update_topics_at_d(d, data, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums)

    topicMeans = _convertMeansToDirichletParam(docLens, topicMeans, topicPrior)

    return ModelState(K, topicPrior, vocabPrior, wordDists, True, dtype, modelName), \
           QueryState(docLens, topicMeans, True), \
           (np.array(iters, dtype=np.int32), np.array(bnds), np.array(likes))


def _step_sizes(last_step, t, g, gg, log_likely, train_plan):
    """
    Evaluates the step size according to the metric specified by rate_algor. Do a separate
    stepSize for each parameter.
    :param last_step: The last values of step sizes
    :param t: the batch indicator
    :param g: the estimate of the expected gradient
    :param gg: the estimate of the expected inner product of the gradients
    :param log_likely: the log-likelihood of the dataset
    :param train_plan: the training plan, specifies the training dataset.
    """
    K, dtype = len(last_step), last_step.dtype

    if train_plan.rate_algor == RateAlgorTimeKappa:
        return constantArray(K, (t + train_plan.rate_delay) ** (-train_plan.forgetting_rate), dtype)
    elif train_plan.rate_algor == RateAlgorAmaria:
        s = last_step[0] * exp(train_plan.rate_a * (train_plan.rate_b * -log_likely - last_step[0]))
        return constantArray(K, s, dtype)
    elif train_plan.rate_algor == RateAlgorVariance:
        s = g.dot(g)
        s[:, np.newaxis] /= gg
        return s
    else:
        raise ValueError ("Unknown rate algorithm " + str(train_plan.rate_algor))

def _sgd_desc (train_plan):
    """
    Return a / delimited string describing the current SGD training schedule
    :param train_plan:
    :return:
    """
    if train_plan.rate_algor == RateAlgorBatch:
        return RateAlgorBatch

    result = train_plan.rate_algor + "/" + str(train_plan.batchSize)
    if train_plan.rate_algor == RateAlgorTimeKappa:
        return result + "/%.2f/%.2f" % (train_plan.rate_delay, train_plan.forgetting_rate)
    elif train_plan.rate_algor == RateAlgorAmaria:
        return result + "/%.2f/%.2f" % (train_plan.rate_a, train_plan.rate_b)
    elif train_plan.rate_algor == RateAlgorVariance:
        return result
    else:
        raise ValueError ("Unknown rate algorithm " + str(train_plan.rate_algor))




def _updateTopicHyperParamsFromMeans(model, query, max_iters=100):
    '''
    Update the hyperparameters on the Dirichlet prior over topics.

    This is a Newton Raphson method. We iterate until convergence or
    the maximum number of iterations is hit. We converge if the 1-norm of
    the difference between the previous and current estimate is less than
    0.001 / K where K is the number of topics.

    This is taken from Tom Minka's tech-note on "Estimating a Dirichlet
    Distribution", specifically the section on estimating a Polya distribution,
    which performed best in experiments. We'll be substituted in the expected
    count of topic assignments to variables.

    At each iteration, the new value of a_k is set to

     \sum_d \Psi(n_dk + a_k) - \Psi (a_k)
    -------------------------------------- * a_k
     \sum_d \Psi(n_d + \sum_j a_j) - \Psi(a_k)

    where the n_dk is the count of times topic k was assigned to tokens in
    document d, and its expected value is the same as the parameter of the
    posterior over topics for that document d, minus the hyper-parameter used
    to estimate that posterior. In this case, we assume that this method have
    been called from within the training routine, so we topicDists is essentially
    the mean of per-token topic-assignments, and thus needs to be scaled
    appropriately

    :param model: all the model parameters, notably the topicPrior, which is
    mutated IN-PLACE.
    :param query: all the document-level parameters, notably the topicDists,
    from which an appropriate prior is noted. It's expected that this contain
    the topic hyper-parameters, as usual, and not any intermediate reprsentations
    (i.e. means) used by the inference procedure.
    '''
    print ("Updating hyper-parameters")
    topic_prior      = model.topicPrior
    old_topic_prior  = topic_prior.copy()

    doc_lens          = query.docLens
    doc_topic_counts  = query.topicDists * doc_lens[:, np.newaxis] + old_topic_prior[np.newaxis, :]

    D, K = doc_topic_counts.shape

    psi_old_tprior = np.ndarray(topic_prior.shape, dtype=topic_prior.dtype)

    for _ in range(max_iters):
        doc_topic_counts += (topic_prior - old_topic_prior)[np.newaxis, :]
        old_topic_prior[:] = topic_prior

        fns.digamma(old_topic_prior, out=psi_old_tprior)

        numer = fns.psi(doc_topic_counts).sum(axis=0) - D * psi_old_tprior
        denom = fns.psi(doc_lens + old_topic_prior.sum()).sum() - D * psi_old_tprior
        topic_prior[:] = old_topic_prior * (numer / denom)

        if la.norm(np.subtract(old_topic_prior, topic_prior), 1) < (0.001 * K):
            break

    # Undo the in-place changes we've been making to the topic distributions

    doc_topic_counts -= old_topic_prior[np.newaxis, :]
    doc_topic_counts /= doc_lens[:, np.newaxis]

    # Make sure it never is zero or negative
    for k in range(K):
        topic_prior[k] = max(topic_prior[k], 1E-6)


def printAndFlushNoNewLine(text):
    sys.stdout.write(text)
    sys.stdout.flush()



#@nb.jit
def query(data, model, query, plan):
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
    _, topics, (_,_,_) = train(data, model, query, plan, updateVocab=False)
    return model, topics

#@nb.jit
def _var_bound_internal(data, model, query, z_dnk = None):
    _convertMeansToDirichletParam(query.docLens, query.topicDists, model.topicPrior)
    result = var_bound(data, model, query, z_dnk)
    _convertDirichletParamToMeans(query.docLens, query.topicDists, model.topicPrior)

    return result


#@nb.jit
def var_bound(data, model, query, z_dnk = None):
    '''
    Determines the variational bounds.
    '''
    bound = 0

    # Unpack the the structs, for ease of access and efficiency
    K, topicPrior, wordPrior, wordDists, dtype = \
        model.K, model.topicPrior, model.vocabPrior, model.wordDists, model.dtype
    docLens, topicDists = \
        query.docLens, query.topicDists

    # Initialize z matrix if necessary
    W,X = data.words, data.links
    D,T = W.shape

    # Perform the digamma transform for E[ln \theta] etc.
    topicDists      = topicDists.copy()
    diTopicDists    = fns.digamma(topicDists)
    diSumTopicDists = fns.digamma(topicDists.sum(axis=1))
    diWordDists     = fns.digamma(model.wordDists)
    diSumWordDists  = fns.digamma(model.wordDists.sum(axis=1))

    # E[ln p(topics|topicPrior)] according to q(topics)
    #
    prob_topics = D * (fns.gammaln(topicPrior.sum()) - fns.gammaln(topicPrior).sum()) \
        + np.sum((topicPrior - 1)[np.newaxis, :] * (diTopicDists - diSumTopicDists[:, np.newaxis]))

    bound += prob_topics

    # and its entropy
    ent_topics = _dirichletEntropy(topicDists)
    bound += ent_topics

    # E[ln p(vocabs|vocabPrior)]
    #
    if type(model.vocabPrior) is float or type(model.vocabPrior) is int:
        prob_vocabs = K * (fns.gammaln(wordPrior * T) - T * fns.gammaln(wordPrior)) \
               + np.sum((wordPrior - 1) * (diWordDists - diSumWordDists[:, np.newaxis] ))
    else:
        prob_vocabs = K * (fns.gammaln(wordPrior.sum()) - fns.gammaln(wordPrior).sum()) \
               + np.sum((wordPrior - 1)[np.newaxis,:] * (diWordDists - diSumWordDists[:, np.newaxis] ))

    bound += prob_vocabs

    # and its entropy
    ent_vocabs = _dirichletEntropy(wordDists)
    bound += ent_vocabs

    # P(z|topic) is tricky as we don't actually store this. However
    # we make a single, simple estimate for this case.
    topicMeans = _convertDirichletParamToMeans(docLens, topicDists, topicPrior)

    prob_words = 0
    prob_z     = 0
    ent_z      = 0
    for d in range(D):
        wordIdx, z = _infer_topics_at_d(d, data, docLens, topicMeans, topicPrior, diWordDists, diSumWordDists)

        # E[ln p(Z|topics) = sum_d sum_n sum_k E[z_dnk] E[ln topicDist_dk]
        exLnTopic = diTopicDists[d, :] - diSumTopicDists[d]
        prob_z += np.dot(z * exLnTopic[:, np.newaxis], W[d, :].data).sum()

        # E[ln p(W|Z)] = sum_d sum_n sum_k sum_t E[z_dnk] w_dnt E[ln vocab_kt]
        prob_words += np.sum(W[d, :].data[np.newaxis, :] * z * (diWordDists[:, wordIdx] - diSumWordDists[:, np.newaxis]))

        # And finally the entropy of Z
        ent_z -= np.dot(z * safe_log(z), W[d, :].data).sum()

    bound += (prob_z + ent_z + prob_words)

    _convertMeansToDirichletParam(docLens, topicMeans, topicPrior)
    return bound


def _dirichletEntropy (P):
    '''
    Entropy of D Dirichlet distributions, with dimension K, whose parameters
    are given by the DxK matrix P
    '''
    D,K    = P.shape
    psums  = P.sum(axis=1)
    lnB   = fns.gammaln(P).sum(axis=1) - fns.gammaln(psums)
    term1 = (psums - K) * fns.digamma(psums)
    term2 = (P - 1) * fns.digamma(P)

    return (lnB + term1 - term2.sum(axis=1)).sum()
