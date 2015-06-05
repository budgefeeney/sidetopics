'''
Created on 15 Apr 2015

@author: bryanfeeney
'''
import sys
import numpy as np
import numpy.random as rd
import scipy.linalg as la
import scipy.sparse as ssp
import scipy.special as fns
import numba as nb


from util.sparse_elementwise import sparseScalarProductOfSafeLnDot
from util.overflow_safe import safe_log
from util.misc import constantArray, converged

from collections import namedtuple

MODEL_NAME = "rtm/vb"
DTYPE      = np.float64

# After how many training iterations should we stop to update the hyperparameters
HyperParamUpdateInterval = 5

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')

QueryState = namedtuple ( \
    'QueryState', \
    'docLens topicDists'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K topicPrior vocabPrior wordDists weights pseudoNegCount regularizer dtype name'
)

def newModelFromExisting(model):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState(\
        model.K, \
        model.topicPrior.copy(), \
        model.vocabPrior, \
        None if model.wordDists is None else model.wordDists.copy(), \
        None if model.weights is None else model.weights.copy(), \
        model.pseudoNegCount, \
        model.regularizer, \
        model.dtype, \
        model.name)


def newModelAtRandom(data, K, pseudoNegCount=None, regularizer=0.001, topicPrior=None, vocabPrior=None, dtype=DTYPE):
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
    psuedoNegCount - since we train only on positive examples, this is the
    count of negative examples to "invent" for our problem
    regularizer - the usual ridge regression coefficient
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
        topicPrior = constantArray((K + 1,), 5.0 / K + 0.5, dtype) # From Griffiths and Steyvers 2004
    if vocabPrior is None:
        vocabPrior = 0.1 + 0.5 # Also from G&S

    topicPrior[K] = 0

    wordDists = np.ones((K,T), dtype=dtype)
    doc_ids = rd.randint(0, data.doc_count, size=K)
    for k in range(K):
        sample_doc = data.words[doc_ids[k], :]
        wordDists[k, sample_doc.indices] += sample_doc.data

    # The weight vector
    weights = np.ones((K + 1,))

    # Count of dummy negative observations. Assume that for every
    # twp papers cited, 1 was considered and discarded
    if pseudoNegCount is None:
        pseudoNegCount = data.doc_count * 0.5 * np.mean(data.links.sum(axis=1).astype(DTYPE))

    return ModelState(K, topicPrior, vocabPrior, wordDists, weights, pseudoNegCount, regularizer, dtype, MODEL_NAME)


def newQueryState(data, modelState):
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
    topicDists  = rd.dirichlet(modelState.topicPrior, size=data.doc_count).astype(modelState.dtype)
    topicDists *= docLens[:, np.newaxis]
    topicDists += modelState.topicPrior[np.newaxis, :]
    topicDists[:, modelState.K] = docLens

    return QueryState(docLens, topicDists)


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
    K       = queryState.topicDists.shape[1] - 1
    result  = queryState.topicDists[:, :K].copy()
    norm    = np.sum(result, axis=1)
    result /= norm[:, np.newaxis]

    return result


def _log_likelihood_internal(data, model, query):
    _convertMeansToDirichletParam(query.docLens, query.topicDists, model.topicPrior)
    result = log_likelihood(data, model, query)
    _convertDirichletParamToMeans(query.docLens, query.topicDists, model.topicPrior)

    return result


def log_likelihood (data, modelState, queryState):
    '''
    Return the log-likelihood of the given data W and X according to the model
    and the parameters inferred for the entries in W and X stored in the
    queryState object.

    Actually returns a vector of D document specific log likelihoods
    '''
    wordLikely = sparseScalarProductOfSafeLnDot(data.words, topicDists(queryState), wordDists(modelState)).sum()
    
    # For likelihood it's a bit tricky. In theory, given d =/= p, and letting 
    # c_d = 1/n_d, where n_d is the word count of document d, it's 
    #
    #   ln p(y_dp|weights) = E[\sum_k weights[k] * (c_d \sum_n z_dnk) * (c_p \sum_n z_pnk)]
    #                      = \sum_k weights[k] * c_d * E[\sum_n z_dnk] * c_p * E[\sum_n z_pnk]
    #                      = \sum_k weights[k] * topicDistsMean[d,k] * topicDistsMean[p,k]
    #                      
    #
    # where topicDistsMean[d,k] is the mean of the k-th element of the Dirichlet parameterised
    # by topicDist[d,:]
    #
    # However in the related paper on Supervised LDA, which uses this trick of average z_dnk,
    # they explicitly say that in the likelihood calculation they use the expectation
    # according to the _variational_ approximate posterior distribution q(z_dn) instead of the
    # actual distribution p(z_dn|topicDist), and thus
    #
    # E[\sum_n z_dnk] = \sum_n E_q[z_dnk] 
    #
    # There's no detail of the likelihood in either of the RTM papers, so we use the
    # variational approch
    
    linkLikely = 0
    
    return wordLikely + linkLikely


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

@nb.autojit
def _inplace_softmax_colwise(z):
    '''
    Softmax transform of the given vector of scores into a vector of
    probabilities. Safe against overflow.

    Transform happens in-place

    :param z: a KxN matrix representing N unnormalised distributions over K
    possibilities, and returns N normalized distributions
    '''
    z_max = z.max(axis=0)
    z -= z_max[np.newaxis, :]

    np.exp(z, out=z)

    z_sum = z.sum(axis=0)
    z /= z_sum[np.newaxis, :]

@nb.autojit
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

#
# ------ <DEBUG> ------
#

def _softmax_rowwise(z):
    r = z.copy()
    _inplace_softmax_rowwise(r)
    return r

def _softmax_colwise(z):
    r = z.copy()
    _inplace_softmax_colwise(r)
    return r

def _vec_softmax(z):
    r = z.copy()
    r -= r.max()
    np.exp(r, out=r)

    r /= r.sum()
    return r

def _vocab_softmax(k, diWordDist, diWordDistSums):
    return _vec_softmax (diWordDist[k,:] - diWordDistSums[k])

#
# -------- </DEBUG> -------
#


@nb.autojit
def _update_topics_at_d(d, data, weights, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums):
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
    K = diWordDists.shape[0]
    wordIdx, z = _infer_topics_at_d(d, data, weights, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums)
    topicMeans[d, :K] = np.dot(z, data.words[d, :].data) / docLens[d]

    if containsInvalidValues(topicMeans[d, :]):
        print ("Ruh-ro")
    return wordIdx, z

@nb.autojit
def _infer_topics_at_d(d, data, weights, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums):
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
    K = diWordDists.shape[0]
    wordIdx = data.words[d, :].indices

    z  = diWordDists[:, wordIdx]
    z -= diWordDistSums[:, np.newaxis]

    distAtD = (topicPrior + docLens[d] * topicMeans[d, :])[:K, np.newaxis]

    z += fns.digamma(distAtD)
    # z -= fns.digamma(distAtD.sum())

    z += _sum_of_scores_at_d(d, data, docLens, weights, topicMeans)[:,np.newaxis]

    _inplace_softmax_colwise(z)

    return wordIdx, z

@nb.autojit
def _sum_of_scores_at_d(d, data, docLens, weights, topicMeans):
    '''

    :return:
    '''
    minNonZero  = 1E-300 if topicMeans.dtype is np.float64 else 1E-30
    K           = topicMeans.shape[1] - 1
    links       = data.links
    linked_docs = links[d, :].indices

    param = (topicMeans[d] * topicMeans[linked_docs, :]).dot(weights)

    scores  = _normpdf_inplace(param.copy())
    scores /= (_probit_inplace(param) + minNonZero)

    scores /= docLens[d]

    return scores.dot(weights[:K] * topicMeans[linked_docs, :K])


@nb.autojit
def train(data, model, query, plan, updateVocab=True):
    '''
    Infers the topic distributions in general, and specifically for
    each individual datapoint, and additionally learns the weights
    needed to predict new links.

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
    iterations, epsilon, logFrequency, fastButInaccurate, debug = \
        plan.iterations, plan.epsilon, plan.logFrequency, plan.fastButInaccurate, plan.debug
    docLens, topicMeans = \
        query.docLens, query.topicDists
    K, topicPrior, vocabPrior, wordDists, weights, negCount, reg, dtype = \
        model.K, model.topicPrior, model.vocabPrior, model.wordDists, model.weights, model.pseudoNegCount, model.regularizer, model.dtype

    # Quick sanity check
    if np.any(docLens < 1):
        raise ValueError ("Input document-term matrix contains at least one document with no words")
    assert dtype == np.float64, "Only implemented for 64-bit floats"

    # Prepare the data for inference
    topicMeans = _convertDirichletParamToMeans(docLens, topicMeans, topicPrior)

    W   = data.words
    D,T = W.shape
    X   = data.links

    iters, bnds, likes = [], [], []

    # Instead of storing the full topic assignments for every individual word, we
    # re-estimate from scratch. I.e for the memberships z which is DxNxT in dimension,
    # we only store a 1xNxT = NxT part.
    z = np.empty((K,), dtype=dtype, order='F')
    diWordDistSums = np.empty((K,), dtype=dtype)
    diWordDists = np.empty(wordDists.shape, dtype=dtype)

    for itr in range(iterations):
        if debug: printAndFlushNoNewLine("\n %4d: " % itr)

        diWordDistSums[:] = wordDists.sum(axis=1)
        fns.digamma(diWordDistSums, out=diWordDistSums)
        fns.digamma(wordDists,      out=diWordDists)

        if updateVocab:
            # Perform inference, updating the vocab
            wordDists[:, :] = vocabPrior
            for d in range(D):
                if debug and d % 100 == 0: printAndFlushNoNewLine(".")
                wordIdx, z = _update_topics_at_d(d, data, weights, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums)
                wordDists[:, wordIdx] += W[d, :].data[np.newaxis, :] * z

            _infer_weights(data, weights, topicMeans, topicPrior, negCount, reg)

            # Log bound and the determine if we can stop early
            if itr % logFrequency == 0:
                iters.append(itr)
                bnds.append(_var_bound_internal(data, model, query))
                likes.append(_log_likelihood_internal(data, model, query))

                if debug: print("%.3f < %.3f" % (bnds[-1], likes[-1]))
                if converged(iters, bnds, len(bnds) - 1, minIters=5):
                    break

            # Update hyperparameters (do this after bound, to make sure bound
            # calculation is internally consistent)
            if itr > 0 and itr % HyperParamUpdateInterval == 0:
                if debug: print("Topic Prior was " + str(topicPrior))
                _updateTopicHyperParamsFromMeans(model, query)
                if debug: print("Topic Prior is now " + str(topicPrior))
        else:
            for d in range(D):
                _ = _update_topics_at_d(d, W, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums)

    topicMeans = _convertMeansToDirichletParam(docLens, topicMeans, topicPrior)

    return ModelState(K, topicPrior, vocabPrior, wordDists, weights, negCount, reg, dtype, model.name), \
           QueryState(docLens, topicMeans), \
           (np.array(iters, dtype=np.int32), np.array(bnds), np.array(likes))


@nb.autojit
def _infer_weights(data, weights, topicMeans, topicPrior, pseudoNegCount, reg, t_0=5, kappa=0.75, max_iters=100):
    '''
    Use gradient ascent to update the weights in-place.

    :param data: the dataaset, only the links are used
    :param weights:  the weights to alter, these are altered IN-PLACE
    :param topicMeans: the means of the topic assignments
    :param topicPrior:  the topic prior
    :param pseudoNegCount: the number of documents that we expect were deliberated excluded for
    every document in the corpus
    :param reg:  the strength of the L2 regularization over weights
    :param t_0: used in the formula step-size = (t_0 + t)^{-kappa}, gives the step size
    at iteration t. t_0, a non-negative integer, slows down converged in early stages of the algorithm
    :param kappa: in the range (0.5, 1], controls the forgetting weight.
    :param max_iters: the maximum number of iterations to execute, regardless of convergence.
    :return: the weights object passed in, which has been mutated in place.
    '''
    old_weights = weights.copy()

    links  = data.links
    D, K   = topicMeans.shape
    K     -= 1 # the final element is just checked in for the intercept

    # Figure out our "error" accumulated by not including the deliberately unlinked
    # documents.
    pseudoDoc    = (topicPrior * topicPrior) / topicPrior.sum()
    pseudoDoc[K] = 1


    grad = np.ndarray(shape=weights.shape, dtype=weights.dtype)
    for t in range(max_iters):
        pseudoParam = np.array(weights.dot(pseudoDoc), dtype=weights.dtype)
        pseudoScore = _normpdf_inplace(pseudoParam.copy()) / (_probit_inplace(pseudoParam) + 1E-50)
        pseudoError = -pseudoNegCount * pseudoScore * pseudoDoc

        old_weights[:] = weights.copy()
        step_size = pow(t_0 + t, -kappa)

        # Figure out the gradient, first the regularizer
        grad[:] = reg * weights

        # then the error from the missed pseudo documents
        grad += pseudoError

        # finally the contribution of all linked documents (we count each
        # pair once only).
        for d in range(D):
            linked_docs = _links_up_to(d, links)
            if len(linked_docs) == 0:
                continue

            doc_diffs = topicMeans[d] * topicMeans[linked_docs, :]
            param = np.asarray(doc_diffs.dot(weights))
            score = _normpdf_inplace(param.copy())
            denom = _probit_inplace(param.copy())
            denom[denom == 0] = 1E-50
            score /= denom

            if np.any(np.isnan(score)) or np.any(np.isinf(score)):
                raise ValueError("Ninfs")

            grad += score.dot(doc_diffs)

        # Use the graident to do an update
        weights *= (1 - step_size)
        grad    *= step_size
        weights += grad

        if la.norm(weights - old_weights, 1) < (0.01 / K):
            break


SqrtTwoPi = 2.5066282746310002
OneOverSqrtTwoPi = 1. / SqrtTwoPi
def _normpdf_inplace(x):
    '''
    Evalate PDF of every element of X according to a standard normal distribution, and
    do it in-place
    '''
    x *= x
    x *= -0.5
    np.exp(x, out=x)
    x *= OneOverSqrtTwoPi
    return x


SqrtTwo = 1.414213562373095048801688724209
OneOverSqrtTwo = 1. / SqrtTwo
def _probit_inplace(x):
    '''
    Probit of every element of the given array, calculated in-place
    '''
    x *= -OneOverSqrtTwo
    fns.erfc(x, out=x)
    x /= 2

    return x

def _updateTopicHyperParamsFromMeans(model, query, max_iters=100):
    '''
    Update the hyperparameters on the Dirichlet prior over topics.

    This is a Newton Raphson method. We iterate until convergence or
    the maximum number of iterations is hit. We converge if the 1-norm of
    the difference between the previous and current estimate is less than
    0.001 / K where K is the number of topics.

    This is taken from Tom Minka's tech-note on "Estimating a Dircihlet
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
    topic_prior      = model.topicPrior
    old_topic_prior  = topic_prior.copy()

    doc_lens          = query.docLens
    doc_topic_counts  = query.topicDists * doc_lens[:, np.newaxis] + old_topic_prior[np.newaxis, :]

    D, K = doc_topic_counts.shape
    K -= 1 # recall topic prior is augmented by one with zero at the last position

    psi_old_tprior = np.ndarray(topic_prior.shape, dtype=topic_prior.dtype)
    topic_prior[K] = 1     # replace the zero augment with 1 to avoid NaNs etc.
    old_topic_prior[K] = 1

    for _ in range(max_iters):
        doc_topic_counts += (topic_prior - old_topic_prior)[np.newaxis, :]
        old_topic_prior[:] = topic_prior

        fns.digamma(old_topic_prior, out=psi_old_tprior)

        numer = fns.psi(doc_topic_counts).sum(axis=0) - D * psi_old_tprior
        denom = fns.psi(doc_lens + old_topic_prior[:K].sum()).sum() - D * psi_old_tprior
        topic_prior[:] = old_topic_prior * (numer / denom)
        topic_prior[K] = 1

        if la.norm(np.subtract(old_topic_prior, topic_prior), 1) < (0.001 * K):
            break

    # Undo the in-place changes we've been making to the topic distributions

    doc_topic_counts -= old_topic_prior[np.newaxis, :]
    doc_topic_counts /= doc_lens[:, np.newaxis]
    topic_prior[K]    = 0


def printAndFlushNoNewLine(text):
    sys.stdout.write(text)
    sys.stdout.flush()



@nb.autojit
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
    _, topics, (_,_,_) =  train(data, model, query, plan, updateVocab=False)
    return model, topics

@nb.autojit
def _var_bound_internal(data, model, query, z_dnk = None):
    _convertMeansToDirichletParam(query.docLens, query.topicDists, model.topicPrior)
    result = var_bound(data, model, query, z_dnk)
    _convertDirichletParamToMeans(query.docLens, query.topicDists, model.topicPrior)

    return result


@nb.autojit
def var_bound(data, model, query, z_dnk = None):
    '''
    Determines the variational bounds.
    '''
    bound = 0
    
    # Unpack the the structs, for ease of access and efficiency
    K, topicPrior, wordPrior, wordDists, weights, negCount, reg, dtype = \
        model.K, model.topicPrior, model.vocabPrior, model.wordDists, model.weights, model.pseudoNegCount, model.regularizer, model.dtype
    docLens, topicDists = \
        query.docLens, query.topicDists

    W,X = data.words, data.links
    D,T = W.shape
    minNonZero = 1E-300 if dtype is np.float64 else 1E-30
        
    # Perform the digamma transform for E[ln \theta] etc.
    topicDists      = topicDists.copy()
    diTopicDists    = fns.digamma(topicDists[:, :K])
    diSumTopicDists = fns.digamma(topicDists[:, :K].sum(axis=1))
    diWordDists     = fns.digamma(model.wordDists)
    diSumWordDists  = fns.digamma(model.wordDists.sum(axis=1))

    # E[ln p(topics|topicPrior)] according to q(topics)
    #
    prob_topics = D * (fns.gammaln(topicPrior[:K].sum()) - fns.gammaln(topicPrior[:K]).sum()) \
        + np.sum((topicPrior[:K] - 1)[np.newaxis, :] * (diTopicDists - diSumTopicDists[:, np.newaxis]))

    bound += prob_topics

    # and its entropy
    ent_topics = _dirichletEntropy(topicDists[:, :K])
    bound += ent_topics
        
    # E[ln p(vocabs|vocabPrior)]
    #
    if type(model.vocabPrior) is float or type(model.vocabPrior) is int:
        prob_vocabs  = K * (fns.gammaln(wordPrior * T) - T * fns.gammaln(wordPrior)) \
               + np.sum((wordPrior - 1) * (diWordDists - diSumWordDists[:,np.newaxis] ))
    else:
        prob_vocabs  = K * (fns.gammaln(wordPrior.sum()) - fns.gammaln(wordPrior).sum()) \
               + np.sum((wordPrior - 1)[np.newaxis,:] * (diWordDists - diSumWordDists[:,np.newaxis] ))

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
        wordIdx, z = _infer_topics_at_d(d, data, weights, docLens, topicMeans, topicPrior, diWordDists, diSumWordDists)

        # E[ln p(Z|topics) = sum_d sum_n sum_k E[z_dnk] E[ln topicDist_dk]
        exLnTopic = diTopicDists[d, :K] - diSumTopicDists[d]
        prob_z += np.dot(z * exLnTopic[:, np.newaxis], W[d, :].data).sum()

        # E[ln p(W|Z)] = sum_d sum_n sum_k sum_t E[z_dnk] w_dnt E[ln vocab_kt]
        prob_words += np.sum(W[d, :].data[np.newaxis, :] * z * (diWordDists[:, wordIdx] - diSumWordDists[:, np.newaxis]))
        
        # And finally the entropy of Z
        ent_z -= np.dot(z * safe_log(z), W[d, :].data).sum()

    bound += (prob_z + ent_z + prob_words)

    # Next, the distribution over links - we just focus on the positives in this case
    for d in range(D):
        links   = _links_up_to(d, X)
        if len(links) == 0:
            continue

        scores  = topicMeans[links, :].dot(weights * topicMeans[d])
        probs   = _probit_inplace(scores) + minNonZero
        lnProbs = np.log(probs, out=probs)

        # expected probability of all links from d to p < d such that y_dp = 1
        bound += lnProbs.sum()

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


def _links_up_to (d, X):
    '''
    Gets all the links that exist to earlier documents in the corpus. Ensures
    that if we iterate through all documents, we only ever consider each link
    once.
    '''
    return _links_up_to_csr(d, X.indptr, X.indices)


@nb.autojit
def _links_up_to_csr(d, Xptr, Xindices):
    '''
    Gets all the links that exist to earlier documents in the corpus. Ensures
    that if we iterate through all documents, we only ever consider each link
    once. Assumes we're working on a DxD CSR matrix.
    '''
    
    start = Xptr[d]
    end   = start
    while end < Xptr[d+1]:
        if Xindices[end] >= d:
            break
        end += 1
    
    result = Xindices[start:end]
    
    return result


def is_undirected_link_predictor():
    return True


@nb.autojit
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
    weights    = model.weights
    topicMeans = _convertDirichletParamToMeans(topics.docLens, topics.topicDists, model.topicPrior)
    D = topicMeans.shape[0]

    # use the topics means to predict links
    mins = np.ones((D,), dtype=model.dtype)
    for d in range(D):
        if len(links[d,:].indices) == 0:
            mins[d] = 1E-300 if model.dtype is np.float64 else 1E-30
        else:
            probs = (topicMeans[d] * topicMeans[links[d,:].indices]).dot(weights)
            mins[d] = _probit_inplace(probs).min()

    _convertMeansToDirichletParam(topics.docLens, topics.topicDists, model.topicPrior) # revert topicDists / topicMeans
    return mins


def extend_topic_prior (prior_vec, extra_field):
    return np.hstack ((prior_vec, extra_field))


@nb.autojit
def link_probs(model, topics, min_link_probs):
    '''
    Generate the probability of a link for all possible pairs of documents,
    but only store those probabilities that are bigger than or equal to the
    minimum. This ensures, hopefully, that we don't materialise a complete
    DxD matrix, but rather the minimum needed to determine the mean
    average precsions

    :param model: the trained model
    :param topics: the topics for each of teh documents we're generating
        links for
    :param min_link_probs: the minimum link probability for each document
    :return: a (hopefully) sparse DxD matrix of link probabilities
    '''
    weights    = model.weights

    # We build the result up as a COO matrix
    rows = []
    cols = []
    vals = []

    topicMeans = _convertDirichletParamToMeans(topics.docLens, topics.topicDists, model.topicPrior)
    D = topicMeans.shape[0]

    # use the topics means to predict links
    for d in range(D):
        probs = (topicMeans[d] * topicMeans).dot(weights)
        probs = _probit_inplace(probs)
        relevant = np.where(probs >= min_link_probs[d])[0]

        rows.extend([d] * len(relevant))
        cols.extend(relevant)
        vals.extend(probs[relevant])

    # return from topic means to the topic distributions
    _convertMeansToDirichletParam(topics.docLens, topics.topicDists, model.topicPrior)

    # Build the COO matrix, then covert it to CSR. Converts lists to numpy
    # arrays to ensure appropriate dtypes
    r = np.array(rows, dtype=np.int32)
    c = np.array(cols, dtype=np.int32)
    v = np.array(vals, dtype=model.dtype)

    return ssp.coo_matrix((v, (r, c)), shape=(D, D)).tocsr()


def containsInvalidValues(x):
    return np.any(np.isnan(x)) or np.any(np.isinf(x))

