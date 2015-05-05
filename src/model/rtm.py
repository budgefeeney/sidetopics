'''
Created on 15 Apr 2015

@author: bryanfeeney
'''
import sys
import numpy as np
import numpy.random as rd
import scipy.sparse as ssp
import scipy.special as fns
import numba as nb

#import model.rtm_fast as compiled
from util.sparse_elementwise import sparseScalarProductOfSafeLnDot
from util.overflow_safe import safe_log
from util.misc import constantArray, converged, clamp

from collections import namedtuple

MODEL_NAME = "rtm/vb"
DTYPE      = np.float64

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
        topicPrior = constantArray((K + 1,), 50.0 / K + 0.5, dtype) # From Griffiths and Steyvers 2004
    if vocabPrior is None:
        vocabPrior = 0.1 + 0.5 # Also from G&S

    topicPrior[K] = 0

    wordDists = rd.dirichlet(constantArray((T,), 2, dtype), size=K).astype(dtype)

    # Peturb to avoid zero probabilities
    wordDists += 1./T
    wordDists /= (wordDists.sum(axis=1))[:, np.newaxis]

    # Scale up so it properly resembles something inferred from this dataset
    # (this avoids catastrophic underflow in softmax)
    wordDists *= data.word_count / K

    # The weight vector
    weights = np.ones((K, 1))

    # Count of dummy negative observations. Assume that for every
    # twp papers cited, 1 was considered and discarded
    if pseudoNegCount is None:
        pseudoNegCount = 0.5 * np.mean(data.links.sum(axis=1).astype(DTYPE))

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

    # Now assign a topic to
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
    result  = queryState.topicDists[:, :K]
    norm    = np.sum(result, axis=1)
    result /= norm[:, np.newaxis]

    return result

#@nb.jit
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

#@nb.jit
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


#@nb.jit
def _update_topics_at_d(d, W, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums):
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
    wordIdx, z = _infer_topics_at_d(d, W, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums)
    topicMeans[d, :K] = np.dot(z, W[d, :].data) / docLens[d]
    return wordIdx, z

#@nb.jit
def _infer_topics_at_d(d, W, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums):
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

    wordIdx = W[d, :].indices
    z  = diWordDists[:, wordIdx]
    z -= diWordDistSums[:, np.newaxis]

    distAtD = (topicPrior + docLens[d] * topicMeans[d, :])[:K, np.newaxis]

    z += fns.digamma(distAtD)
    z -= fns.digamma(distAtD.sum())

    _inplace_softmax_colwise(z)
    return wordIdx, z


#@nb.jit
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
        printAndFlushNoNewLine("\n %4d: " % itr)

        diWordDistSums[:] = wordDists.sum(axis=1)
        fns.digamma(diWordDistSums, out=diWordDistSums)
        fns.digamma(wordDists,      out=diWordDists)

        if updateVocab:
            wordDists[:, :] = vocabPrior
            for d in range(D):
                if d % 100 == 0:
                    printAndFlushNoNewLine(".")
                wordIdx, z = _update_topics_at_d(d, W, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums)
                wordDists[:, wordIdx] += W[d, :].data[np.newaxis, :] * z

            if True: # itr % logFrequency == 0:
                iters.append(itr)
                bnds.append(_var_bound_internal(data, model, query))
                likes.append(_log_likelihood_internal(data, model, query))

                if len(bnds) > 6 and ".3f" % bnds[-1] == ".3f" % bnds[-2]:
                    break

        else:
            for d in range(D):
                wordIdx, z = _update_topics_at_d(d, W, docLens, topicMeans, topicPrior, diWordDists, diWordDistSums)

    topicMeans = _convertMeansToDirichletParam(docLens, topicMeans, topicPrior)

    return ModelState(K, topicPrior, vocabPrior, wordDists, weights, negCount, reg, dtype, model.name), \
           QueryState(docLens, topicMeans), \
           (np.array(iters, dtype=np.int32), np.array(bnds), np.array(likes))


def printAndFlushNoNewLine (text):
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
    _, topics, (_,_,_) =  train(data, model, query, plan, updateVocab=False)
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
    K, topicPrior, wordPrior, wordDists, weights, negCount, reg, dtype = \
        model.K, model.topicPrior, model.vocabPrior, model.wordDists, model.weights, model.pseudoNegCount, model.regularizer, model.dtype
    docLens, topicDists = \
        query.docLens, query.topicDists

    # Initialize z matrix if necessary
    W,X = data.words, data.links
    D,T = W.shape
        
    # Perform the digamma transform for E[ln \theta] etc.
    topicDists      = topicDists.copy()
    diTopicDists    = fns.digamma(topicDists[:, :K])
    diSumTopicDists = fns.digamma(topicDists[:, :K].sum(axis=1))
    diWordDists     = fns.digamma(model.wordDists)
    diSumWordDists  = fns.digamma(model.wordDists.sum(axis=1))

    print("")
    pad = "       "

    # E[ln p(topics|topicPrior)] according to q(topics)
    #
    prob_topics = D * (fns.gammaln(topicPrior[:K].sum()) - fns.gammaln(topicPrior[:K]).sum()) \
        + np.sum((topicPrior[:K] - 1)[np.newaxis, :] * (diTopicDists - diSumTopicDists[:, np.newaxis]))

    bound += prob_topics
    print(pad + "E[ln p(topics|topicPrior)] = %.3f" % prob_topics)

    # and its entropy
    ent_topics = _dirichletEntropy(topicDists[:, :K])
    bound += ent_topics
    print(pad + "H[q(topics)]                = %.3f" % ent_topics)
        
    # E[ln p(vocabs|vocabPrior)]
    #
    if type(model.vocabPrior) is float or type(model.vocabPrior) is int:
        prob_vocabs  = K * (fns.gammaln(wordPrior * T) - T * fns.gammaln(wordPrior)) \
               + np.sum((wordPrior - 1) * (diWordDists - diSumWordDists[:,np.newaxis] ))
    else:
        prob_vocabs  = K * (fns.gammaln(wordPrior.sum()) - fns.gammaln(wordPrior).sum()) \
               + np.sum((wordPrior - 1)[np.newaxis,:] * (diWordDists - diSumWordDists[:,np.newaxis] ))

    bound += prob_vocabs
    print(pad + "E[ln p(vocabs|vocabPrior)]  = %.3f" % prob_vocabs)

    # and its entropy
    ent_vocabs = _dirichletEntropy(wordDists)
    bound += ent_vocabs
    print(pad + "H[q(vocabs)                 = %.3f " % ent_vocabs)

    # P(z|topic) is tricky as we don't actually store this. However
    # we make a single, simple estimate for this case.
    # NOTE COPY AND PASTED FROM iterate_f32  / iterate_f64 (-ish)
    topicMeans = _convertDirichletParamToMeans(docLens, topicDists, topicPrior)

    prob_words = 0
    prob_z     = 0
    ent_z      = 0
    for d in range(D):
        wordIdx, z = _infer_topics_at_d(d, W, docLens, topicMeans, topicPrior, diWordDists, diSumWordDists)

        # E[ln p(Z|topics) = sum_d sum_n sum_k E[z_dnk] E[ln topicDist_dk]
        exLnTopic = diTopicDists[d, :K] - diSumTopicDists[d]
        prob_z += np.dot(z * exLnTopic[:, np.newaxis], W[d, :].data).sum()

        # E[ln p(W|Z)] = sum_d sum_n sum_k sum_t E[z_dnk] w_dnt E[ln vocab_kt]
        prob_words += np.sum(W[d, :].data[np.newaxis, :] * z * (diWordDists[:, wordIdx] - diSumWordDists[:, np.newaxis]))
        
        # And finally the entropy of Z
        ent_z -= np.dot(z * safe_log(z), W[d, :].data).sum()

    bound += (prob_z + ent_z + prob_words)

    print(pad + "E[ln p(Z|topics)            = %.3f" % prob_z)
    print(pad + "H[q(Z)]                     = %.3f" % ent_z)
    print(pad + "E[ln p(W|Z,vocabs)          = %.3f" % prob_words)


    # Next, the distribution over links - we just focus on the positives in this case
    # for d in range(D):
    #     links   = links_up_to(d, X)
    #     scales  = topicMeans[links,:].dot(weights * topicMeans[d])
    #     probs   = compiled.probit(scales)
    #     lnProbs = np.log(probs)
    #
    #     # expected probability of all links from d to p < d such that y_dp = 1
    #     bound += lnProbs
    #
    #     # and the entropy
    #     bound     -= np.sum(probs * lnProbs)
    #     probs[:]   = 1 - probs
    #     lnProbs[:] = np.log(probs)
    #     bound     -= np.sum(probs * lnProbs)

    topicDists = _convertMeansToDirichletParam(docLens, topicMeans, topicPrior)
    print(pad + "Bound                       = %.3f" % bound)
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


def links_up_to (d, X):
    '''
    Gets all the links that exist to earlier documents in the corpus. Ensures
    that if we iterate through all documents, we only ever consider each link
    once.
    '''
    return links_up_to_csr(d, X.indptr, X.indices)


nb.jit
def links_up_to_csr(d, Xptr, Xindices):
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


@nb.jit
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
    topicMeans = topics.topicDists
    D = topicMeans.shape[0]

    # derive topic means from the topic distributions
    # note there is a risk of loss of precision in all this which I just accept
    topicPriorExt = extend_topic_prior(model.topicPrior, 0)
    topicMeans -= topicPriorExt[np.newaxis,:]
    topicMeans /= topics.docLens[:, np.newaxis]

    # use the topics means to predict links
    mins = np.ones((D,), dtype=model.dtype)
    for d in range(D):
        probs = (topicMeans[d] * topicMeans[links[d,:].indices]).dot(weights)
        mins[d] = compiled.probit(probs).min()

    # return from topic means to the topic distributions
    topicMeans *= topics.docLens[:, np.newaxis]
    topicMeans += topicPriorExt[np.newaxis,:]

    return mins

def extend_topic_prior (prior_vec, extra_field):
    return np.hstack ((prior_vec, extra_field))

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
    topicMeans = topics.topicDists
    D = topicMeans.shape[0]

    # We build the result up as a COO matrix
    rows = []
    cols = []
    vals = []

    # derive topic means from the topic distributions
    # note there is a risk of loss of precision in all this which I just accept
    topicPriorExt = extend_topic_prior(model.topicPrior, 0)
    topicMeans -= topicPriorExt[np.newaxis,:]
    topicMeans /= topics.docLens[:, np.newaxis]

    # use the topics means to predict links
    mins = np.ones((D,), dtype=model.dtype)
    for d in range(D):
        probs = (topicMeans[d] * topicMeans).dot(weights)
        probs = compiled.probit(probs)
        relevant = np.where(probs >= mins[d])[0]
        print ("Non-neglible links: %d / %d" % (len(relevant), D))

        rows.extend[[d] * len(relevant)]
        cols.extend(relevant)
        vals.extend(probs[relevant])

    # return from topic means to the topic distributions
    topicMeans *= topics.docLens[:, np.newaxis]
    topicMeans += topicPriorExt[np.newaxis,:]

    # Build the COO matrix, then covert it to CSR. Converts lists to numpy
    # arrays to ensure appropriate dtypes
    r = np.array(rows, dtype=np.int32)
    c = np.array(cols, dtype=np.int32)
    v = np.array(vals, dtype=model.dtype)

    return ssp.coo_matrix((v (r, c)), shape=(D,D)).tocsr()


if __name__ == '__main__':
    test = np.array([-1, 3, 5, -4 , 4, -3, 1], dtype=np.float64)
    print (str (compiled.normpdf(test)))
