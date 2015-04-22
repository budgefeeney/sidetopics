'''
Created on 15 Apr 2015

@author: bryanfeeney
'''

import numpy as np
import numpy.random as rd
import scipy.sparse as ssp
import scipy.special as fns
import numba as nb

import model.rtm_fast as compiled
from util.sparse_elementwise import sparseScalarProductOfSafeLnDot
from util.overflow_safe import safe_log
from util.misc import constantArray, converged, clamp
from model.lda_cvb import toWordList

from collections import namedtuple

MODEL_NAME = "rtm/vb"
DTYPE      = np.float64

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')

QueryState = namedtuple ( \
    'QueryState', \
    'W_list docLens topicDists'\
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
        topicPrior = constantArray((K,), 50.0 / K, dtype) # From Griffiths and Steyvers 2004
    if vocabPrior is None:
        vocabPrior = 0.1 # Also from G&S

    vocabPriorVec = constantArray((T,), vocabPrior, dtype)
    wordDists = rd.dirichlet(vocabPriorVec, size=K).astype(dtype)

    # Peturb to avoid zero probabilities
    wordDists += 1./T
    wordDists /= (wordDists.sum(axis=1))[:,np.newaxis]

    # The weight vector
    weights = np.ones ((K,1))

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
    K = modelState.K
    D,_ = data.words.shape
    
    print("Converting Bag of Words matrix to List of List representation... ", end="")
    W_list, docLens = toWordList(data.words)
    print("Done")

    # Initialise the per-token assignments at random according to the dirichlet hyper
    # This is super-slow
    topicPriorExt = np.hstack(modelState.topicPrior, 0)
    topicDists = rd.dirichlet(topicPriorExt, size=D).astype(modelState.dtype)
    topicDists[:,K] = docLens

    # Now assign a topic to
    return QueryState(W_list, docLens, topicDists)


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
    result = modelState.wordDists
    norm   = np.sum(modelState.wordDists, axis=1)
    result /= norm[:,np.newaxis]

    return result


def topicDists (queryState):
    '''
    The D x K matrix of topics distributions inferred for the K topics
    across all D documents
    '''
    K       = queryState.topicDists.shape[1] - 1
    result  = queryState.topicDists[:,:K]
    norm    = np.sum(result, axis=1)
    result /= norm[:,np.newaxis]

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



def train(data, model, query, plan):
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
    W_list, docLens, topicDists = \
        query.W_list, query.docLens, query.topicDists
    K, topicPrior, vocabPrior, wordDists, weights, negCount, reg, dtype = \
        model.K, model.topicPrior, model.vocabPrior, model.wordDists, model.weights, model.pseudoNegCount, model.regularizer, model.dtype

    W   = data.words
    D,T = W.shape
    X   = data.links

    # Quick sanity check
    if np.any(docLens < 1):
        raise ValueError ("Input document-term matrix contains at least one document with no words")
    assert dtype == np.float64, "Only implemented for 64-bit floats"

    # Book-keeping for logs
    logPoints    = 1 if logFrequency == 0 else iterations // logFrequency
    boundIters   = np.zeros(shape=(logPoints,))
    boundValues  = np.zeros(shape=(logPoints,))
    likelyValues = np.zeros(shape=(logPoints,))
    bvIdx = 0

    # Instead of storing the full topic assignments for every individual word, we
    # re-estimate from scratch. I.e for the memberships z which is DxNxT in dimension,
    # we only store a 1xNxT = NxT part.
    z_dnk = np.empty((docLens.max(), K), dtype=dtype, order='F')

    do_iterations = compiled.iterate_f64

    # Iterate in segments, pausing to take measures of the bound / likelihood
    segIters  = logFrequency
    remainder = iterations - segIters * (logPoints - 1)
    totalItrs = 0
    for segment in range(logPoints - 1):
        print ("Starting training")
        totalItrs += do_iterations (segIters, D, K, T, \
                 W_list, docLens, \
                 topicPrior, vocabPrior, \
                 z_dnk, topicDists, wordDists)

        boundIters[bvIdx]   = segment * segIters
        boundValues[bvIdx]  = var_bound(W, X, model, query, z_dnk)
        likelyValues[bvIdx] = log_likelihood(W, X, model, query)
        bvIdx += 1

        if converged (boundIters, boundValues, bvIdx, epsilon, minIters=5):
            boundIters, boundValues, likelyValues = clamp (boundIters, boundValues, likelyValues, bvIdx)
            return ModelState(K, topicPrior, vocabPrior, wordDists, weights, negCount, reg, dtype, model.name), \
                QueryState(W_list, docLens, topicDists), \
                (boundIters, boundValues, likelyValues)

        print ("Segment %d/%d Total Iterations %d Bound %10.2f Likelihood %10.2f" % (segment, logPoints, totalItrs, boundValues[bvIdx - 1], likelyValues[bvIdx - 1]))

    # Final batch of iterations.
    do_iterations (remainder, D, K, T, \
                 W_list, docLens, \
                 topicPrior, vocabPrior, \
                 z_dnk, topicDists, wordDists)

    boundIters[bvIdx]   = iterations - 1
    boundValues[bvIdx]  = var_bound(W, X, model, query, z_dnk)
    likelyValues[bvIdx] = log_likelihood(W, X, model, query)


    return ModelState(K, topicPrior, vocabPrior, wordDists, weights, negCount, reg, dtype, model.name), \
           QueryState(W_list, docLens, topicDists), \
           (boundIters, boundValues, likelyValues)


def var_bound(data, model, query, z_dnk = None):
    '''
    Determines the variational bounds.
    '''
    bound = 0
    
    # Unpack the the structs, for ease of access and efficiency
    K, topicPrior, wordPrior, wordDists, weights, negCount, reg, dtype = \
        model.K, model.topicPrior, model.vocabPrior, model.wordDists, model.weights, model.pseudoNegCount, model.regularizer, model.dtype
    W_list, docLens, topicDists = \
        query.W_list, query.docLens, query.topicDists

    # Initialize z matrix if necessary
    W,X = data.words, data.links
    D,T = W.shape
    maxN = docLens.max()
    if z_dnk is None:
        z_dnk = np.empty(shape=(maxN, K), dtype=dtype)
        
    # Perform the digamma transform for E[ln \theta] etc.
    diTopicDists    = fns.digamma(topicDists[:,:K])
    diSumTopicDists = fns.digamma(topicDists[:,:K].sum(axis=1))
    diWordDists     = fns.digamma(model.wordDists)
    diSumWordDists  = fns.digamma(model.wordDists.sum(axis=1))
    
    # P(topics|topicPrior)
    #
    bound += D * fns.gammaln(topicPrior.sum()) - fns.gammaln(topicPrior).sum() \
           + np.sum((topicPrior - 1)[np.newaxis,:] * (diTopicDists - diSumTopicDists[:,np.newaxis] ))
    
    # and its entropy
    bound += dirichletEntropy(topicDists[:,:K])
        
    # P(vocabs|vocabPrior)
    #
    if type(model.vocabPrior) is float:
        bound += D * fns.gammaln(wordPrior * T) - T * fns.gammaln(wordPrior) \
               + np.sum((wordPrior - 1) * (diWordDists - diSumWordDists[:,np.newaxis] ))
    else:
        bound += D * fns.gammaln(wordPrior.sum()) - fns.gammaln(wordPrior).sum() \
               + np.sum((wordPrior - 1)[np.newaxis,:] * (diWordDists - diSumWordDists[:,np.newaxis] ))
    
    # and its entropy
    bound += dirichletEntropy(wordDists)

    # P(z|topic) is tricky as we don't actually store this. However
    # we make a single, simple estimate for this case.
    # NOTE COPY AND PASTED FROM iterate_f32  / iterate_f64 (-ish)
    for d in range(D):
        z_dnk[0:docLens[d],:] = diWordDists.T[W_list[d,0:docLens[d]],:] \
                              + diTopicDists[d,:]
        
        # We've been working in (expected) log-space till now, before we
        #  go to true probability space rescale so we don't underflow everywhere
        maxes  = z_dnk.max(axis=1)
        z_dnk -= maxes[:,np.newaxis]
        np.exp(z_dnk, out=z_dnk)
        
        # Now normalize so probabilities sum to one
        sums   = z_dnk.sum(axis=1)
        z_dnk /= sums[:,np.newaxis]
        
        # Now use to calculate  E[ln p(Z|topics), E[ln p(W|Z)
        bound += np.sum(z_dnk[0:docLens[d],:] * (diTopicDists[d,:] - diSumTopicDists[d]) )
        bound += np.sum(z_dnk[0:docLens[d],:].T * (diWordDists[:,W_list[d,0:docLens[d]]] - diSumWordDists[:,np.newaxis]))
        
        # And finally the entropy of Z
        bound -= np.sum(z_dnk[0:docLens[d],:] * safe_log(z_dnk[0:docLens[d],:]))
    
    # Next, the distribution over links - we just focus on the positives in this case
    topicPriorExt = extend_topic_prior(topicPrior, 0)
    topicDists -= topicPriorExt[np.newaxis,:]
    topicDists /= docLens[d] # Turns this into a topic means array
    for d in range(D):
        links   = links_up_to(d, X)
        scales  = topicDists[links,:].dot(weights * topicDists[d])
        probs   = compiled.probit(scales)
        lnProbs = np.log(probs)
        
        # expected probability of all links from d to p < d such that y_dp = 1
        bound += lnProbs
        
        # and the entropy
        bound     -= np.sum(probs * lnProbs)
        probs[:]   = 1 - probs
        lnProbs[:] = np.log(probs)
        bound     -= np.sum(probs * lnProbs)
        
    
    topicDists *= docLens[d] # Turns this back into a regularized topic distributions array
    topicDists += topicPriorExt[np.newaxis,:]
    return bound


def dirichletEntropy (P):
    '''
    Entropy of D Dirichlet distributions, with dimension K, whose parameters
    are given by the DxK matrix P
    '''
    D,K    = P.shape
    psums  = P.sum(axis=1)
    lnB   = fns.gammaln(P).sum(axis = 1) - fns.gammaln(psums)
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


nb.autojit
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
