# -*- coding: utf-8 -*-
'''
A similarity based recommender for papers.

For a target document, this returns the documents which are most similar
to it.
'''

__author__ = 'bryanfeeney'

from collections import namedtuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as ssp

from util.sparse_elementwise import sparseScalarQuotientOfDot

import model.lda_vb_python as lda
LDA="/lda/vb"

# import model.gibbs as lda
# LDA="/lda/gibbs"

# ==============================================================
# CONSTANTS
# ==============================================================

DTYPE=np.float32 # A default, generally we should specify this in the model setup

DEBUG=False

MODEL_NAME_PREFIX="sim/"
TF_IDF="/tfidf"

SQRT_TWO=1.414213562


# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')

QueryState = namedtuple ( \
    'QueryState', \
    'reps'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'ldaModel K method dtype name'
)

# ==============================================================
# PUBLIC API
# ==============================================================


def is_undirected_link_predictor():
    return False

def newModelFromExisting(model, withLdaModel=None):
    '''
    Creates a _deep_ copy of the given model
    '''
    return ModelState(\
        withLdaModel \
            if withLdaModel is not None \
            else lda.newModelFromExisting(model.ldaModel), \
        model.K, \
        model.method, \
        model.dtype, \
        model.name)


def newModelAtRandom(data, K, method=TF_IDF, topicPrior=None, vocabPrior=lda.VocabPrior, ldaModel=None, dtype=DTYPE):
    '''
    Creates a new LRO ModelState for the given training set and
    the given number of topics. Everything is instantiated purely
    at random, except for vocabularies, which are seeded with
    random documents, to get a good starting point.

    :param data: the DataSet, must contain words and links.
    :param K:    the number of topics
    :param method: the method by which the documents will be compared, either their
    LDA topic distribution or their TF_IDF scores
    :param topicPrior: the prior over topics, either a scalar or a K-dimensional vector
    :param vocabPrior: the prior over vocabs, either a scalar or a T-dimensional vector
    :param dtype: the datatype to be used throughout.

    Return:
    A ModelState object
    '''
    assert K > 1,   "There must be at least two topics"
    assert K < 255, "There can be no more than 255 topics"
    D,T = data.words.shape
    Q,P = data.links.shape
    assert D == Q and Q == P, "Link matrix must be square and have same row-count as word-matrix"

    if ldaModel is None:
        ldaModel = lda.newModelAtRandom(data, K, topicPrior, vocabPrior, dtype)

    if method == TF_IDF:
        modelName = MODEL_NAME_PREFIX + TF_IDF
    elif method == LDA:
        modelName = MODEL_NAME_PREFIX + LDA
    else:
        raise ValueError("Incorrect method name")

    return ModelState(ldaModel, K, method, dtype, modelName)


def newQueryState(data, model, ldaQuery=None):
    '''
    Creates a new LRO QueryState object. This contains all
    parameters and random variables tied to individual
    datapoints.

    Param:
    :param data:  the dataset, must contain words and links.
    :param model: the model state object

    Return:
    A QueryState object
    '''

    return QueryState(None)


def newTrainPlan(iterations=500, epsilon=None, ldaIterations=None, ldaEpilson=1, logFrequency=10, fastButInaccurate=False, debug=False):
    '''
    Create a training plan determining how many iterations we
    process, how often we plot the results, how often we log
    the variational bound, etc.

    This only applies if the method is LDA. TF-IDF has no training to speak of.

    epsilon is oddly measured, we just evaluate the angle of the line segment between
    the last value of the bound and the current, and if it's less than the given angle,
    then stop.
    '''

    return TrainPlan(iterations, epsilon, logFrequency, fastButInaccurate, debug)


def train (data, model, query, trainPlan, isQuery=False):
    '''
    Infers the topic distributions in general, and specifically for
    each individual datapoint.

    Params:
    :param data: the dataset, must contain both words and links
    :param model: the actual model, which is modified in-place
    :param query: the query results - essentially all the "local" variables
            matched to the given observations
    :param trainPlan: how to execute the training process (e.g. iterations,
                 log-interval etc.)

    Return:
    An new modelstate and a new querystate object with the learnt parameters,
    and and a tuple of iteration, vb-bound measurement and log-likelhood
    measurement
    '''
    iterations, epsilon, logFrequency, fastButInaccurate, debug = \
        trainPlan.iterations, trainPlan.epsilon, trainPlan.logFrequency, trainPlan.fastButInaccurate, trainPlan.debug
    ldaModel, method, K, dtype, modelName = \
        model.ldaModel, model.method, model.K, model.dtype, model.name
    reps = query.reps

    D, K = data.doc_count, ldaModel.K

    # Step 1: Learn the topics using vanilla LDA
    ldaQuery = lda.newQueryState(data, ldaModel)
    if method == TF_IDF:
        # First do TF
        docLens = np.squeeze(np.array(data.words.sum(axis=1)))
        reps = data.words.copy()
        #reps /= docLens[:, np.newaxis] replaced with line below to retain sparsity
        reps = ssp.diags(np.reciprocal(docLens), 0).dot(reps)

        occ  = data.words.astype(np.bool).astype(dtype)
        docCount = np.squeeze(np.array(occ.sum(axis=0)))
        docCount += 1
        idf = np.log(D / docCount)

        # reps *= idf[np.newaxis, :]
        reps = reps.dot(ssp.diags(idf, 0))
    elif method == LDA:
        plan = lda.newTrainPlan(iterations, logFrequency=logFrequency, debug=debug)
        if isQuery:
            ldaQuery = lda.query(data, ldaModel, ldaQuery, plan)
        elif not ldaModel.processed:
            ldaModel, ldaQuery, (_, _, _) = lda.train(data, ldaModel, ldaQuery, plan)
        reps = np.sqrt(lda.topicDists(ldaQuery))
    else:
        raise ValueError("Unknown method %s" % method)


    return ModelState(ldaModel, K, method, dtype, modelName), \
           QueryState(reps), \
           ([0], [0], [0])


def query(data, model, query, plan):
    '''
    Given a _trained_ model, attempts to predict the topics and topic offsets
    for each of the given inputs.

    Params:
    :param data:  the dataset of words, features and links of which only words are used in this model
    :param model: the _trained_ model
    :param query: the query state generated for the query dataset
    :param plan:  used in this case as we need to tighten up the approx

    Returns:
    The model state and query state, in that order. The model state is
    unchanged, the query is.
    '''
    _, query, (_, _, _) = train(data, model, query, plan, isQuery=True)

    return model, query


def log_likelihood (data, model, query):
    '''
    Return the log-likelihood of the given data W according to the model
    and the parameters inferred for the entries in W stored in the
    queryState object.

    This deliberately excludes links
    '''
    if model.method == TF_IDF:
        return 0
    elif model.method == LDA:
        return lda.log_likelihood(data, model.ldaModel, query=None, topicDistOverride=query.reps ** 2)
    else:
        raise ValueError ("Unknown method %s " %  model.method)


def var_bound(data, modelState, queryState):
    '''
    Determines the variational bounds. Values are mutated in place, but are
    reset afterwards to their initial values. So it's safe to call in a serial
    manner.
    '''
    return 0


def min_link_probs(model, query, links, docSubset=None):
    '''
    For every document, for each of the given links, determine the
    probability of the least likely link (i.e the document-specific
    minimum of probabilities).

    :param model: the model object
    :param query: the query state object, contains topics and topic
    offsets
    :param links: a DxD matrix of links for each document (row)
    :param docSubset: a list of documents to consider for evaluation. If
    none all documents are considered.
    :return: a vector with the minimum out-link probabilties for each
        document in the subset
    '''
    if model.method == TF_IDF:
        return min_link_probs_tfidf(model, query, links, docSubset)
    elif model.method == LDA:
        return min_link_probs_lda(model, query, links, docSubset)
    else:
        raise ValueError ("Unknown method %s " %  model.method)

def min_link_probs_tfidf(model, query, links, docSubset):
    assert model.method == TF_IDF
    reps = query.reps
    if docSubset is None:
        docSubset = [d for d in range(reps.shape[0])]
    D = len(docSubset)

    norms = csr_row_norms(reps) # Numpy 1.9 allows la.norm(X, axis=1), but is too modern
    mins  = np.empty((D,), dtype=model.dtype)
    outRow = -1
    for d in range(D):
        outRow += 1
        probs = []
        for i in range(len(links[d,:].indices)): # For each observed link
            l = links[d,:].indices[i]            # which we denote l
            linkProbMat = reps[d,:].dot(reps[l,:].T)
            linkProb = 0 if linkProbMat.nnz == 0 else linkProbMat.data[0]
            linkProb /= norms[d] * norms[l]
            probs.append(linkProb)
        mins[outRow] = min(probs) if len(probs) > 0 else -1

    return mins


def min_link_probs_lda(model, query, links, docSubset):
    assert model.method == LDA
    reps = query.reps
    if docSubset is None:
        docSubset = [d for d in range(reps.shape[0])]
    D = len(docSubset)

    print ("Inferring minimal link probabilities (Similarity/LDA) ")
    mins = np.empty((D,), dtype=model.dtype)
    outRow = -1
    for d in range(D):
        outRow += 1
        probs = []
        for i in range(len(links[d,:].indices)): # For each observed link
            l = links[d,:].indices[i]            # which we denote l
            diffNorm = la.norm(reps[d] - reps[l])
            linkProb = SQRT_TWO / max(diffNorm, 1E-30)
            probs.append(linkProb)
        mins[outRow] = min(probs) if len(probs) > 0 else -1

    return mins


def link_probs(model, query, min_link_probs, docSubset):
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
    :param docSubset: a list of documents to consider for evaluation. If
    none all documents are considered.
    :return: a (hopefully) sparse len(docSubset)xD matrix of link probabilities
    '''
    if model.method == TF_IDF:
        return link_probs_tfidf(model, query, min_link_probs, docSubset)
    elif model.method == LDA:
        return link_probs_lda(model, query, min_link_probs, docSubset)
    else:
        raise ValueError ("Unknown method %s " %  model.method)


def link_probs_tfidf(model, query, min_link_probs, docSubset=None):
    '''
    This is essentially just the cosine similarity between the TF-IDF
    scores of all possible pairs. This ranges from zero to one, with
    one indicating the pairs are identical
    '''
    assert model.method == TF_IDF
    reps = query.reps

    # Determine the size of the output
    actualDocCount = reps.shape[0]
    if docSubset is None:
        docSubset = [d for d in range(actualDocCount)]
    D = len(docSubset)

    # We build the result up as a COO matrix
    rows = []
    cols = []
    vals = []

    # Infer the link probabilities
    outRow = -1
    norms = csr_row_norms(reps) # Numpy 1.9 allows la.norm(X, axis=1), but is too modern
    for d in range(D):
        outRow += 1
        inrep = reps[d,:]

        probs = reps.dot(inrep.T).todense().data
        probs /= norms
        probs /= norms[d]

        relevant   = np.where(probs >= min_link_probs[d] - 1E-9)[0]

        rows.extend([outRow] * len(relevant))
        cols.extend(relevant)
        vals.extend(probs[relevant])

    # Build the COO matrix, then covert it to CSR. Converts lists to numpy
    # arrays to ensure appropriate dtypes
    r = np.array(rows, dtype=np.int32)
    c = np.array(cols, dtype=np.int32)
    v = np.array(vals, dtype=model.dtype)

    return ssp.coo_matrix((v, (r, c)), shape=(D, actualDocCount)).tocsr()


def csr_row_norms(X):
    norm_dat = np.abs(X.data * X.data)

    X2 = ssp.csr_matrix((norm_dat, X.indices, X.indptr), shape=X.shape)

    norms = np.squeeze(np.array(X2.sum(axis=1)))
    norms **= 0.5

    return norms

def link_probs_lda(model, query, min_link_probs, docSubset=None):
    assert model.method == LDA
    reps = query.reps

    # Determine the size of the output
    actualDocCount = reps.shape[0]
    if docSubset is None:
        docSubset = [d for d in range(actualDocCount)]
    D = len(docSubset)

    # We build the result up as a COO matrix
    rows = []
    cols = []
    vals = []

    # Infer the link probabilities
    print ("Inferring link probabilities (Similarity/LDA) ")
    outRow = -1
    for d in range(D):
        outRow += 1
        inrep = reps[d,:]

        # Helliger distance
        probs = inrep[np.newaxis,:] - reps
        probs *= probs
        probs = probs.sum(axis=1)
        np.sqrt(probs, out=probs)
        probs = SQRT_TWO / probs

        relevant = np.where(probs >= min_link_probs[d] - 1E-9)[0]

        rows.extend([outRow] * len(relevant))
        cols.extend(relevant)
        vals.extend(probs[relevant])

    # Build the COO matrix, then covert it to CSR. Converts lists to numpy
    # arrays to ensure appropriate dtypes
    r = np.array(rows, dtype=np.int32)
    c = np.array(cols, dtype=np.int32)
    v = np.array(vals, dtype=model.dtype)

    return ssp.coo_matrix((v, (r, c)), shape=(D, actualDocCount)).tocsr()