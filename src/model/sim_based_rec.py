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


import model.lda_vb_python as lda
LDA="lda/vb"

# import model.gibbs as lda
# LDA="lda/gibbs"

# ==============================================================
# CONSTANTS
# ==============================================================

DTYPE=np.float32 # A default, generally we should specify this in the model setup

DEBUG=False

MODEL_NAME_PREFIX="sim/"
TF_IDF="tfidf"

SQRT_TWO=1.414213562


# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')

QueryState = namedtuple ( \
    'QueryState', \
    'reps ldaTopics'\
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


def newQueryState(_, model, ldaTopics=None):
    '''
    Creates a new LRO QueryState object. This contains all
    parameters and random variables tied to individual
    datapoints.

    Param:
    :param _:  the dataset, not used
    :param model:  the model, not used
    :param ldaQuery: the model state object

    Return:
    A QueryState object
    '''

    return QueryState(None, ldaTopics)


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
    ldaTopics = query.ldaTopics

    D, K = data.doc_count, ldaModel.K

    # Step 1: Learn the topics using vanilla LDA
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
            _, ldaTopics = lda.query(data, ldaModel, lda.newQueryState(data, ldaModel), plan)
        elif ldaTopics is None or not ldaTopics.processed:
            ldaModel, ldaTopics, (_, _, _) = lda.train(data, ldaModel, lda.newQueryState(data, ldaModel), plan)
        reps = np.sqrt(lda.topicDists(ldaTopics))
    else:
        raise ValueError("Unknown method %s" % method)


    return ModelState(ldaModel, K, method, dtype, modelName), \
           QueryState(reps, ldaTopics), \
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
    :return: a vector with the minimum out-link probabilties for each
        document in the subset
    '''
    if model.method == TF_IDF:
        return min_link_probs_tfidf(model, train_tops, query_tops, links, docSubset)
    elif model.method == LDA:
        return min_link_probs_lda(model, train_tops, query_tops, links, docSubset)
    else:
        raise ValueError ("Unknown method %s " %  model.method)

def min_link_probs_tfidf(model, train_tops, query_tops, query_links, docSubset):
    assert model.method == TF_IDF
    src_reps = query_tops.reps
    dst_reps = train_tops.reps
    if docSubset is None:
        docSubset = [q for q in range(src_reps.shape[0])]
    Q = len(docSubset)

    src_norms = csr_row_norms(src_reps) # Numpy 1.9 allows la.norm(X, axis=1), but is too modern
    dst_norms = csr_row_norms(dst_reps)
    mins  = np.empty((Q,), dtype=model.dtype)
    outRow = -1
    for src in docSubset:
        outRow += 1

        probs = []
        for i in range(len(query_links[src,:].indices)): # For each destination doc
            dst = query_links[src,:].indices[i]          # which we denote dst
            linkProbMat = src_reps[src,:].dot(dst_reps[dst,:].T)
            linkProb = 0 if linkProbMat.nnz == 0 else linkProbMat.data[0]
            linkProb /= src_norms[src] * dst_norms[dst]
            probs.append(linkProb)
        mins[outRow] = min(probs) if len(probs) > 0 else -1

    return mins


def min_link_probs_lda(model, train_tops, query_tops, query_links, docSubset):
    assert model.method == LDA
    src_reps = query_tops.reps
    dst_reps = train_tops.reps
    if docSubset is None:
        docSubset = [q for q in range(src_reps.shape[0])]
    Q = len(docSubset)

    print ("Inferring minimal link probabilities (Similarity/LDA) ")
    mins = np.empty((Q,), dtype=model.dtype)
    outRow = -1
    for src in docSubset:
        outRow += 1
        probs = []
        for i in range(len(query_links[src,:].indices)): # For each query target link
            dst = query_links[src,:].indices[i]          # which we denote dst
            diffNorm = la.norm(src_reps[src] - dst_reps[dst])
            linkProb = SQRT_TWO / max(diffNorm, 1E-30)
            probs.append(linkProb)
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
    :param docSubset: a list of documents to consider for evaluation. If
    none all documents are considered.
    :return: a (hopefully) sparse len(docSubset)xD matrix of link probabilities
    '''
    if model.method == TF_IDF:
        return link_probs_tfidf(model, train_tops, query_tops, min_link_probs, docSubset)
    elif model.method == LDA:
        return link_probs_lda(model, train_tops, query_tops, min_link_probs, docSubset)
    else:
        raise ValueError ("Unknown method %s " %  model.method)


def link_probs_tfidf(model, train_tops, query_tops, min_link_probs, docSubset=None):
    '''
    This is essentially just the cosine similarity between the TF-IDF
    scores of all possible pairs. This ranges from zero to one, with
    one indicating the pairs are identical
    '''
    assert model.method == TF_IDF
    src_reps = query_tops.reps
    dst_reps = train_tops.reps

    # Determine the size of the output
    D = dst_reps.shape[0]
    if docSubset is None:
        docSubset = [q for q in range(src_reps.shape[0])]
    Q = len(docSubset)

    # We build the result up as a COO matrix
    rows = []
    cols = []
    vals = []

    # Infer the link probabilities
    outRow = -1
    src_norms = csr_row_norms(src_reps) # Numpy 1.9 allows la.norm(X, axis=1), but is too modern
    dst_norms = csr_row_norms(dst_reps)
    for src in docSubset:
        outRow += 1

        probs = np.squeeze(np.asarray(dst_reps.dot(src_reps[src,:].T).todense()))
        probs /= dst_norms
        probs /= src_norms[src]

        relevant = np.where(probs >= min_link_probs[outRow] - 1E-9)[0]

        rows.extend([outRow] * len(relevant))
        cols.extend(relevant)
        vals.extend(probs[relevant])

    # Build the COO matrix, then covert it to CSR. Converts lists to numpy
    # arrays to ensure appropriate dtypes
    r = np.array(rows, dtype=np.int32)
    c = np.array(cols, dtype=np.int32)
    v = np.array(vals, dtype=model.dtype)

    return ssp.coo_matrix((v, (r, c)), shape=(Q, D)).tocsr()


def csr_row_norms(X):
    norm_dat = np.abs(X.data * X.data)

    X2 = ssp.csr_matrix((norm_dat, X.indices, X.indptr), shape=X.shape)

    norms = np.squeeze(np.array(X2.sum(axis=1)))
    norms **= 0.5

    return norms

def link_probs_lda(model, train_tops, query_tops, min_link_probs, docSubset=None):
    assert model.method == LDA
    src_reps = query_tops.reps
    dst_reps = train_tops.reps

    # Determine the size of the output
    D = dst_reps.shape[0]
    if docSubset is None:
        docSubset = [q for q in range(src_reps.shape[0])]
    Q = len(docSubset)

    # We build the result up as a COO matrix
    rows = []
    cols = []
    vals = []

    # Infer the link probabilities
    print ("Inferring link probabilities (Similarity/LDA) ")
    outRow = -1
    for src in docSubset:
        outRow += 1
        src_rep = src_reps[src,:]

        # Helliger distance
        probs = src_rep[np.newaxis,:] - dst_reps
        probs *= probs
        probs = probs.sum(axis=1)
        np.sqrt(probs, out=probs)
        probs[probs < 1E-30] = 1E-30
        probs = SQRT_TWO / probs

        relevant = np.where(probs >= min_link_probs[outRow] - 1E-9)[0]

        rows.extend([outRow] * len(relevant))
        cols.extend(relevant)
        vals.extend(probs[relevant])

    # Build the COO matrix, then covert it to CSR. Converts lists to numpy
    # arrays to ensure appropriate dtypes
    r = np.array(rows, dtype=np.int32)
    c = np.array(cols, dtype=np.int32)
    v = np.array(vals, dtype=model.dtype)

    return ssp.coo_matrix((v, (r, c)), shape=(Q, D)).tocsr()