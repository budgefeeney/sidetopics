# -*- coding: utf-8 -*-
'''
Implements the model from "Modelling Document Citations with
Latent Random Offsets"

@author: bryanfeeney
'''

__author__ = 'bryanfeeney'


from collections import namedtuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as ssp

from util.sparse_elementwise import sparseScalarProductOfSafeLnDot

import model.lda_gibbs as lda

# ==============================================================
# CONSTANTS
# ==============================================================

DTYPE=np.float32 # A default, generally we should specify this in the model setup

DEBUG=False

MODEL_NAME="lro/vb"


# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'ldaPlan iterations epsilon logFrequency fastButInaccurate debug')

QueryState = namedtuple ( \
    'QueryState', \
    'ldaQuery offsetTopicDists'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'ldaModel K noiseVar predVar scale dtype name'
)

# ==============================================================
# PUBLIC API
# ==============================================================

def wordDists(model):
    return lda.wordDists(model.ldaModel)


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
        model.noiseVar, \
        np.array(model.predVar), \
        model.scale, \
        model.dtype, \
        model.name)


def newModelAtRandom(data, K, noiseVar=9, predVar=None, topicPrior=None, vocabPrior=lda.VocabPrior, ldaModel=None, dtype=DTYPE):
    '''
    Creates a new LRO ModelState for the given training set and
    the given number of topics. Everything is instantiated purely
    at random, except for vocabularies, which are seeded with
    random documents, to get a good starting point.

    :param data: the DataSet, must contain words and links.
    :param K:    the number of topics
    :noiseVar:   the noise variance determining offset size
    :predVar:    the various around predictions, a two element vector, the first
                 being the prediction noise when links re not observed, the second
                 when links are observed
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
    if predVar is None:
        predVar = np.array([0.01, 1])
    assert len(predVar) == 2
    scale = 1

    return ModelState(ldaModel, K, noiseVar, predVar, scale, dtype, MODEL_NAME)


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
    if ldaQuery is None:
        ldaQuery = lda.newQueryState(data, model.ldaModel)
    offsets  = np.zeros((data.doc_count, model.K))

    return QueryState(ldaQuery, offsets)


def newTrainPlan(iterations=100, epsilon=None, ldaIterations=None, ldaEpilson=1, logFrequency=10, fastButInaccurate=False, debug=False):
    '''
    Create a training plan determining how many iterations we
    process, how often we plot the results, how often we log
    the variational bound, etc.

    epsilon is oddly measured, we just evaluate the angle of the line segment between
    the last value of the bound and the current, and if it's less than the given angle,
    then stop.
    '''
    if ldaIterations is None:
        ldaIterations = 500 # iterations
    ldaPlan = lda.newTrainPlan(ldaIterations, ldaEpilson, logFrequency, fastButInaccurate, debug)

    return TrainPlan(ldaPlan, iterations, epsilon, logFrequency, fastButInaccurate, debug)


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
    ldaPlan, iterations, epsilon, logFrequency, fastButInaccurate, debug = \
        trainPlan.ldaPlan, trainPlan.iterations, trainPlan.epsilon, trainPlan.logFrequency, trainPlan.fastButInaccurate, trainPlan.debug
    ldaModel, noiseVar, predVar, scale, dtype = \
        model.ldaModel, model.noiseVar, model.predVar, model.scale, model.dtype
    ldaQuery, offsetTopicDists = \
        query.ldaQuery, query.offsetTopicDists

    D, K = data.doc_count, ldaModel.K
    epsilon = 0.01 * D * K if epsilon is None else epsilon
    tau = [predVar[0], predVar[1]]

    # Step 1: Learn the topics using vanilla LDA
    if isQuery and not ldaQuery.processed:
        ldaQuery = lda.query(data, ldaModel, ldaQuery, ldaPlan)
    elif not ldaModel.processed:
        ldaModel, ldaQuery, (_, _, _) = lda.train(data, ldaModel, ldaQuery, ldaPlan)

    tops = lda.topicDists(ldaQuery)
    offs = tops.copy()
    topsSum = tops.T.dot(tops)

    # Step 2: reverse the links matrix so we can talk about the origin (not target) of links
    inlinks = data.links.T.tocsr()

    # Step 3: Learn the scaling factor and offsets for each link's target-doc till converged
    print ("Learning Offsets")
    for itr in range(iterations):
        # Record the current scale of the offsets
        before = offs.sum()

        # Update the scale
        lhs, rhs = 0, 0
        for p in range(data.doc_count):
            lhs += (tau[1] - tau[0]) * (tops[inlinks[p,:].indices,:].dot(offs[p,:]) ** 2).sum()
            lhs += tau[0] * (offs[p,:].dot(topsSum).dot(offs[p,:]) - offs[p,:].dot(np.outer(tops[p,:],tops[p,:])).dot(offs[p,:]))
            rhs += tau[1] * tops[inlinks[p,:].indices,:].dot(offs[p,:]).sum()

        scale = rhs / lhs

        # Update the offset for every target doc
        for p in range(data.doc_count):
            lhs  = (tau[1] - tau[0]) * np.einsum("dj,k->jk", tops[inlinks[p,:].indices,:], tops[p,:])
            lhs += tau[0] * (np.einsum("dj,k->jk", tops, tops[p,:]) - np.outer(tops[p,:], tops[p,:]))
            lhs *= (scale * scale)
            lhs[np.diag_indices_from(lhs)] += noiseVar

            rhs  = tops[p,:] + scale * tau[1] * tops[inlinks[p,:].indices,:].sum(axis=0)

            offs[p,:] = la.inv(lhs).dot(rhs)

        # Check has the offsets changed significantly
        after = offs.sum()
        if abs(before - after) < epsilon:
            break


    return ModelState(ldaModel, K, noiseVar, predVar, scale, dtype, MODEL_NAME), \
           QueryState(ldaQuery, offs), \
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
    return lda.log_likelihood(data, model.ldaModel, query.ldaQuery)


def var_bound(data, modelState, queryState):
    '''
    Determines the variational bounds. Values are mutated in place, but are
    reset afterwards to their initial values. So it's safe to call in a serial
    manner.
    '''
    return 0


def min_link_probs(model, query, links):
    '''
    For every document, for each of the given links, determine the
    probability of the least likely link (i.e the document-specific
    minimum of probabilities).

    :param model: the model object
    :param query: the query state object, contains topics and topic
    offsets
    :param links: a DxD matrix of links for each document (row)
    :return: a D-dimensional vector with the minimum probabilties for each
        link
    '''
    scale = model.scale
    tops  = lda.topicDists(query.ldaQuery)
    offs  = query.offsetTopicDists
    D     = tops.shape[0]

    mins = np.empty((D,), dtype=model.dtype)
    for d in range(D):
        probs = []
        for i in range(len(links[d,:].indices)): # For each observed link
            l = links[d,:].indices[i]            # which we denote l
            linkProb = scale * tops[d,:].dot(offs[l,:])
            probs.append(linkProb)
        mins[d] = min(probs) if len(probs) > 0 else -1

    return mins


def link_probs(model, query, min_link_probs):
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
    scale = model.scale
    tops  = lda.topicDists(query.ldaQuery)
    offs  = query.offsetTopicDists
    D     = tops.shape[0]

    # We build the result up as a COO matrix
    rows = []
    cols = []
    vals = []

    # Infer the link probabilities
    for d in range(D):
        probs      = scale * offs.dot(tops[d,:])
        relevant   = np.where(probs >= min_link_probs[d] - 1E-9)[0]

        rows.extend([d] * len(relevant))
        cols.extend(relevant)
        vals.extend(probs[relevant])

    # Build the COO matrix, then covert it to CSR. Converts lists to numpy
    # arrays to ensure appropriate dtypes
    r = np.array(rows, dtype=np.int32)
    c = np.array(cols, dtype=np.int32)
    v = np.array(vals, dtype=model.dtype)

    return ssp.coo_matrix((v, (r, c)), shape=(D, D)).tocsr()