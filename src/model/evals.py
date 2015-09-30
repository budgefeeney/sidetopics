'''
Created on 20 Apr 2015

@author: bryanfeeney
'''

import numpy as np
#import numba as nb
from model.common import DataSet

Perplexity="perplexity"
MeanAveragePrecAllDocs="meanavgprec_all"
MeanPrecRecAtMAllDocs="meanprecrec_all"
LroMeanPrecRecAtMAllDocs="lro_meanprecrec_all"
LroMeanPrecRecAtMFeatSplit="lro_meanprecrec_featsplit"
HashtagPrecAtM="hashtag_prec_at_m"

EvalNames = [Perplexity, MeanAveragePrecAllDocs, MeanPrecRecAtMAllDocs, \
             HashtagPrecAtM, LroMeanPrecRecAtMAllDocs, LroMeanPrecRecAtMFeatSplit]

AllGroups = (-1, -1)

def perplexity_from_like(log_likely, token_count):
    if type(token_count) is DataSet:
        token_count = token_count.word_and_link_count
    return np.exp(-log_likely / token_count)


def word_perplexity(log_likely_fn, model, query, data):
    return perplexity_from_like(log_likely_fn(data, model, query), data.word_count)

#@nb.autojit
def mean_reciprocal_rank(expected_links, estim_link_probs):
    '''
    Another way of evaluating a ranking. A query out-link's reciprocal
    rank is just one over its rank in the list of returned out-links. We average
    these query ranks over all query out-links for all documents and return

    :param expected_links: the links whose rank we check
    :param estim_link_probs: the probabilities of (almost) all links, including
    all expected links, in a sparse CSR format.
    :return: the rank score, as a single double.
    '''
    D = expected_links.shape[0]
    rank_sum   = 0.0
    rank_count = 0

    docs_lacking_links = []
    for d in range(D):
        # Take out the indices (i.e. IDs) of the expected links
        expt_indices = [e for e in expected_links[d,:].indices]
        if len(expt_indices) == 0:
            docs_lacking_links.append(d)
            continue

        # Rank the received indices by associated value in descending order
        row = estim_link_probs[d,:]
        ind = (np.argsort(row.data))
        recv_indices = row.indices[ind[::-1]]

        # Sum up reciprocal ranks
        r = 0
        while r < len(recv_indices) and len(expt_indices) > 0:
            e = 0
            while e < len(expt_indices):
                if recv_indices[r] == expt_indices[e]:
                    del expt_indices[e]
                    rank_sum += 1./(r+1)
                    rank_count += 1
                e += 1
            r += 1

    if len(docs_lacking_links) > 0:
        print (str(len(docs_lacking_links)) + " of the " + str(D) + " documents had no links to check, including documents " + ", ".join(str(d) for d in docs_lacking_links[:10]))
    return rank_sum / rank_count


def mean_prec_rec_at(expected_links, estim_link_probs, at=None, groups=None):
    '''
    Returns the average, across all documents in the corpus, of the
    precision-at-m scores for each of the test documents, and the recall
    at-m scores for each of the test documents

    Precision at m is the precision of the top m documents, ranked
    by probability, returned by the algorithm.

    The precision is the number of of relevant documents returned
    as a proportion of all documents returned (up to M in this case)

    Recall is the number of relevant documents returned as a proportion
    of all possible relevant documents.

    :param expected_links: the links we expect to find
    :param estim_link_probs: the DxD sparse CSR matrix of link probabilities
    :param at: how far to go down to calculate precision, may be a list. Defaults
    to 100
    :param groups: break down precision at m further, by considering those subsets
    of documents that have the given range of documents. e.g for two segments, one
    less than 10, one between 10 (inclusive) and 15 (exclusive) and 15 or greater
    specify [(0,10), (10, 15), (15,1000)]. By default there is one value which
    includes everything. Also  documents with no outlinks are skipped always.
    :return: First a dictionary of tuples to lists, the tuples denoting how many links
    were in the documents considered, the lists being the precisions at m evaluated
    for each of the values in "at'. Secondly a similarly structured dictionary of
    recall at m, for every m and group. Thirdly a dictionary of those same groups to
    docCounts
    '''
    def find_group (outLinkCount, rangeTuples):
        i = 0
        while rangeTuples[i][0] == AllGroups[0] \
            or outLinkCount < rangeTuples[i][0] \
            or outLinkCount > rangeTuples[i][1]:
            i += 1
        return rangeTuples[i]

    if at is None:
        ms = [100]
    if type(at) is not list:
        ms = [at]
    else:
        ms = at

    if groups is None:
        groups = [(0, 10000)]
    if not AllGroups in groups:
        groups.append(AllGroups)

    precs_at_m = {group : [0] * len(ms) for group in groups}
    recs_at_m  = {group : [0] * len(ms) for group in groups}
    D = expected_links.shape[0]
    docCounts = { group : 0 for group in groups }

    docs_lacking_links = []
    for d in range(D):
        # Take out the indices (i.e. IDs) of the expected links
        expt_indices = expected_links[d,:].indices
        if len(expt_indices) == 0:
            docs_lacking_links.append(d)
            continue
        g = find_group(len(expt_indices), groups)

        # Rank the received indices by associated value in descending order
        row = estim_link_probs[d,:]
        ind = (np.argsort(row.data))
        recv_indices = row.indices[ind[::-1]]

        # Sum up the expt_indices at m
        expt_set = set(expt_indices)
        for i in range(len(ms)):
            m = ms[i]
            # Calculate the precision at m
            recv_set = set(recv_indices[:(m+1)])
            if len (recv_set) == 0:
                print ("Ruh-ro")
                continue

            prec_at_m = len(recv_set.intersection(expt_set)) / m
            rec_at_m  = len(recv_set.intersection(expt_set)) / len(expt_set)

            precs_at_m[g][i] += prec_at_m
            recs_at_m[g][i]  += rec_at_m

            precs_at_m[AllGroups][i] += prec_at_m
            recs_at_m[AllGroups][i]  += rec_at_m

        docCounts[g] += 1
        docCounts[AllGroups] += 1

    # Return the mean of average-precisions
    for g in precs_at_m.keys():
        precs = precs_at_m[g]
        precs_at_m[g] = [p / docCounts[g] for p in precs]

        recs = recs_at_m[g]
        recs_at_m[g] = [r / docCounts[g] for r in recs]

    if len(docs_lacking_links) > 0:
        print (str(len(docs_lacking_links)) + " of the " + str(D) + " documents had no links to check, including documents " + ", ".join(str(d) for d in docs_lacking_links[:10]))
    return precs_at_m, recs_at_m, docCounts


def mean_average_prec(expected_links, estim_link_probs):
    '''
    Returns the average of all documents' average-precision scores.

    This metric penalises absent links, but also cases where in the ranking of
    retrieved links there are a lot of irrelevant links ahead of and/or between
    the relevant links.

    The precision is the number of relevant results returned as a proprotion
    of all results returned.

    The precision-at-m is the precision evaluated on the first m elements of the
    returned links (ranked by their associated probabilities)

    The average-precision is the average of precision-at-m scores calculated at all
    positions, m, where a relevant link was found in the list of returned links.

    And as outlined above, the mean-average-precision is the average of these
    average-precision scores across all documents in the corpus.

    :param expected_links a DxD binary CSR matrix. The d-th row contains all out-links
    for document d.
    :param estim_link_probs a DxD CSR matrix. The d-th row contains all out-links
    for document d. We rank the indices by the data to get the ranked list
    of links

    :returns the average of average-precision scores for all documents
    '''
    sum_ap = 0.0
    D      = expected_links.shape[0]

    docs_lacking_links = []
    for d in range(D):
        # Take out the indices (i.e. IDs) of the expected links
        expt_indices = expected_links[d,:].indices
        if len(expt_indices) == 0:
            docs_lacking_links.append(d)
            continue

        # Rank the received indices by associated value in descending order
        row = estim_link_probs[d,:]
        ind = (np.argsort(row.data))
        recv_indices = row.indices[ind[::-1]]

        # Sum up the expt_indices at m
        expt_set = set(expt_indices)
        sum_prec = 0.0
        for e in expt_indices:
            # Find the position, m, of line e in the received link indices
            m, M = 0, len(recv_indices)
            if M < 0:
                print ("Warning: no indices returned for document " + str(d))
            while m < M and recv_indices[m] != e:
                m += 1
            if m == (M - 1) and recv_indices[m] != e:
                print ("WARNING: Could not find link ID " + str(e) + " in the received indices")

            # Calculate the precision at m
            recv_set = set(recv_indices[:(m+1)])
            if len (recv_set) == 0:
                print ("Ruh-ro")
            prec_at_m = len(recv_set.intersection(expt_set)) / len(recv_set)

            # Add precision at that position, m, to the sum of precisions
            # print ("Document %d, Precision at %d is %5.3f    %d in %10s  --?--> %s" % (d, m, prec_at_m, e, str(expt_indices), str(recv_indices[:(m+1)])))
            sum_prec += prec_at_m

        # Add teh average-precision to the sum of average-precisions
        sum_ap += (sum_prec / len(expt_indices))
        # print ("\tAverage Precision = %5.3f  Cumulant Sum = %5.3f" % (sum_prec / len(expt_indices), sum_ap))

    # Return the mean of average-precisions
    if len(docs_lacking_links) > 0:
        print (str(len(docs_lacking_links)) + " of the " + str(D) + " documents had no links to check, including documents " + ", ".join(str(d) for d in docs_lacking_links[:10]))
    return sum_ap / D



