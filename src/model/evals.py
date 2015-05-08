'''
Created on 20 Apr 2015

@author: bryanfeeney
'''

import numpy as np
import numba as nb

Perplexity="perplexity"
MeanAveragePrecAllDocs="meanavgprec_all"

EvalNames = [Perplexity, MeanAveragePrecAllDocs]

def perplexity_from_like(log_likely, token_count):
    return np.exp(-log_likely / token_count)


def word_perplexity(log_likely_fn, model, query, data):
    return perplexity_from_like(log_likely_fn(data, model, query), data.word_count)

@nb.autojit
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

    for d in range(D):
        # Take out the indices (i.e. IDs) of the expected links
        expt_indices = expected_links[d,:].indices
        if len(expt_indices) == 0:
            print("No links to match in document %d" % (d,))
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
            while m < M and recv_indices[m] != e:
                m += 1
            if m == (M - 1) and recv_indices[m] != e:
                print ("WARNING: Could not find link ID " + str(e) + " in the received indices")

            # Calculate the precision at m
            recv_set = set(recv_indices[:(m+1)])
            prec_at_m = len(recv_set.intersection(expt_set)) / len(recv_set)

            # Add precision at that position, m, to the sum of precisions
            # print ("Document %d, Precision at %d is %5.3f    %d in %10s  --?--> %s" % (d, m, prec_at_m, e, str(expt_indices), str(recv_indices[:(m+1)])))
            sum_prec += prec_at_m

        # Add teh average-precision to the sum of average-precisions
        sum_ap += (sum_prec / len(expt_indices))
        # print ("\tAverage Precision = %5.3f  Cumulant Sum = %5.3f" % (sum_prec / len(expt_indices), sum_ap))

    # Return the mean of average-precisions
    return sum_ap / D



