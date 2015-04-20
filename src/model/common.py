__author__ = 'bryanfeeney'

import numpy as np
import numpy.random as rd
import scipy.sparse as ssp
import pickle as pkl
from math import ceil

class DataSet:
    '''
    The input to one of our algorithms. Contains at a minimum words. May also contain
    features and a matrix of links.
    '''
    def __init__(self, words, feats=None, links=None):
        self._check_and_assign_matrices(words, feats, links)

    def __init__(self, words_file, feats_file=None, links_file=None):
        with open(words_file, 'rb') as f:
            words = pkl.load(f)

        if feats_file is not None:
            with open(feats_file, 'rb') as f:
                feats = pkl.load(f)

        if links_file is not None:
            with open(links_file, 'rb') as f:
                links = pkl.load(f)

        self._check_and_assign_matrices(words, feats, links)

    def _check_and_assign_matrices(self, words, feats=None, links=None):
        assert words.shape[0] > 0, "No rows in the document-words matrix"
        assert words.shape[1] > 100, "Fewer than 100 words in the document-words matrix, which seems unlikely"
        assert feats is None or feats.shape[0] == words.shape[0], "Differing row-counts for document-word and document-feature matrices"
        assert links is None or links.shape[0] == words.shape[0], "Differing row-counts for document-word and document-link matrices"

        assert type(words) is ssp.csr_matrix, "Words are not stored as a sparse CSR matrix"
        assert type(feats) is ssp.csr_matrix, "Features are not stored as a sparse CSR matrix"
        assert type(links) is ssp.csr_matrix, "Links are not stored as a sparse CSR matrix"

        self._words = words
        self._feats = feats
        self._links = links


    @property()
    def words(self):
        return self._words

    @property()
    def links(self):
        return self._links

    @property()
    def feats(self):
        return self._feats

    @property()
    def doc_count(self):
        return self._words.shape[0]

    @property()
    def word_count(self):
        return self._words.sum()

    @property()
    def link_count(self):
        assert self._links is not None, "Calling link_count when no links matrix was every loaded"
        return self._links.sum()

    def prune_and_shuffle(self, min_doc_len=0.5):
        '''
        This IN-PLACE operation prunes out any documents where the document-length
        is less than the minimum, and the shuffles the matrices in a coherent manner.

        The order of the remaining matrices is returned.
        '''
        doc_lens = np.squeeze(np.asarray(self._words.sum(axis=1)))
        if doc_lens.min() < min_doc_len:
            print("Input doc-term matrix contains some empty rows. These have been removed.")
            good_rows = (np.where(doc_lens > 0.5))[0]
            order = good_rows
        else:
            doc_count = self._words.shape[0]
            order = np.linspace(0, doc_count - 1, doc_count)

        rd.shuffle(order)
        self._words = self._words[order, :]
        self._feats = self._feats[order, :]
        self._links = \
            self._links[order, order] \
            if self._links.shape[0] == self._links.shape[1] \
            else self._links[order, :]

        return order

    def cross_valid_split (self, test_fold_id, num_folds):
        '''
        For cross-validation, used the K-folds method to partition the
        data in the train and query components.

        Returns a tuple, the left being the Input object with the training
        data, the right being the input object with the query data
        '''
        doc_count = self.doc_count
        query_size  = ceil(doc_count / num_folds) # a single fold
        train_size = doc_count - query_size

        start = query_size * test_fold_id
        end   = start + query_size

        query_range = np.arange(start, end) % doc_count
        train_range = np.arange(end, end + train_size) % doc_count

        train = Input ( \
            self._words[train_range], \
            None if self._feats is None else self._feats[train_range], \
            None if self._links is None else self._links[train_range]
        )
        query = Input ( \
            self._words[query_range], \
            None if self._feats is None else self._feats[query_range], \
            None if self._links is None else self._links[query_range]
        )

    def doc_completion_split(self):
        '''
        Returns two variants of this dataset - usually this is the query segment
        from cross_valid_split().

        If there are no features, this takes every file and partitions it in two,
        such that the distribution of words in both sides is roughly equal.

        If there are features, this just returns two references to this unchanged
        object.

        This is always split with the a custom RNG seeded with 0xBADB055
        '''
        if self._feats is not None:
            return self, self

        rng = rd.RandomState(0xBADB055)

        dat    = self._words.data
        jitter = rng.normal(scale=0.3, size=len(dat)).astype(dtype=W.dtype)
        evl    = dat + jitter
        est    = np.around(evl / 2.0)
        evl    = dat - est

        return \
            ssp.csr_matrix((est, self._words.indices,  self._words.indptr), shape=self._words.shape), \
            ssp.csr_matrix((evl,  self._words.indices,  self._words.indptr), shape=self._words.shape)