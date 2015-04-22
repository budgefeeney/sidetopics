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
    def __init__(self, words, feats=None, links=None, order=None):
        '''
        The three matrices that make up our features. The order vector gives
        the postions of rows in the original matrices. Specifying an order does
        _not_ change these matrices, however if these matrices were changed before
        this DataSet object was created, you should specify the order there.
        '''
        self._check_and_assign_matrices(words, feats, links, order)


    def __init__(self, words_file, feats_file=None, links_file=None):
        with open(words_file, 'rb') as f:
            words = pkl.load(f)

        if feats_file is not None:
            with open(feats_file, 'rb') as f:
                feats = pkl.load(f)
        else:
            feats = None

        if links_file is not None:
            with open(links_file, 'rb') as f:
                links = pkl.load(f)
        else:
            links = None

        self._check_and_assign_matrices(words, feats, links, None)


    def _check_and_assign_matrices(self, words, feats=None, links=None, order=None):
        assert words.shape[0] > 0, "No rows in the document-words matrix"
        assert words.shape[1] > 100, "Fewer than 100 words in the document-words matrix, which seems unlikely"

        assert feats is None or feats.shape[0] == words.shape[0], "Differing row-counts for document-word and document-feature matrices"
        assert links is None or links.shape[0] == words.shape[0], "Differing row-counts for document-word and document-link matrices"

        assert type(words) is ssp.csr_matrix, "Words are not stored as a sparse CSR matrix"
        assert feats is None or type(feats) is ssp.csr_matrix, "Features are not stored as a sparse CSR matrix"
        assert links is None or type(links) is ssp.csr_matrix, "Links are not stored as a sparse CSR matrix"

        self._words = words
        self._feats = feats
        self._links = links

        self._order = order if order is not None \
            else np.linspace(0, words.shape[0] - 1, words.shape[0]).astype(np.int32)


    @property
    def words(self):
        return self._words


    @property
    def links(self):
        return self._links


    @property
    def feats(self):
        return self._feats


    @property
    def doc_count(self):
        return self._words.shape[0]


    @property
    def word_count(self):
        return self._words.sum()

    @property
    def order(self):
        '''
        :return: the subset of the rows originally read in that are contained in these
        matrices
        '''
        return self._order

    def has_links(self):
        return self._links is not None

    def has_features(self):
        return self._feats is not None


    @property
    def link_count(self):
        assert self._links is not None, "Calling link_count when no links matrix was every loaded"
        return self._links.sum()

    def convert_to_dtype(self, dtype):
        '''
        Inplace conversion to given dtype
        '''
        self.convert_to_dtypes(dtype, dtype, dtype)

    def convert_to_dtypes(self, words_dtype, feats_dtype, links_dtype):
        '''
        Inplace conversion to given dtypes
        '''
        self._words = self._words.astype(words_dtype)
        self._feats = None if self._feats is None else self._feats.astype(feats_dtype)
        self._links = None if self._links is None else self._links.astype(links_dtype)

    def reorder (self, order):
        '''
        Reorders the rows of all matrices according to the given order, which may
        be smaller than the total of rows. Additionally, if links is square, its
        columns are also re-ordered.

        Maybe this should update the order property as well...
        '''
        self._words = self._words[order, :]
        self._feats = None if self._feats is None else self._feats[order, :]
        self._links = None if self._links is None \
            else (self._links[order, order] \
                    if self._links.shape[0] == self._links.shape[1] \
                    else self._links[order, :])


    def convert_to_undirected_graph(self):
        '''
        Converts the link matrix to an undirected graph in-place. If a link
        exists in either direction, it will now exist in both directions.
        '''
        assert self._links is not None, "Can't call this if there are no links"
        self._links = self._links + self._links.T


    def convert_to_binary_link_matrix(self):
        '''
        Converts the link matrix to a binary 0-1 matrix in-place.
        '''
        assert self._links is not None, "Can't call this if there are no links"
        self._links.data.fill(1)


    def prune_and_shuffle(self, min_doc_len=0.5, min_link_count=0):
        '''
        This IN-PLACE operation prunes out any documents where the document-length
        is less than the minimum, and the shuffles the matrices in a coherent manner.

        The order of the remaining matrices is returned.

        Note that if no links matrix exists, min_link_count is ignored.
        '''
        doc_lens = np.squeeze(np.asarray(self._words.sum(axis=1)))
        if doc_lens.min() < min_doc_len:
            print("Input doc-term matrix contains some empty rows. These have been removed.")
            good_rows = (np.where(doc_lens > 0.5))[0]
            self._order = good_rows
        else:
            doc_count = self._words.shape[0]
            self._order = np.linspace(0, doc_count - 1, doc_count)

        rd.shuffle(self._order)
        self.reorder(self._order)

        if min_link_count == 0 or not self.has_links():
            return self._order

        link_counts = np.squeeze(np.array(self._links.sum(axis=1)))
        sufficient_docs = np.where(link_counts >= min_link_count)[0]
        while len(sufficient_docs) < self._links.shape[0]:
            self.reorder(sufficient_docs)
            self._order = self._order[sufficient_docs]
            link_counts = np.squeeze(np.array(self._links.sum(axis=1)))
            sufficient_docs = np.where(link_counts >= min_link_count)[0]

        return self._order


    def cross_valid_split (self, test_fold_id, num_folds):
        '''
        For cross-validation, used the K-folds method to partition the
        data in the train and query components.

        Returns a tuple, the left being the Input object with the training
        data, the right being the input object with the query data
        '''
        assert test_fold_id < num_folds, "The query fold ID can't be greater than the total number of folds"
        assert num_folds > 1, "The number of folds should be greater than one"

        doc_count = self.doc_count
        query_size  = ceil(doc_count / num_folds) # a single fold
        train_size = doc_count - query_size

        start = query_size * test_fold_id
        end   = start + query_size

        query_range = np.arange(start, end) % doc_count
        train_range = np.arange(end, end + train_size) % doc_count

        train = DataSet ( \
            self._words[train_range], \
            None if self._feats is None else self._feats[train_range], \
            None if self._links is None else self._links[train_range], \
            self._order[train_range]
        )
        query = DataSet ( \
            self._words[query_range], \
            None if self._feats is None else self._feats[query_range], \
            None if self._links is None else self._links[query_range], \
            self._order[query_range]
        )

        return train, query


    def doc_completion_split(self, seed=0xBADB055):
        '''
        Returns two variants of this dataset - usually this is the query segment
        from cross_valid_split().

        If there are no features, this takes every file and partitions it in two,
        such that the distribution of words in both sides is roughly equal.

        If there are features, this just returns two references to this unchanged
        object.

        This is always split with the a custom RNG seeded with the given seed
        '''
        if self._feats is not None:
            return self, self

        rng = rd.RandomState(seed)

        dat    = self._words.data
        jitter = rng.normal(scale=0.3, size=len(dat)).astype(dtype=self._words.dtype)
        evl    = dat + jitter
        est    = np.around(evl / 2.0)
        evl    = dat - est

        words_train = ssp.csr_matrix((est, self._words.indices, self._words.indptr), shape=self._words.shape)
        words_query = ssp.csr_matrix((evl, self._words.indices, self._words.indptr), shape=self._words.shape)

        return \
            DataSet(words_train, self._feats, self._links), \
            DataSet(words_query, self._feats, self._links)


    def link_prediction_split(self, symmetric=True, seed=0xC0FFEE):
        '''
        Returns two variants of this DataSet, both having the exact same words and features
        (by reference, no copies), but having different sets of links. For each document,
        its links are partitioned into two sets at random, according to the given seed. Where
        a link occurs n > 1 times, all n occurrences will be put in one set.

        If symmetric is true, the two partitioned matrices will also be symmetric. This is
        primarily a requirement of undirected graphs.
        '''
        assert self._links is not None, "Can't do a link prediction split if there are no links!"

        rng = rd.RandomState(seed)

        if symmetric:
            links_train, links_query = _split_symm(self._links, rng)
        else:
            links_train, links_query = _split(self._links, rng)


        return \
            DataSet(self._words, self._feats, links_train), \
            DataSet(self._words, self._feats, links_query)


def _split(X, rng):
    '''
    Given a  matrix, splits it into two matrices such that their sum is equal to the
    given matrix. Moreover, if an element of one of the partitioned matrices is non-zero,
    then it _will_ be zero in the other partitioned matrix

    A random number generator is provided to determine how the "random" split should be
    performed.
    '''
    nnz = X.nnz
    coo = X.tocoo(copy=False)

    row, col, dat = coo.row, coo.col, coo.data

    ind = np.arange(nnz)
    rng.shuffle (ind)
    split_point = nnz / 2

    left_ind  = np.sort(ind[:split_point])
    right_ind = np.sort(ind[split_point:])

    return \
        ssp.coo_matrix ((dat[left_ind],  (row[left_ind],  col[left_ind])),  shape=X.shape).tocsr(), \
        ssp.coo_matrix ((dat[right_ind], (row[right_ind], col[right_ind])), shape=X.shape).tocsr(),


def _symm_csr(dat, row, col, shape):
    '''
    Creates a symmetric matrix from the given data and row and column coordinates,
    where the coorindates are for the upper triangle only. These are then copied
    to create the lower-triangle. The returned matrix is in CSR format.
    '''
    count = len(dat)

    nu_dat = np.ndarray(shape=(count*2,), dtype=dat.dtype)
    nu_row = np.ndarray(shape=(count*2,),  dtype=row.dtype)
    nu_col = np.ndarray(shape=(count*2,),  dtype=col.dtype)

    nu_dat[:count] = dat
    nu_dat[count:] = dat

    nu_row[:count] = row
    nu_row[count:] = col

    nu_col[:count] = col
    nu_col[count:] = row

    return ssp.coo_matrix((nu_dat, (nu_row, nu_col)), shape=shape).tocsr()


def _split_symm(X, rng):
    '''
    Given a symmetric matrix, splits it into two symmetric matrices such that their
    sum is equal to the given matrix. Moreover, if an element of one of the
    partitioned matrices is non-zero, then it _will_ be zero in the other partitioned
    matrix

    A random number generator is provided to determine how the "random" split should be
    performed.
    '''
    nnz = X.nnz
    coo = X.tocoo(copy=False)

    row, col, dat = coo.row, coo.col, coo.data

    ind = np.array([i for i in range(nnz) if row[i] >= col[i]], dtype=np.int32)
    rng.shuffle (ind)
    split_point = len(ind) / 2

    left_ind  = np.sort(ind[:split_point])
    right_ind = np.sort(ind[split_point:])

    return \
        _symm_csr (dat[left_ind],  row[left_ind],  col[left_ind],  X.shape), \
        _symm_csr (dat[right_ind], row[right_ind], col[right_ind], X.shape),