__author__ = 'bryanfeeney'

import numpy as np
import numpy.random as rd
import scipy.sparse as ssp
import pickle as pkl
from math import ceil

class DataSet:

    def __init__(self, words, feats=None, links=None, order=None, limit=0):
        '''
        The three matrices that make up our features.

        If order is not none, then it means that the matrice were altered before
        being passed in, and order is the set of indices applied to the original
        (unseen) matrices before they were passed to this constructor. Thus,
        unlike from_files(), providing an order to this constructor will not
        cause the given matrices to be altered.

        If limit is greater than zero, then only the first "limit" documents are
        considered.
        '''
        assert words.shape[0] > 0,   "No rows in the document-words matrix"
        assert words.shape[1] > 100, "Fewer than 100 words in the document-words matrix, which seems unlikely"

        assert feats is None or feats.shape[0] == words.shape[0], "Differing row-counts for document-word and document-feature matrices"
        assert links is None or links.shape[0] == words.shape[0], "Differing row-counts for document-word and document-link matrices"

        assert type(words) is ssp.csr_matrix, "Words are not stored as a sparse CSR matrix"
        assert links is None or type(links) is ssp.csr_matrix, "Links are not stored as a sparse CSR matrix"

        # Now that the checks are done with, assign the values and apply the ordering.
        self._words = words
        self._feats = feats
        self._links = links

        num_docs = words.shape[0]
        if 0 < limit < num_docs:
            order = np.linspace(0, limit - 1, limit).astype(np.int32) \
                if order is None \
                else order[:limit]
            num_docs = limit

        if order is not None:
            self._order = order
        else:
            self._order = np.linspace(0, num_docs - 1, num_docs).astype(np.int32)

    @classmethod
    def from_files(cls, words_file, feats_file=None, links_file=None, order=None, limit=0):
        '''
        The three matrices that make up our features. Each one is loaded in from
        the given pickle file.

        The order specifies the subset of rows to consider (and columns in the
        case that links is square). This subset is extracted and passed to the
        DataSet constructor to build the DataSet object.

        If limit is greater than zero, then only the first "limit" documents are
        considered.
        '''
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

        result = DataSet(words, feats, links, order=None, limit=limit)
        if order is not None:
            result._reorder(order)

        return result

    def copy_with_changes(self, words=None, feats=None, links=None):
        return DataSet(
            self._words if words is None else words,
            self._feats if feats is None else feats,
            self._links if links is None else links,
            self._order
        )

    def copy(self, deep=False):
        if deep:
            return DataSet(
                self._words.copy(deep=True),
                self._feats.copy(deep=True) if self._feats is not None else None,
                self._links.copy(deep=True) if self._links is not None else None,
                self._order.copy(deep=True) if self._order is not None else None
            )
        else:
            return DataSet(
                self._words,
                self._feats,
                self._links,
                self._order
            )


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
    def word_and_link_count(self):
        return self.word_count + (self.link_count if self.has_links() else 0)

    @property
    def link_count(self):
        assert self._links is not None, "Calling link_count when no links matrix was every loaded"
        return self._links.sum()


    def __len__(self):
        return self._words.shape[0]


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


    def add_intercept_to_feats_if_required(self):
        '''
        If there is no element in the features which is set to 1 for all
        documents, then add just such an element and return True.

        If such an element already exists, do nothing and return False
        :return: True if features where changed, False otherwise.
        '''
        if self._feats is None:
            return;

        means = self._feats.mean(axis=0)
        if means.max() < (1-1E-30):
            self._feats = ssp.hstack((self._feats, np.ones((self.doc_count, 1))), "csr")
            return True

        return False


    def _reorder (self, order):
        '''
        Reorders the rows of all matrices according to the given order, which may
        be smaller than the total of rows. Additionally, if links is square, its
        columns are also re-ordered.

        Maybe this should update the order property as well...
        '''
        self._words = self._words[order, :]
        self._feats = None if self._feats is None else self._feats[order, :]
        self._links = _reorder_link_matrix(self._links, order)


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


    def prune_and_shuffle(self, min_doc_len=0.5, min_link_count=0, seed=0xC0FFEE):
        '''
        This IN-PLACE operation prunes out any documents where the document-length
        is less than the minimum, and the shuffles the matrices in a coherent manner.

        The order of the remaining matrices is returned.

        Note that if no links matrix exists, min_link_count is ignored.
        '''
        rng = rd.RandomState(seed)

        tmp = self._words.astype(np.bool).astype(np.int32)

        doc_lens = np.squeeze(np.asarray(tmp.sum(axis=1)))
        if doc_lens.min() < min_doc_len:
            good_rows = (np.where(doc_lens >= min_doc_len))[0]
            self._order = good_rows
            print ("Removed %d documents with fewer than %d unique words. %d documents remain" \
                   % (len(doc_lens) - len(good_rows), min_doc_len, len(good_rows)))
        else:
            doc_count = self._words.shape[0]
            self._order = np.linspace(0, doc_count - 1, doc_count)

        rng.shuffle(self._order)
        self._reorder(self._order)

        if min_link_count == 0 or not self.has_links():
            return self._order

        trimmed = False
        original_num_docs = self.doc_count
        link_counts = np.squeeze(np.array(self._links.sum(axis=1)))
        sufficient_docs = np.where(link_counts >= min_link_count)[0]
        while len(sufficient_docs) < self._links.shape[0]:
            trimmed = True
            self._reorder(sufficient_docs)
            self._order = self._order[sufficient_docs]
            link_counts = np.squeeze(np.array(self._links.sum(axis=1)))
            sufficient_docs = np.where(link_counts >= min_link_count)[0]

        if trimmed:
            print("Removed %d documents whose out-link counts were less than %d. %d documents remain" \
                   % (original_num_docs - self.doc_count, min_link_count, self.doc_count))

        return self._order


    def cross_valid_split_indices (self, test_fold_id, num_folds):
        '''
        For cross-validation, used the K-folds method to partition the
        data in the train and query components.

        Returns a tuple, the left being the Input object with the training
        data, the right being the input object with the query data

        This just returns the train and query indices respectively
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

        return train_range, query_range


    def cross_valid_split (self, test_fold_id, num_folds):
        '''
        For cross-validation, used the K-folds method to partition the
        data in the train and query components.

        Returns a tuple, the left being the Input object with the training
        data, the right being the input object with the query data
        '''
        train_range, query_range = self.cross_valid_split_indices(test_fold_id, num_folds)

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


    def doc_completion_split(self, min_doc_len=0, seed=0xBADB055, est_prop=0.5):
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
        jitter = rng.normal(scale=est_prop * 0.666, size=len(dat))
        evl    = dat + jitter
        est    = np.around(evl * est_prop)
        evl    = dat - est

        est = est.astype(self._words.dtype)
        evl = evl.astype(self._words.dtype)

        words_train = ssp.csr_matrix((est, self._words.indices, self._words.indptr), shape=self._words.shape)
        words_query = ssp.csr_matrix((evl, self._words.indices, self._words.indptr), shape=self._words.shape)

        if min_doc_len > 0:
            t_lens = np.squeeze(np.asarray(words_train.sum(axis=1)))
            q_lens = np.squeeze(np.asarray(words_query.sum(axis=1)))
            t_empties = np.where(t_lens < min_doc_len)[0]
            q_empties = np.qhere(q_lens < min_doc_len)[0]

            empties = np.unique ([t_empties, q_empties])
            if len (empties) > 0:
                words_train = words_train[empties,:]
                words_query = words_query[empties,:]
                print ("Removed %d documents with fewer than %d words. %d documents remain" % (len(empties), min_doc_len, words_train.shape[0] - len(empties)))

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


    def folded_link_prediction_split(self, min_link_count, fold_id, fold_count, symmetric=False):
        '''
        Returns two variants of this DataSet, both having the exact same words and features
        (by reference, no copies), but having different sets of links. For each document
        having more than min_link_count links, a proportion of those links is removed.
        The proportion is 1/folds. Which links are removed are chosen deterministically
        by the fold_id

        If symmetric is true, the two partitioned matrices will also be symmetric. This is
        primarily a requirement of undirected graphs.
        '''
        assert self._links is not None, "Can't do a link prediction split if there are no links!"

        if symmetric:
            raise ValueError("Symmetric splits are not supported")
        else:
            links_train, links_query, docSubset = _folded_split(self._links, fold_id, fold_count, min_link_count)

        return \
            DataSet(self._words, self._feats, links_train), \
            DataSet(self._words, self._feats, links_query), \
            docSubset

    def split_on_feature(self, features):
        '''
        Returns two datasets. The left consists of all documents for which
        none of the features are set. The right consists of all documents for
        which the first feature is set.

        :param features: the features used to make the split
        :return: two DataSet object, with _shallow_ copies of the given
        data
        '''
        def two_dim_reorder(matrix, row_order, col_order):
            return (matrix[row_order,:])[:,col_order]

        Epsilon = 1E-30
        assert self._feats is not None, "Need features to make this split"
        assert len(features) > 0, "Need at least two features for this to work"

        mask      = np.squeeze(np.array(np.abs(self._feats[:,features].sum(axis=1))))
        trainDocs = np.where(mask < Epsilon)[0]
        queryDocs = np.where(np.ndarray(buffer=self._feats[:,features[0]].todense(), shape=(self._feats.shape[0],)) > Epsilon)[0]

        return DataSet(words=self._words[trainDocs,:], \
                       feats=self._feats[trainDocs,:], \
                       links=two_dim_reorder(self._links, trainDocs, trainDocs), \
                       order=self._order[trainDocs]), \
               DataSet(words=self._words[queryDocs,:], \
                       feats=self._feats[queryDocs,:], \
                       links=two_dim_reorder(self._links, queryDocs, trainDocs), \
                       order=self._order[queryDocs]), \
               trainDocs


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
    rng.shuffle(ind)
    split_point = len(ind) / 2

    left_ind  = np.sort(ind[:split_point])
    right_ind = np.sort(ind[split_point:])

    return \
        _symm_csr (dat[left_ind],  row[left_ind],  col[left_ind],  X.shape), \
        _symm_csr (dat[right_ind], row[right_ind], col[right_ind], X.shape),


def _folded_split(X, fold_id, fold_count, min_link_count):
    '''
    Given a  matrix, splits it into two matrices such that their sum is equal to the
    given matrix. Moreover, if an element of one of the partitioned matrices is non-zero,
    then it _will_ be zero in the other partitioned matrix

    The left matrix has (1-fold_count)/fold_count links. The right matrix has
    the remnants. The split is determined by links, not link counts - i.e. all of a
    links counts are either in the left matrix or the right.

    Which links move where is set deterministically by the fold_id

    Return a typle containing
     - a matrix with the majority of links
     - a matrix with the removed links
     - a list of the rows where links were removed.
    '''
    Lptr, Lind, Ldat = [0], [], []
    Rptr, Rind, Rdat = [0], [], []
    docSubset = []

    start = 0
    row   = 0
    while row < X.shape[0]:
        col_count  = X.indptr[row + 1] - X.indptr[row]
        end        = start + col_count

        if col_count < min_link_count:
            cols = [c for c in X.indices[start:end]]
            vals = [v for v in X.data[start:end]]

            Lind += cols
            Ldat += vals
            Lptr.append(Lptr[-1] + len(cols))

            Rind += cols
            Rdat += vals
            Rptr.append(Rptr[-1] + len(cols))
        else:
            query_size   = col_count // fold_count
            query_start  = start + query_size * fold_id
            query_end    = min(query_start + query_size, end)

            Lind += [c for c in X.indices[start:query_start]]
            Lind += [c for c in X.indices[query_end:end]]
            Ldat += [v for v in X.data[start:query_start]]
            Ldat += [v for v in X.data[query_end:end]]
            Lptr.append (Lptr[-1] + col_count - query_size)

            Rind += [c for c in X.indices[query_start:query_end]]
            Rdat += [v for v in X.data[query_start:query_end]]
            Rptr.append (Rptr[-1] + query_size)

            docSubset.append(row)

        start  = end
        row   += 1

    L = ssp.csr_matrix((np.array(Ldat, dtype=X.dtype), \
                        np.array(Lind, dtype=np.int32), \
                        np.array(Lptr, dtype=np.int32)), shape=X.shape)
    R = ssp.csr_matrix((np.array(Rdat, dtype=X.dtype), \
                        np.array(Rind, dtype=np.int32), \
                        np.array(Rptr, dtype=np.int32)), shape=X.shape)

    return L, R, docSubset


def _reorder_link_matrix(matrix, order):
    '''
    Returns a matrix containing only the given rows (in order).
    If the original matrix is square, the resulting matrix will
     also only have the columns specific by order
    :param matrix: the matrix to reorder
    :param order: the ordering to apply to rows (and maybe columns)
    :return:the re-ordered matrix
    '''
    return None if matrix is None \
            else ((matrix[order, :])[:, order] \
                if matrix.shape[0] == matrix.shape[1] \
                else matrix[order, :])