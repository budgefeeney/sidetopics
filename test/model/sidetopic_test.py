#!/usr/bin/python
# -*- coding: utf-8 -*- 

'''
Created on 3 Jul 2013

@author: bryanfeeney
'''
import unittest
from model.sidetopic import newVbModelState, train, rowwise_softmax, normalizerows_ip

import numpy as np
import scipy.linalg as la
import numpy.random as rd
import scipy.sparse as ssp

class StmTest(unittest.TestCase):
    '''
    Provides basic unit tests for the variational SideTopic inference engine using
    small inputs derived from known parameters.
    '''

    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testInference(self):
        rd.seed(0xC0FFEE) # Global init for repeatable test!
        
        T = 100 # Vocabulary size, the number of "terms"
        K = 5
        F = 8
        D = 200
        
        avgWordsPerDoc = 200
        
        # Determine what A, U and V should be by doing PCA on
        # a random matrix A, then recomputing it given the decomposition
        A = rd.random((F,K)) * 10
        (U, S, _) = la.svd (A)
        
        cdf = [sum(S[:f]) for f in xrange(1,F+1)]
        P = len ([i for i in xrange(F) if cdf[i] > 0.75 * sum(S)])
        
        if P == F: raise ValueError("Can't reduce the dimension")
        
        U = U[:,:P]; 
        V = np.ndarray((P,K))
        for col in xrange(K):
            (soln, _, _, _ ) = la.lstsq(U, A[:,col]) 
            V[:,col] = soln
        
        A = U.dot(V)
        
        # Create the vocabulary
        #
        vocab = normalizerows_ip (rd.random((K, T)))
        
        # Create our (sparse) features X, then our topic proportions tpcs
        # then our word counts W
        X = np.array([1 if rd.random() < 0.3 else 0 for _ in xrange(D*F)]).reshape(D,F)
#        X = ssp.csr_matrix(X)
        
        lmda = X.dot(A)
        print ("lmda.mean() = %f" % (lmda.mean()))
        tpcs = rowwise_softmax (lmda)
        
        docLens = rd.poisson(avgWordsPerDoc, (D,))
        W = tpcs.dot(vocab)
        W *= docLens[:, np.newaxis]
        W = np.array(W, dtype=np.int32) # truncate word counts to integers
        
        #
        # Now finally try to train the model
        #
        modelState = newVbModelState(K, F, T, P)
        (trainedState, queryState) = train (modelState, X, W, logInterval=1, iterations=1000)
        
        print("Sum of squared difference between true and estimated A is %f" % (np.sum((A - trainedState.A)**2)))
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()