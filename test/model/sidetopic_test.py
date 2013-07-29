#!/usr/bin/python
# -*- coding: utf-8 -*- 

'''
Created on 3 Jul 2013

@author: bryanfeeney
'''
from __future__ import division
import unittest

from math import sqrt

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

    
    def testInferenceFromModelDerivedExample(self):
        rd.seed(0xC0FFEE) # Global init for repeatable test
        
        T = 100 # Vocabulary size, the number of "terms". Must be a square number
        K = 6   # Topics: This cannot be changed without changing the code that generates the vocabulary
        F = 8   # Features
        P = 6   # Size of principle subspace
        D = 200 # Sample documents (each with associated features) 
        avgWordsPerDoc = 500
        
        # Generate vocab
        beta = 0.01
        betaVec = np.ndarray((T,))
        betaVec.fill(beta)
        vocab = np.zeros((K,T))
        for k in xrange(K):
            vocab[k,:] = rd.dirichlet(betaVec)
        
        # Generate U, then V, then A
        vStdev = 5.0
        uStdev = 5.0
        aStdev = 2.0
        tau    = 0.1
        
        U = rd.multivariate_normal(np.zeros((F,P)).reshape(F * P,1), np.kron(uStdev * np.eye(F), tau * np.eye(P)))
        V = rd.multivariate_normal(np.zeros((P,K)).reshape(P * K,1), np.kron(vStdev * np.eye(P), tau * np.eye(K)))
        V = rd.multivariate_normal(U.dot(V).reshape(F * K, 1), np.kron(aStdev * np.eye(F), tau * np.eye(K)))
        
        # Generate the input features. Assume the features are multinomial and sparse
        # (matches the twitter example)
        featuresHyper = 0.01;
        maxFeaturesOn = 3;
        X = rd.multinomial(3, rd.dirichlet(featuresHyper), (D,F))
        
        # Use the features and the matrix A to generate the topics and documents
        tpcs = rowwise_softmax (X.dot(A))
        
        docLens = rd.poisson(avgWordsPerDoc, (D,))
        W = tpcs.dot(vocab)
        W *= docLens[:, np.newaxis]
        W = np.array(W, dtype=np.int32) # truncate word counts to integers
        
        #
        # Now finally try to train the model
        #
        modelState = newVbModelState(K, F, T, P)
        (trainedState, queryState) = train (modelState, X, W, logInterval=1, iterations=200)
        
        tpcs_inf = rowwise_softmax(queryState.lmda)
        W_inf    = np.array(tpcs_inf.dot(trainedState.vocab) * queryState.docLen[:,np.newaxis], dtype=np.int32)
        
        print("Model Driven Test-Case")
        print("=====================================================================")
        print("Average, squared, per-element difference between true and estimated:")
        print("    Topic Distribution:    %f" % (np.sum((tpcs - tpcs_inf)**2) / len(tpcs),))
        print("    Vocab Distribution:    %f" % (np.sum((vocab - trainedState.vocab)**2) / len(vocab),))
        print("Average absolute difference between true and reconstructed documents:")
        print("    Documents:             %f" % (np.sum(np.abs(W - W_inf)) / np.sum(W),))
        
        
        print("End of Test")
        
        
        

    def _testInferenceFromHandcraftedExample(self):
        rd.seed(0xC0FFEE) # Global init for repeatable test
        
        T = 100 # Vocabulary size, the number of "terms". Must be a square number
        K = 6   # Topics: This cannot be changed without changing the code that generates the vocabulary
        F = 8   # Features
        D = 200 # Sample documents (each with associated features) 
        
        avgWordsPerDoc = 500
        
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
        
        # The vocabulary. Presented graphically there are two with horizontal bands
        # (upper lower); two with vertical bands (left, right);  and two with 
        # horizontal bands (inside, outside)
        vocab = makeSixTopicVocab(T)
        
        # Create our (sparse) features X, then our topic proportions ("tpcs")
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
        (trainedState, queryState) = train (modelState, X, W, logInterval=1, iterations=200)
        
        tpcs_inf = rowwise_softmax(queryState.lmda)
        W_inf    = np.array(tpcs_inf.dot(trainedState.vocab) * queryState.docLen[:,np.newaxis], dtype=np.int32)
                
        print("Handcrafted Test-Case")
        print("=====================================================================")
        print("Average, squared, per-element difference between true and estimated:")
        print("    Topic Distribution:    %f" % (np.sum((tpcs - tpcs_inf)**2) / len(tpcs),))
        print("    Vocab Distribution:    %f" % (np.sum((vocab - trainedState.vocab)**2) / len(vocab),))
        print("Average absolute difference between true and reconstructed documents")
        print("    Documents:             %f" % (np.sum(np.abs(W - W_inf)) / np.sum(W),))
        
        
        print("End of Test")
        

def makeSixTopicVocab(T):
    '''
    Create the vocabulary. Two topics will have horizontal bands. Two will
    have vertical bands. One will be a everything within a diagonal band
    One will be everything without.
    
    Requires T, the number of words in the vocabulary
    '''
    K = 6
    vocab = rd.random((K, T))
    side  = int(sqrt(T))
    
    AMPLIFIED = 8
    LOWERED   = 2
    RANGE     = AMPLIFIED + LOWERED
    
    upperArr = np.array([AMPLIFIED if i < side/2 else LOWERED   for i in xrange(side)])
    lowerArr = np.array([LOWERED   if i < side/2 else AMPLIFIED for i in xrange(side)])
    bandArr  = makeBandMatrix (side)
    innerBand = (bandArr * (AMPLIFIED - LOWERED)) + LOWERED;
    outerBand = -(innerBand - RANGE)
    
    vocab[0,:] = (vocab[0,:].reshape(side,side) * upperArr[:,np.newaxis]).reshape(1,T)
    vocab[1,:] = (vocab[1,:].reshape(side,side) * lowerArr[:,np.newaxis]).reshape(1,T)
    vocab[2,:] = (vocab[2,:].reshape(side,side) * upperArr[np.newaxis,:]).reshape(1,T)
    vocab[3,:] = (vocab[3,:].reshape(side,side) * lowerArr[np.newaxis,:]).reshape(1,T)
    vocab[4,:] = (vocab[4,:].reshape(side,side) * innerBand).reshape(1,T)
    vocab[5,:] = (vocab[5,:].reshape(side,side) * outerBand).reshape(1,T)
    
    return normalizerows_ip(vocab)

def makeBandMatrix (side):
    '''
    Creates a square band matrix with ones on the band, and zeros elsewhere.
    The band is as wide as possible limited by the fact that no more that
    50% of the elements are 1
    
    This is very, very slow
    
    Params:
    side - the length of a side. The matrix dimensions are (side,side)
    '''
    bandWidth = 1
    bandOnes = side;
    while bandOnes < 0.5 * side * side:
        bandOnes += (side - bandWidth) * 2
        bandWidth += 1
        
    
    mat = np.ndarray((side,side))
    mat.fill(0)
    for row in xrange(side):
        for band in xrange(bandWidth):
            if row + band < side:
                mat[row, row+band] = 1
            if row - band >= 0:
                mat[row, row-band] = 1
    
    return mat



def _showme(mat):
    '''
    Presents a matrix as integer percentages. For debugging only
    '''
    mat = np.array(mat * 100, dtype=np.int32)
    (rows, _) = mat.shape
    for row in xrange(rows):
        print str(mat[row,:])

def _showvoc(vocab, k):
    '''
    Presents the given vocabulary distribution as as a square matrix of percentages
    '''
    side = int (sqrt(len(vocab[k,:])))
    _showme (vocab[k,:].reshape(side,side))
    
def _showvocs(vocab):
    '''
    Presents the given vocabulary distribution as as a square matrix of percentages
    '''
    (K,_) = vocab.shape
    
    for k in xrange(K):
        print ("K = %d" % (k))
        print ("=============================================")
        side = int (sqrt(len(vocab[k,:])))
        _showme (vocab[k,:].reshape(side,side))

def vec(A):
    '''
    The standard vec operator from linear algebra, which extracts all the columns
    of the given matrix, and from left to right stacks them from top to bottom.
    
    Params
    A - a PxN matrix
    
    Return
    a P*N,1 vector
    '''
    return np.reshape(np.transpose(A), (-1,1))

def matrix_normal(mean, rowCov, colCov):
    '''
    Draw from a matrix variate normal distribution
    
    mean the mean matrix, with dimensions PxN
    rowCov the row covariance matrix, with dimensions NxN
    colCov the column covariance matrix, with dimensions PxP
    
    Return
    a PxN matrix
    '''
    return rd.multivariate_normal(vec(mean), np.kron(colCov, rowCov))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()