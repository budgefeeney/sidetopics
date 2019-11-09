'''
Created on 1 Dec 2013

@author: bryanfeeney
'''
import unittest

import numpy as np
import numpy.random as rd
import scipy as sp
import scipy.sparse as ssp
import scipy.linalg as la
import scipy.sparse.linalg as sla

from sidetopics.util.sparse_elementwise import sparseScalarProductOfDot, sparseScalarProductOfSafeLnDot, sparseScalarQuotientOfDot

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testscaleProductOfQuotient(self):
        rd.seed(0xC0FFEE)
        
        D = 100
        T = 200
        K = 16
        
        W_d = np.floor(rd.random((D,T)) * 1.4)
        
        W_s = ssp.csr_matrix(W_d)
        topics = rd.random((D,K))
        vocab  = rd.random((K,T))
        
        expected = W_d / topics.dot(vocab)
        received = sparseScalarQuotientOfDot(W_s, topics, vocab)
        
        diff = np.asarray(expected - received.todense())
        trNorm = np.sum(diff * diff)
        print (str(trNorm))
        
        print (str(diff))
        
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()