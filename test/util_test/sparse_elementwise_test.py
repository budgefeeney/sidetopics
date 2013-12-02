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

from util.sparse_elementwise import sparseScalarProductOfLnDot

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        rd.seed(0xC0FFEE)
        
        D = 1000
        T = 2000
        K = 10
        
        W = ssp.csr_matrix(np.floor(rd.random((D,T)) * 1.2))
        topics = rd.random((D,K))
        vocab  = rd.random((K,T))
        
        expected = W.multiply(np.log(np.dot(topics, vocab)))
        received = sparseScalarProductOfLnDot(W, topics, vocab)
        
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()