'''
Created on 11 Nov 2013

@author: bryanfeeney
'''
import unittest

import numpy as np
import numpy.random as rd
import scipy as sp
import scipy.sparse as ssp
import scipy.linalg as la
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

from util.vectrans import *

class Test(unittest.TestCase):

    def setUp(self):
        rd.seed(0xC0FFEE)

    # The example here is
    #
    #   11 12
    #   22 22
    #   31 32
    #   41 42
    #   51 52
    #   61 61
    #
    # begin reshaped as
    #
    #   11 31 51
    #   21 41 61
    #   12 32 52
    #   22 42 62

    def testDenseVecTransF4(self):
        A = np.array([[11, 12], [21, 22], [31, 32], [41, 42], [51, 52], [61, 61]]).astype(np.float32)
        A2 = vec_transpose(A, 2)
        
        self.assertEquals (4, A2.shape[0])
        self.assertEquals (3, A2.shape[1])
        self.assertEquals (np.float32, A2.dtype)
        
        self.assertFloatEqual(11, A2[0,0])
        self.assertFloatEqual(31, A2[0,1])
        self.assertFloatEqual(51, A2[0,2])
        
        self.assertFloatEqual(21, A2[1,0])
        self.assertFloatEqual(31, A2[1,1])
        self.assertFloatEqual(41, A2[1,2])
        
        self.assertFloatEqual(12, A2[2,0])
        self.assertFloatEqual(32, A2[2,1])
        self.assertFloatEqual(52, A2[2,2])
        
        self.assertFloatEqual(22, A2[3,0])
        self.assertFloatEqual(42, A2[3,1])
        self.assertFloatEqual(62, A2[3,2])

    def testDenseVecTransF8(self):
        A = np.array([[11, 12], [21, 22], [31, 32], [41, 42], [51, 52], [61, 61]]).astype(np.float64)
        A2 = vec_transpose(A, 2)
        
        self.assertEquals (4, A2.shape[0])
        self.assertEquals (3, A2.shape[1])
        self.assertEquals (np.float64, A2.dtype)
        
        self.assertFloatEqual(11, A2[0,0])
        self.assertFloatEqual(31, A2[0,1])
        self.assertFloatEqual(51, A2[0,2])
        
        self.assertFloatEqual(21, A2[1,0])
        self.assertFloatEqual(31, A2[1,1])
        self.assertFloatEqual(41, A2[1,2])
        
        self.assertFloatEqual(12, A2[2,0])
        self.assertFloatEqual(32, A2[2,1])
        self.assertFloatEqual(52, A2[2,2])
        
        self.assertFloatEqual(22, A2[3,0])
        self.assertFloatEqual(42, A2[3,1])
        self.assertFloatEqual(62, A2[3,2])
    
    

    def testDenseVecTransInt(self):
        A = np.array([[11, 12], [21, 22], [31, 32], [41, 42], [51, 52], [61, 61]]).astype(np.int32)
        A2 = vec_transpose(A, 2)
        
        self.assertEquals (4, A2.shape[0])
        self.assertEquals (3, A2.shape[1])
        self.assertEquals (np.float64, A2.dtype)
        
        self.assertFloatEqual(11, A2[0,0])
        self.assertFloatEqual(31, A2[0,1])
        self.assertFloatEqual(51, A2[0,2])
        
        self.assertFloatEqual(21, A2[1,0])
        self.assertFloatEqual(31, A2[1,1])
        self.assertFloatEqual(41, A2[1,2])
        
        self.assertFloatEqual(12, A2[2,0])
        self.assertFloatEqual(32, A2[2,1])
        self.assertFloatEqual(52, A2[2,2])
        
        self.assertFloatEqual(22, A2[3,0])
        self.assertFloatEqual(42, A2[3,1])
        self.assertFloatEqual(62, A2[3,2])
        
    def assertFloatEqual(self, lhs, rhs):
        Format = "%7.3f"
        lhsStr = Format % lhs
        rhsStr = Format % rhs
        
        self.assertEqual(lhsStr, rhsStr)
        
    def testSparseVecTransF4(self):
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDenseVecTrans']
    unittest.main()