'''
Created on 17 Jan 2014

@author: bryanfeeney
'''
from util.array_utils import normalizerows_ip, rowwise_softmax
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import model.ctm as ctm
import numpy as np
import numpy.random as rd
import scipy as sp
import scipy.linalg as la
import scipy.sparse as ssp
import scipy.sparse.linalg as sla
import unittest



DTYPE=np.float32

class Test(unittest.TestCase):


    def testOnModelHandcraftedData(self):
        #
        # Create the vocab
        #
        T = 3 * 3
        K = 8
        
        # Horizontal bars
        vocab1 = ssp.coo_matrix(([1, 1, 1], ([0, 0, 0], [0, 1, 2])), shape=(3,3)).todense()
        vocab2 = ssp.coo_matrix(([1, 1, 1], ([1, 1, 1], [0, 1, 2])), shape=(3,3)).todense()
        vocab3 = ssp.coo_matrix(([1, 1, 1], ([2, 2, 2], [0, 1, 2])), shape=(3,3)).todense()
        
        # Vertical bars
        vocab4 = ssp.coo_matrix(([1, 1, 1], ([0, 1, 2], [0, 0, 0])), shape=(3,3)).todense()
        vocab5 = ssp.coo_matrix(([1, 1, 1], ([0, 1, 2], [1, 1, 1])), shape=(3,3)).todense()
        vocab6 = ssp.coo_matrix(([1, 1, 1], ([0, 1, 2], [2, 2, 2])), shape=(3,3)).todense()
        
        # Diagonals
        vocab7 = ssp.coo_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])), shape=(3,3)).todense()
        vocab8 = ssp.coo_matrix(([1, 1, 1], ([2, 1, 0], [0, 1, 2])), shape=(3,3)).todense()
        
        # Put together
        T = vocab1.shape[0] * vocab1.shape[1]
        vocabs = [vocab1, vocab2, vocab3, vocab4, vocab5, vocab6, vocab7, vocab8]
        
        # Create a single matrix with the flattened vocabularies
        vocabVectors = []
        for vocab in vocabs:
            vocabVectors.append (np.squeeze(np.asarray (vocab.reshape((1,T)))))
        
        vocab = normalizerows_ip(np.array(vocabVectors, dtype=DTYPE))
        
        # Plot the vocab
        ones = np.ones(vocabs[0].shape)
        for k in range(K):
            plt.subplot(3, 3, k)
            plt.imshow(ones - vocabs[k], interpolation="none", cmap = cm.Greys_r)
        plt.show()
        
        #
        # Create the corpus
        #
        rd.seed(0xC0FFEE)
        D = 100

        # Make sense (of a sort) of this by assuming that these correspond to
        # Kittens    Omelettes    Puppies    Oranges    Tomatoes    Dutch People    Basketball    Football
        topicMean = np.array([10, 25, 5, 15, 5, 5, 10, 25])
        topicCovar = np.array(\
            [[ 100,    5,     55,      20,     5,     15,      4,      0], \
             [ 5,    100,      5,      10,    70,      5,      0,      0], \
             [ 55,     5,    100,       5,     5,     10,      0,      5], \
             [ 20,    10,      5,     100,    30,     30,     20,     10], \
             [ 5,     70,      5,     30,    100,      0,      0,      0], \
             [ 15,     5,     10,     30,      0,    100,     10,     40], \
             [ 4,      0,      0,     20,      0,     10,    100,     20], \
             [ 0,      0,      5,     10,      0,     40,     20,    100]], dtype=DTYPE) / 100.0
        
        meanWordCount = 150
        wordCounts = rd.poisson(meanWordCount, size=D)
        topicDists = rd.multivariate_normal(topicMean, topicCovar, size=D)
        W = topicDists.dot(vocab) * wordCounts[:, np.newaxis]
        W = ssp.csr_matrix (W.astype(DTYPE))
        
        #
        # Train the model
        #
        model      = ctm.newModelAtRandom(W, K, dtype=DTYPE)
        queryState = ctm.newQueryState(W, model)
        trainPlan  = ctm.newTrainPlan(plot=True, logFrequency=1)
        
        self.assertTrue (0.99 < np.sum(model.topicMean) < 1.01)
        
        return self._doTest (W, model, queryState, trainPlan)
    
    
    def _doTest (self, W, model, queryState, trainPlan):
        print ("Initial bound is %f\n\n" % ctm.var_bound(W, model, queryState))
        
        model, query = ctm.train (W, model, queryState, trainPlan)
        
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()