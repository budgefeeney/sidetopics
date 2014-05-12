'''
Created on 12 May 2014

@author: bryanfeeney
'''
import unittest
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle as pkl

import model.lda_cvb as lda
from model.lda_cvb import DTYPE

class Test(unittest.TestCase):


    def testOnRealData(self):
        rd.seed(0xBADB055)
        path = "/Users/bryanfeeney/Desktop/NIPS"
        with open(path + "/ar.pkl", 'rb') as f:
            _, W, _, d = pkl.load(f)
            
        if len(d) == 1:
            d = d[0]
        
        if W.dtype != DTYPE:
            W = W.astype(DTYPE)
        
        freq = np.squeeze(np.asarray(W.sum(axis=0)))
        scale = np.reciprocal(1 + freq)
       
        # Initialise the model  
        K = 10
        model      = lda.newModelAtRandom(W, K, dtype=DTYPE)
        queryState = lda.newQueryState(W, model)
        trainPlan  = lda.newTrainPlan(iterations=200, logFrequency=1, fastButInaccurate=False, debug=True)
        
        # Train the model, and the immediately save the result to a file for subsequent inspection
        model, query, (bndItrs, bndVals, bndLikes) = lda.train (W, None, model, queryState, trainPlan)
#        with open(newModelFileFromModel(model), "wb") as f:
#            pkl.dump ((model, query, (bndItrs, bndVals, bndLikes)), f)
        
        # Plot the evolution of the bound during training.
        fig, ax1 = plt.subplots()
        ax1.plot(bndItrs, bndVals, 'b-')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Bound', color='b')
        
        ax2 = ax1.twinx()
        ax2.plot(bndItrs, bndLikes, 'r-')
        ax2.set_ylabel('Likelihood', color='r')
        
        fig.show()
        plt.show()
        
        vocab = lda.vocab(model)
        plt.imshow(vocab, interpolation="none", cmap = cm.Greys_r)
        plt.show()
        
    
        # Print out the most likely topic words
        topWordCount = 100
        kTopWordInds = [self.topWordInds(d, vocab[k,:] * scale, topWordCount) \
                        for k in range(K)]
        
        print ("Perplexity: %f\n\n" % lda.perplexity(W, model, query))
        print ("\t\t".join (["Topic " + str(k) for k in range(K)]))
        print ("\n".join ("\t".join (d[kTopWordInds[k][c]] + "\t%0.4f" % vocab[k][kTopWordInds[k][c]] for k in range(K)) for c in range(topWordCount)))
        
    def topWords (self, wordDict, vocab, count=10):
        return [wordDict[w] for w in self.topWordInds(wordDict, vocab, count)]

    
    def topWordInds (self, wordDict, vocab, count=10):
        return vocab.argsort()[-count:][::-1]
    
    def printTopics(self, wordDict, vocab, count=10):
        words = vocab.argsort()[-count:][::-1]
        for wordIdx in words:
            print("%s" % wordDict[wordIdx])
        print("")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()