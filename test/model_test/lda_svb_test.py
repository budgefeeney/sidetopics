# -*- coding: utf-8 -*-

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

import model.lda_vb_python as lda
from model.common import DataSet
from model.evals import word_perplexity

NipsPath='/Users/bryanfeeney/Desktop/NIPS/'
NipsWordsPath=NipsPath + 'W_ar.pkl'
NipsCitePath=NipsPath + 'X_ar.pkl'
NipsDictPath=NipsPath + 'words.pkl'

class Test(unittest.TestCase):


    def testOnRealData(self):
        dtype = np.float64 # DTYPE
        
        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=NipsWordsPath, links_file=NipsCitePath)
        with open(NipsDictPath, "rb") as f:
            d = pkl.load(f)

        data.convert_to_dtype(dtype)
        data.prune_and_shuffle(min_doc_len=50, min_link_count=0)
        
        # IDF frequency for when we print out the vocab later
        freq = np.squeeze(np.asarray(data.words.sum(axis=0)))
        scale = np.reciprocal(1 + freq)
       
        # Initialise the model  
        K = 10
        model      = lda.newModelAtRandom(data, K, dtype=dtype)
        queryState = lda.newQueryState(data, model)
        trainPlan  = lda.newTrainPlan(iterations=10, logFrequency=2, debug=True, batchSize=5, rate_retardation=1, forgetting_rate=0.75)
        
        # Train the model, and the immediately save the result to a file for subsequent inspection
        model, query, (bndItrs, bndVals, bndLikes) = lda.train (data, model, queryState, trainPlan)
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
        
        vocab = lda.wordDists(model)
        plt.imshow(vocab, interpolation="nearest", cmap=cm.Greys_r)
        plt.show()
            
        # Print out the most likely topic words
        topWordCount = 100
        kTopWordInds = [topWordIndices(vocab[k, :] * scale, topWordCount) \
                        for k in range(K)]
        
        print ("Prior %s" % (str(model.topicPrior)))
        print ("Perplexity: %f\n\n" % word_perplexity(lda.log_likelihood, model, query, data))
        print ("\t\t".join (["Topic " + str(k) for k in range(K)]))
        print ("\n".join ("\t".join (d[kTopWordInds[k][c]] + "\t%0.4f" % vocab[k][kTopWordInds[k][c]] for k in range(K)) for c in range(topWordCount)))
        
def topWords (wordDict, vocab, count=10):
    return [wordDict[w] for w in topWordIndices(vocab, count)]

def topWordIndices (vocab, count=10):
    return vocab.argsort()[-count:][::-1]

def printTopics(wordDict, vocab, count=10):
    words = vocab.argsort()[-count:][::-1]
    for wordIdx in words:
        print("%s" % wordDict[wordIdx])
    print("")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()