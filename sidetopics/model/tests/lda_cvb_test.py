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

import sidetopics.model.lda_cvb as lda
from sidetopics.model.common import DataSet
from sidetopics.model.evals import perplexity_from_like

AclPath = "/Users/bryanfeeney/iCloud/Datasets/ACL/ACL.100/"
AclWordPath = AclPath + "words-freq.pkl"
AclCitePath = AclPath + "ref.pkl"
AclDictPath = AclPath + "words-freq-dict.pkl"

ReutersPath = "/Users/bryanfeeney/Desktop/reuters/"
ReutersWordPath = ReutersPath + "W.pkl"
ReutersDictPath = ReutersPath + "dict.pkl"
ReutersCitePath = None

class Test(unittest.TestCase):


    def testPerplexityOnRealData(self):
        dtype = np.float32 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=ReutersWordPath, links_file=ReutersCitePath)
        with open(ReutersDictPath, "rb") as f:
            d = pkl.load(f)

        data.convert_to_dtype(np.int32)
        data.prune_and_shuffle(min_doc_len=50, min_link_count=2)
        # data.convert_to_undirected_graph()
        # data.convert_to_binary_link_matrix()

        # IDF frequency for when we print out the vocab later
        freq = np.squeeze(np.asarray(data.words.sum(axis=0)))
        scale = np.reciprocal(1 + freq)

        # Initialise the model
        K = 10
        model      = lda.newModelAtRandom(data, K, dtype=dtype)
        queryState = lda.newQueryState(data, model, debug=True)
        trainPlan  = lda.newTrainPlan(iterations=50, logFrequency=10, fastButInaccurate=False, debug=True)

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
        plt.imshow(vocab, interpolation="none", cmap = cm.Greys_r)
        plt.show()

        # Print out the most likely topic words
        topWordCount = 10
        kTopWordInds = [self.topWordInds(vocab[k,:], topWordCount) for k in range(K)]

        like = lda.log_likelihood(data, model, query)
        perp = perplexity_from_like(like, data.word_count)

        print ("Prior %s" % (str(model.topicPrior)))
        print ("Perplexity: %f\n\n" % perp)

        for k in range(model.K):
            print ("\nTopic %d\n=============================" % k)
            print ("\n".join("%-20s\t%0.4f" % (d[kTopWordInds[k][c]], vocab[k][kTopWordInds[k][c]]) for c in range(topWordCount)))


    def topWords (self, wordDict, vocab, count=10):
        return [wordDict[w] for w in self.topWordInds(wordDict, vocab, count)]


    def topWordInds (self, vocab, count=10):
        return vocab.argsort()[-count:][::-1]

    def printTopics(self, wordDict, vocab, count=10):
        words = vocab.argsort()[-count:][::-1]
        for wordIdx in words:
            print("%s" % wordDict[wordIdx])
        print("")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()