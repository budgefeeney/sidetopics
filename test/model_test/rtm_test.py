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

import model.rtm as rtm
from model.common import DataSet
from model.evals import perplexity_from_like, mean_average_prec

AclPath = "/Users/bryanfeeney/iCloud/Datasets/ACL/ACL.100/"
AclWordPath = AclPath + "words-freq.pkl"
AclCitePath = AclPath + "ref.pkl"
AclDictPath = AclPath + "words-freq-dict.pkl"

MinDocLen = 50
MinLinkCount = 8
Iters = 200
LogFreq = 5
TopicCount = 10

class RtmTest(unittest.TestCase):

    def topWords (self, wordDict, vocab, count=10):
        return [wordDict[w] for w in self.topWordInds(wordDict, vocab, count)]


    def topWordInds (self, vocab, count=10):
        return vocab.argsort()[-count:][::-1]

    def printTopics(self, wordDict, vocab, count=10):
        words = vocab.argsort()[-count:][::-1]
        for wordIdx in words:
            print("%s" % wordDict[wordIdx])
        print("")


    def _testPerplexityOnRealData(self):
        dtype = np.float64 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=AclWordPath, links_file=AclCitePath)
        with open(AclDictPath, "rb") as f:
            dic = pkl.load(f)

        data.convert_to_dtype(dtype)
        data.convert_to_undirected_graph()
        data.convert_to_binary_link_matrix()
        data.prune_and_shuffle(min_doc_len=MinDocLen, min_link_count=MinLinkCount)

        # IDF frequency for when we print out the vocab later
        freq = np.squeeze(np.asarray(data.words.sum(axis=0)))
        scale = np.reciprocal(1 + freq)

        # Initialise the model
        K = TopicCount
        model      = rtm.newModelAtRandom(data, K, dtype=dtype)
        queryState = rtm.newQueryState(data, model)
        trainPlan  = rtm.newTrainPlan(iterations=50, logFrequency=LogFreq, fastButInaccurate=False, debug=True)

        # Train the model, and the immediately save the result to a file for subsequent inspection
        model, query, (bndItrs, bndVals, bndLikes) = rtm.train (data, model, queryState, trainPlan)
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

        # Print out the most likely topic words
        # scale = np.reciprocal(1 + np.squeeze(np.array(data.words.sum(axis=0))))
        vocab = rtm.wordDists(model)
        topWordCount = 10
        kTopWordInds = [self.topWordInds(vocab[k,:], topWordCount) for k in range(K)]

        like = rtm.log_likelihood(data, model, query)
        perp = perplexity_from_like(like, data.word_count)

        print ("Prior %s" % (str(model.topicPrior)))
        print ("Perplexity: %f\n\n" % perp)

        for k in range(model.K):
            print ("\nTopic %d\n=============================" % k)
            print ("\n".join("%-20s\t%0.4f" % (d[kTopWordInds[k][c]], vocab[k][kTopWordInds[k][c]]) for c in range(topWordCount)))


    def testMapOnRealData(self):
        dtype = np.float64 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=AclWordPath, links_file=AclCitePath)
        with open(AclDictPath, "rb") as f:
            dic = pkl.load(f)

        data.convert_to_dtype(dtype)
        data.convert_to_undirected_graph()
        data.convert_to_binary_link_matrix()
        data.prune_and_shuffle(min_doc_len=MinDocLen, min_link_count=MinLinkCount)

        trainData, testData = data.doc_completion_split()

        for pseudoNegCount in (5, 10, 25, 50, 100):
            rd.seed(0xC0FFEE)

            # Initialise the model
            K = TopicCount
            model      = rtm.newModelAtRandom(trainData, K, dtype=dtype, pseudoNegCount=trainData.doc_count * pseudoNegCount)
            queryState = rtm.newQueryState(trainData, model)
            trainPlan  = rtm.newTrainPlan(iterations=50, logFrequency=LogFreq, fastButInaccurate=False, debug=False)

            # Train the model, and the immediately save the result to a file for subsequent inspection
            model, topics, (bndItrs, bndVals, bndLikes) = rtm.train(trainData, model, queryState, trainPlan)
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

            # Print out the most likely topic words
            # scale = np.reciprocal(1 + np.squeeze(np.array(data.words.sum(axis=0))))
            vocab = rtm.wordDists(model)
            topWordCount = 10
            kTopWordInds = [self.topWordInds(vocab[k, :], topWordCount) for k in range(K)]

            like = rtm.log_likelihood(trainData, model, topics)
            perp = perplexity_from_like(like, trainData.word_count)

            # print ("Prior %s" % (str(model.topicPrior)))
            print ("Pseudo Neg-Count: %d " % pseudoNegCount)
            print ("\tTrain Perplexity: %f\n\n" % perp)

            # for k in range(model.K):
            #     print ("\nTopic %d\n=============================" % k)
            #     print ("\n".join("%-20s\t%0.4f" % (d[kTopWordInds[k][c]], vocab[k][kTopWordInds[k][c]]) for c in range(topWordCount)))

            min_probs  = rtm.min_link_probs(model, topics, testData.links)
            link_probs = rtm.link_probs(model, topics, min_probs)
            try:
                map = mean_average_prec(testData.links, link_probs)
            except:
                print ("Unexpected error")

            print("\tThe Mean-Average-Precision is %.3f" % map)


#

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()