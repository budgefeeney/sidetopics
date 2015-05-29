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

import model.mtm as mtm
import model.mtm2 as mtm2
import model.ctm_bohning as ctm
import model.lda_vb_python as lda
from model.common import DataSet
from model.evals import perplexity_from_like, mean_average_prec

AclPath = "/Users/bryanfeeney/iCloud/Datasets/ACL/ACL.100/"
AclWordPath = AclPath + "words-freq.pkl"
AclCitePath = AclPath + "ref.pkl"
AclDictPath = AclPath + "words-freq-dict.pkl"

MinDocLen = 50
MinLinkCount = 2
Iters = 200
LogFreq = 5
TopicCount = 10

class MtmTest(unittest.TestCase):


    def topWords (self, wordDict, vocab, count=10):
        return [wordDict[w] for w in self.topWordInds(wordDict, vocab, count)]


    def topWordInds (self, vocab, count=10):
        return vocab.argsort()[-count:][::-1]

    def printTopics(self, wordDict, vocab, count=10):
        words = vocab.argsort()[-count:][::-1]
        for wordIdx in words:
            print("%s" % wordDict[wordIdx])
        print("")


    def testPerplexityOnRealData(self):
        dtype = np.float64 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=AclWordPath, links_file=AclCitePath)
        with open(AclDictPath, "rb") as f:
            d = pkl.load(f)

        data.convert_to_dtype(dtype)
        data.prune_and_shuffle(min_doc_len=MinDocLen, min_link_count=MinLinkCount)

        # IDF frequency for when we print out the vocab later
        freq = np.squeeze(np.asarray(data.words.sum(axis=0)))
        scale = np.reciprocal(1 + freq)

        # Initialise the model
        K = 50
        model      = mtm.newModelAtRandom(data, K, K - 1, dtype=dtype)
        queryState = mtm.newQueryState(data, model)
        trainPlan  = mtm.newTrainPlan(iterations=200, logFrequency=10, fastButInaccurate=False, debug=True)

        # Train the model, and the immediately save the result to a file for subsequent inspection
        model, query, (bndItrs, bndVals, bndLikes) = mtm.train (data, model, queryState, trainPlan)
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

        fig, ax1 = plt.subplots()
        ax1.imshow(model.topicCov, interpolation="nearest", cmap=cm.Greys_r)
        fig.show()
        plt.show()

        # Print out the most likely topic words
        # scale = np.reciprocal(1 + np.squeeze(np.array(data.words.sum(axis=0))))
        vocab = mtm.wordDists(model)
        topWordCount = 10
        kTopWordInds = [self.topWordInds(vocab[k,:], topWordCount) for k in range(K)]

        like = mtm.log_likelihood(data, model, query)
        perp = perplexity_from_like(like, data.word_count)

        print ("Prior %s" % (str(model.topicPrior)))
        print ("Perplexity: %f\n\n" % perp)

        for k in range(model.K):
            print("\nTopic %d\n=============================" % k)
            print("\n".join("%-20s\t%0.4f" % (d[kTopWordInds[k][c]], vocab[k][kTopWordInds[k][c]]) for c in range(topWordCount)))


    def testPerplexityOnRealDataWithMtm2(self):
        dtype = np.float64 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=AclWordPath, links_file=AclCitePath)
        with open(AclDictPath, "rb") as f:
            d = pkl.load(f)

        data.convert_to_dtype(dtype)
        data.prune_and_shuffle(min_doc_len=MinDocLen, min_link_count=MinLinkCount)

        # IDF frequency for when we print out the vocab later
        freq = np.squeeze(np.asarray(data.words.sum(axis=0)))
        scale = np.reciprocal(1 + freq)

        # Initialise the model
        K = 10 # TopicCount
        model      = mtm2.newModelAtRandom(data, K, dtype=dtype)
        queryState = mtm2.newQueryState(data, model)
        trainPlan  = mtm2.newTrainPlan(iterations=200, logFrequency=10, fastButInaccurate=False, debug=False)

        # Train the model, and the immediately save the result to a file for subsequent inspection
        model, query, (bndItrs, bndVals, bndLikes) = mtm2.train (data, model, queryState, trainPlan)
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

        fig, ax1 = plt.subplots()
        ax1.imshow(model.sigT, interpolation="nearest", cmap=cm.Greys_r)
        fig.show()
        plt.show()

        # Print out the most likely topic words
        # scale = np.reciprocal(1 + np.squeeze(np.array(data.words.sum(axis=0))))
        vocab = mtm2.wordDists(model)
        topWordCount = 10
        kTopWordInds = [self.topWordInds(vocab[k,:], topWordCount) for k in range(K)]

        like = mtm2.log_likelihood(data, model, query)
        perp = perplexity_from_like(like, data.word_count)

        print ("Perplexity: %f\n\n" % perp)

        for k in range(model.K):
            print("\nTopic %d\n=============================" % k)
            print("\n".join("%-20s\t%0.4f" % (d[kTopWordInds[k][c]], vocab[k][kTopWordInds[k][c]]) for c in range(topWordCount)))



    def testPerplexityOnRealDataWithCtm(self):
        dtype = np.float64 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=AclWordPath, links_file=AclCitePath)
        with open(AclDictPath, "rb") as f:
            d = pkl.load(f)

        data.convert_to_dtype(dtype)
        data.prune_and_shuffle(min_doc_len=MinDocLen, min_link_count=MinLinkCount)

        # IDF frequency for when we print out the vocab later
        freq = np.squeeze(np.asarray(data.words.sum(axis=0)))
        scale = np.reciprocal(1 + freq)

        # Initialise the model
        K = 10 # TopicCount
        model      = ctm.newModelAtRandom(data, K, dtype=dtype)
        queryState = ctm.newQueryState(data, model)
        trainPlan  = ctm.newTrainPlan(iterations=200, logFrequency=10, fastButInaccurate=False, debug=False)

        # Train the model, and the immediately save the result to a file for subsequent inspection
        model, query, (bndItrs, bndVals, bndLikes) = ctm.train (data, model, queryState, trainPlan)
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

        fig, ax1 = plt.subplots()
        ax1.imshow(model.sigT, interpolation="nearest", cmap=cm.Greys_r)
        fig.show()
        plt.show()

        # Print out the most likely topic words
        # scale = np.reciprocal(1 + np.squeeze(np.array(data.words.sum(axis=0))))
        vocab = ctm.wordDists(model)
        topWordCount = 10
        kTopWordInds = [self.topWordInds(vocab[k,:], topWordCount) for k in range(K)]

        like = ctm.log_likelihood(data, model, query)
        perp = perplexity_from_like(like, data.word_count)

        print ("Perplexity: %f\n\n" % perp)

        for k in range(model.K):
            print("\nTopic %d\n=============================" % k)
            print("\n".join("%-20s\t%0.4f" % (d[kTopWordInds[k][c]], vocab[k][kTopWordInds[k][c]]) for c in range(topWordCount)))


    def testPerplexityOnRealDataWithCtmInc(self):
        dtype = np.float64 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=AclWordPath, links_file=AclCitePath)
        with open(AclDictPath, "rb") as f:
            d = pkl.load(f)

        data.convert_to_dtype(dtype)
        data.prune_and_shuffle(min_doc_len=MinDocLen, min_link_count=MinLinkCount)

        # IDF frequency for when we print out the vocab later
        freq = np.squeeze(np.asarray(data.words.sum(axis=0)))
        scale = np.reciprocal(1 + freq)

        # Initialise the model
        topicCounts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        perps = []
        for K in topicCounts:
            model      = ctm.newModelAtRandom(data, K, dtype=dtype)
            queryState = ctm.newQueryState(data, model)
            trainPlan  = ctm.newTrainPlan(iterations=800, logFrequency=10, fastButInaccurate=False, debug=False)

            # Train the model, and the immediately save the result to a file for subsequent inspection
            model, query, (bndItrs, bndVals, bndLikes) = ctm.train (data, model, queryState, trainPlan)
    #        with open(newModelFileFromModel(model), "wb") as f:
    #            pkl.dump ((model, query, (bndItrs, bndVals, bndLikes)), f)

            # Print out the most likely topic words
            # scale = np.reciprocal(1 + np.squeeze(np.array(data.words.sum(axis=0))))
            vocab = ctm.wordDists(model)
            topWordCount = 10
            kTopWordInds = [self.topWordInds(vocab[k,:], topWordCount) for k in range(K)]

            like = ctm.log_likelihood(data, model, query)
            perp = perplexity_from_like(like, data.word_count)

            perps.append(perp)

            print ("K = %2d : Perplexity = %f\n\n" % (K, perp))
            #
            # for k in range(model.K):
            #     print("\nTopic %d\n=============================" % k)
            #     print("\n".join("%-20s\t%0.4f" % (d[kTopWordInds[k][c]], vocab[k][kTopWordInds[k][c]]) for c in range(topWordCount)))

        # Plot the evolution of the bound during training.
        fig, ax1 = plt.subplots()
        ax1.plot(topicCounts, perps, 'b-')
        ax1.set_xlabel('Topic Count')
        ax1.set_ylabel('Perplexity', color='b')

        fig.show()
        plt.show()


    def testPerplexityOnRealDataWithLdaInc(self):
        dtype = np.float64 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=AclWordPath, links_file=AclCitePath)
        with open(AclDictPath, "rb") as f:
            d = pkl.load(f)

        data.convert_to_dtype(dtype)
        data.prune_and_shuffle(min_doc_len=MinDocLen, min_link_count=MinLinkCount)

        # IDF frequency for when we print out the vocab later
        freq = np.squeeze(np.asarray(data.words.sum(axis=0)))
        scale = np.reciprocal(1 + freq)

        # Initialise the model
        topicCounts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        perps = []
        for K in topicCounts:
            model      = lda.newModelAtRandom(data, K, dtype=dtype)
            queryState = lda.newQueryState(data, model)
            trainPlan  = lda.newTrainPlan(iterations=800, logFrequency=10, fastButInaccurate=False, debug=False)

            # Train the model, and the immediately save the result to a file for subsequent inspection
            model, query, (bndItrs, bndVals, bndLikes) = lda.train (data, model, queryState, trainPlan)
    #        with open(newModelFileFromModel(model), "wb") as f:
    #            pkl.dump ((model, query, (bndItrs, bndVals, bndLikes)), f)

            # Print out the most likely topic words
            # scale = np.reciprocal(1 + np.squeeze(np.array(data.words.sum(axis=0))))
            # vocab = lda.wordDists(model)
            # topWordCount = 10
            # kTopWordInds = [self.topWordInds(vocab[k,:], topWordCount) for k in range(K)]

            like = lda.log_likelihood(data, model, query)
            perp = perplexity_from_like(like, data.word_count)

            perps.append(perp)

            print ("K = %2d : Perplexity = %f\n\n" % (K, perp))
            #
            # for k in range(model.K):
            #     print("\nTopic %d\n=============================" % k)
            #     print("\n".join("%-20s\t%0.4f" % (d[kTopWordInds[k][c]], vocab[k][kTopWordInds[k][c]]) for c in range(topWordCount)))

        # Plot the evolution of the bound during training.
        fig, ax1 = plt.subplots()
        ax1.plot(topicCounts, perps, 'b-')
        ax1.set_xlabel('Topic Count')
        ax1.set_ylabel('Perplexity', color='b')

        fig.show()
        plt.show()