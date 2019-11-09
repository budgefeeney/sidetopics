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

import sidetopics.model.mtm as mtm
import sidetopics.model.mtm2 as mtm2
import sidetopics.model.ctm_bohning as ctm
import sidetopics.model.lda_vb_python as lda
import sidetopics.model.lda_vb as lda_old
import sidetopics.model.lda_gibbs as lda_gibbs
from sidetopics.model.common import DataSet
from sidetopics.model.evals import perplexity_from_like, mean_average_prec

AclPath = "/Users/bryanfeeney/iCloud/Datasets/ACL/ACL.100/"
AclWordPath = AclPath + "words-freq.pkl"
AclCitePath = AclPath + "ref.pkl"
AclDictPath = AclPath + "words-freq-dict.pkl"

MinDocLen = 50
MinLinkCount = 2
Iters = 200
LogFreq = 5
TopicCount = 10

NumFolds = 5

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
        K = 30 # TopicCount
        model      = mtm2.newModelAtRandom(data, K, dtype=dtype)
        queryState = mtm2.newQueryState(data, model)
        trainPlan  = mtm2.newTrainPlan(iterations=200, logFrequency=10, fastButInaccurate=False, debug=False)

        # Train the model, and the immediately save the result to a file for subsequent inspection
        model, query, (bndItrs, bndVals, bndLikes) = mtm2.train(data, model, queryState, trainPlan)
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
        vocab = mtm2.wordDists(model)
        topWordCount = 10
        kTopWordInds = [self.topWordInds(vocab[k,:], topWordCount) for k in range(K)]

        like = mtm2.log_likelihood(data, model, query)
        perp = perplexity_from_like(like, data.word_count)

        print("Perplexity: %f\n\n" % perp)

        for k in range(model.K):
            print("\nTopic %d\n=============================" % k)
            print("\n".join("%-20s\t%0.4f" % (d[kTopWordInds[k][c]], vocab[k][kTopWordInds[k][c]]) for c in range(topWordCount)))

        print ("Most likely documents for each topic")
        print ("====================================")
        with open ("/Users/bryanfeeney/iCloud/Datasets/ACL/ACL.100/doc_ids.pkl", 'rb') as f:
            fileIds = pkl.load (f)
        docs_dict = [fileIds[fi] for fi in data.order]

        for k in range(model.K):
            arg_max_prob = np.argmax(query.means[:, k])
            print("K=%2d  Document ID = %s (found at %d)" % (k, docs_dict[arg_max_prob], arg_max_prob))

        print ("Done")

        with open ("/Users/bryanfeeney/Desktop/mtm2-" + str(K) + ".pkl", "wb") as f:
            pkl.dump((model, query), f)

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


    def testCrossValPerplexityOnRealDataWithCtmInc(self):
        dtype = np.float64 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=AclWordPath, links_file=AclCitePath)

        data.convert_to_dtype(dtype)
        data.prune_and_shuffle(min_doc_len=MinDocLen, min_link_count=MinLinkCount)

        # Initialise the model
        trainPlan = ctm.newTrainPlan(iterations=800, logFrequency=10, fastButInaccurate=False, debug=False)
        queryPlan = ctm.newTrainPlan(iterations=100, logFrequency=10, fastButInaccurate=False, debug=False)

        topicCounts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        for K in topicCounts:
            trainPerps = []
            queryPerps = []
            for fold in range(1): # range(NumFolds):
                trainData, queryData = data.cross_valid_split(fold, NumFolds)

                model = ctm.newModelAtRandom(trainData, K, dtype=dtype)
                query = ctm.newQueryState(trainData, model)

                # Train the model, and the immediately save the result to a file for subsequent inspection
                model, trainResult, (_, _, _) = ctm.train (trainData, model, query, trainPlan)

                like = ctm.log_likelihood(trainData, model, trainResult)
                perp = perplexity_from_like(like, trainData.word_count)
                trainPerps.append(perp)

                query = ctm.newQueryState(queryData, model)
                model, queryResult = ctm.query(queryData, model, query, queryPlan)

                like = ctm.log_likelihood(queryData, model, queryResult)
                perp = perplexity_from_like(like, queryData.word_count)
                queryPerps.append(perp)

            trainPerps.append(sum(trainPerps) / NumFolds)
            queryPerps.append(sum(queryPerps) / NumFolds)
            print("K=%d,Segment=Train,%s" % (K, ",".join([str(p) for p in trainPerps])))
            print("K=%d,Segment=Query,%s" % (K, ",".join([str(p) for p in queryPerps])))



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


    def testCrossValPerplexityOnRealDataWithLdaInc(self):
        ActiveFolds = 3
        dtype = np.float64 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=AclWordPath, links_file=AclCitePath)

        data.convert_to_dtype(dtype)
        data.prune_and_shuffle(min_doc_len=MinDocLen, min_link_count=MinLinkCount)

        # Initialise the model
        trainPlan = lda.newTrainPlan(iterations=800, logFrequency=10, fastButInaccurate=False, debug=False)
        queryPlan = lda.newTrainPlan(iterations=50, logFrequency=5, fastButInaccurate=False, debug=False)

        topicCounts = [30, 35, 40, 45, 50] # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        for K in topicCounts:
            trainPerps = []
            queryPerps = []
            for fold in range(ActiveFolds): # range(NumFolds):
                trainData, queryData = data.cross_valid_split(fold, NumFolds)

                model = lda.newModelAtRandom(trainData, K, dtype=dtype)
                query = lda.newQueryState(trainData, model)

                # Train the model, and the immediately save the result to a file for subsequent inspection
                model, trainResult, (_, _, _) = lda.train (trainData, model, query, trainPlan)

                like = lda.log_likelihood(trainData, model, trainResult)
                perp = perplexity_from_like(like, trainData.word_count)
                trainPerps.append(perp)

                estData, evalData = queryData.doc_completion_split()
                query = lda.newQueryState(estData, model)
                model, queryResult = lda.query(estData, model, query, queryPlan)

                like = lda.log_likelihood(evalData, model, queryResult)
                perp = perplexity_from_like(like, evalData.word_count)
                queryPerps.append(perp)

            trainPerps.append(sum(trainPerps) / ActiveFolds)
            queryPerps.append(sum(queryPerps) / ActiveFolds)
            print("K=%d,Segment=Train,%s" % (K, ",".join([str(p) for p in trainPerps])))
            print("K=%d,Segment=Query,%s" % (K, ",".join([str(p) for p in queryPerps])))


    def testCrossValPerplexityOnRealDataWithLdaOldInc(self):
        ActiveFolds = 3
        dtype = np.float64 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=AclWordPath, links_file=AclCitePath)

        data.convert_to_dtype(dtype)
        data.prune_and_shuffle(min_doc_len=MinDocLen, min_link_count=MinLinkCount)

        # Initialise the model
        trainPlan = lda_old.newTrainPlan(iterations=800, logFrequency=200, fastButInaccurate=False, debug=False)
        queryPlan = lda_old.newTrainPlan(iterations=24, logFrequency=12, fastButInaccurate=False, debug=False)

        topicCounts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        for K in topicCounts:
            trainPerps = []
            queryPerps = []
            for fold in range(ActiveFolds): # range(NumFolds):
                trainData, queryData = data.cross_valid_split(fold, NumFolds)

                model = lda_old.newModelAtRandom(trainData, K, dtype=dtype)
                query = lda_old.newQueryState(trainData, model)

                # Train the model, and the immediately save the result to a file for subsequent inspection
                model, trainResult, (_, _, _) = lda_old.train (trainData, model, query, trainPlan)

                like = lda_old.log_likelihood(trainData, model, trainResult)
                perp = perplexity_from_like(like, trainData.word_count)
                trainPerps.append(perp)

                query = lda_old.newQueryState(queryData, model)
                model, queryResult = lda_old.query(queryData, model, query, queryPlan)

                like = lda_old.log_likelihood(queryData, model, queryResult)
                perp = perplexity_from_like(like, queryData.word_count)
                queryPerps.append(perp)

            trainPerps.append(sum(trainPerps) / ActiveFolds)
            queryPerps.append(sum(queryPerps) / ActiveFolds)
            print("K=%d,Segment=Train,%s" % (K, ",".join([str(p) for p in trainPerps])))
            print("K=%d,Segment=Query,%s" % (K, ",".join([str(p) for p in queryPerps])))


    def testCrossValPerplexityOnRealDataWithLdaGibbsInc(self):
        ActiveFolds = 3
        dtype = np.float64 # DTYPE

        rd.seed(0xBADB055)
        data = DataSet.from_files(words_file=AclWordPath, links_file=AclCitePath)

        data.convert_to_dtype(np.int32) # Gibbs expects integers as input, regardless of model dtype
        data.prune_and_shuffle(min_doc_len=MinDocLen, min_link_count=MinLinkCount)

        # Training setup
        TrainSamplesPerTopic = 10
        QuerySamplesPerTopic = 2
        Thin = 2
        Debug = False

        # Start running experiments
        topicCounts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        for K in topicCounts:
            trainPlan = lda_gibbs.newTrainPlan(K * TrainSamplesPerTopic, thin=Thin, debug=Debug)
            queryPlan = lda_gibbs.newTrainPlan(K * QuerySamplesPerTopic, thin=Thin, debug=Debug)

            trainPerps = []
            queryPerps = []
            for fold in range(ActiveFolds): # range(NumFolds):
                trainData, queryData = data.cross_valid_split(fold, NumFolds)
                estData, evalData = queryData.doc_completion_split()

                model = lda_gibbs.newModelAtRandom(trainData, K, dtype=dtype)
                query = lda_gibbs.newQueryState(trainData, model)

                # Train the model, and the immediately save the result to a file for subsequent inspection
                model, trainResult, (_, _, _) = lda_gibbs.train (trainData, model, query, trainPlan)

                like = lda_gibbs.log_likelihood(trainData, model, trainResult)
                perp = perplexity_from_like(like, trainData.word_count)
                trainPerps.append(perp)

                query = lda_gibbs.newQueryState(estData, model)
                _, queryResult = lda_gibbs.query(estData, model, query, queryPlan)

                like = lda_gibbs.log_likelihood(evalData, model, queryResult)
                perp = perplexity_from_like(like, evalData.word_count)
                queryPerps.append(perp)

            trainPerps.append(sum(trainPerps) / ActiveFolds)
            queryPerps.append(sum(queryPerps) / ActiveFolds)
            print("K=%d,Segment=Train,%s" % (K, ",".join([str(p) for p in trainPerps])))
            print("K=%d,Segment=Query,%s" % (K, ",".join([str(p) for p in queryPerps])))