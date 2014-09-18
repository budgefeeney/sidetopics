'''
Created on 26 Nov 2013

@author: bryanfeeney
'''
import argparse as ap
import pickle as pkl
import numpy as np
import numpy.random as rd
import scipy.sparse as ssp
import sys
import time
import os
from math import ceil

DTYPE=np.float32
    
CtmBouchard   = 'ctm_bouchard'
CtmBohning    = 'ctm_bohning'
StmYvBouchard = 'stm_yv_bouchard'
StmYvBohning  = 'stm_yv_bohning'
LdaCvbZero    = 'lda_cvb0'
LdaVb         = 'lda_vb'

ModelNames = ', '.join([CtmBouchard, CtmBohning, StmYvBouchard, StmYvBohning, LdaCvbZero, LdaVb])


def run(args):
    '''
    Parses the command-line arguments (excluding the application name portion). 
    Executes a cross-validation run accordingly, saving the output at the end
    of each run.
    
    Returns the list of files created.
    '''
    
    #
    # Enumerate all possible arguments
    #
    parser = ap.ArgumentParser(description='Execute a topic-modeling run.')
    parser.add_argument('--model', '-m', dest='model', metavar=' ', \
                    help='The type of mode to use, options are ' + ModelNames)
    parser.add_argument('--num-topics', '-k', dest='K', type=int, metavar=' ', \
                    help='The number of topics to fit')
    parser.add_argument('--num-lat-topics', '-q', dest='Q', type=int, metavar=' ', \
                    help='The number of latent topics (i.e. rank of the topic covariance matrix)')
    parser.add_argument('--num-lat-feats', '-p', dest='P', type=int, metavar=' ', \
                    help='The number of latent features (i.e. rank of the features covariance matrix)')
    parser.add_argument('--words', '-w', dest='words', metavar=' ', \
                    help='The path to the pickle file containing a DxT array or matrix of the word-counts across all D documents')
    parser.add_argument('--feats', '-x', dest='feats', metavar=' ', \
                    help='The path to the pickle file containing a DxF array or matrix of the features across all D documents')
    parser.add_argument('--eval', '-v', dest='eval', default="perplexity", metavar=' ', \
                    help='Evaluation metric, only available is: perplexity or likelihood')
    parser.add_argument('--out-model', '-o', dest='out_model', default=None, metavar=' ', \
                    help='Optional output path in which to store the model')
    parser.add_argument('--log-freq', '-l', dest='log_freq', type=int, default=10, metavar=' ', \
                    help='Log frequency - how many times to inspect the bound while running')
    parser.add_argument('--iters', '-i', dest='iters', type=int, default=500, metavar=' ', \
                    help='The maximum number of iterations to run when training')
    parser.add_argument('--query-iters', '-j', dest='query_iters', type=int, default=100, metavar=' ', \
                    help='The maximum number of iterations to run when querying, by default same as when training')
    parser.add_argument('--min-vb-change', '-e', dest='min_vb_change', type=float, default=1, metavar=' ', \
                    help='The amount by which the variational bound must change at each log-interval to avoid inference being stopped early.')
    parser.add_argument('--topic-var', dest='topic_var', type=float, default=0.1, metavar=' ', \
                    help="Scale of the prior isotropic variance over topics")
    parser.add_argument('--feat-var', dest='feat_var', type=float, default=0.1, metavar=' ', \
                    help="Scale of the prior isotropic variance over features")
    parser.add_argument('--lat-topic-var', dest='lat_topic_var', type=float, default=0.1, metavar=' ', \
                    help="Scale of the prior isotropic variance over latent topics")
    parser.add_argument('--lat-feat-var', dest='lat_feat_var', type=float, default=0.1, metavar=' ', \
                    help="Scale of the prior isotropic variance over latent features")
    parser.add_argument('--folds', '-f', dest='folds', type=int, default=1, metavar=' ', \
                    help="Number of cross validation folds.")
    parser.add_argument('--debug', '-b', dest='debug', type=bool, default=False, metavar=' ', \
                    help="Display a debug message, with the bound, after every variable update")
    parser.add_argument('--dtype', '-t', dest='dtype', default="f4", metavar=' ', \
                    help="Datatype to use, values are f4 and f8 for single and double-precision floats respectively")
    
    #
    # Parse the arguments
    #
    print ("Args are : " + str(args))
    args = parser.parse_args(args)
    
    print ("Random seed is 0xC0FFEE")
    rd.seed(0xC0FFEE)
    
    dtype = np.float32 if args.dtype=='f4' else np.float64
    
    #
    # Load in the files. As the cross-validation slices aren't randomized, we
    # randomly re-order the data to help ensure that there's no patterns in the
    # data that might hurt on querying 
    #
    with open(args.words, 'rb') as f:
        W = pkl.load(f)
        
        #
        # <Debug>, use bagging to increase the dataset twenty-fold
        #
#         BagFactor = 20
#         base = W
#         W = np.ndarray(shape=(W.shape[0] * BagFactor, W.shape[1]), dtype=np.int32)
#         for rowId in range(base.shape[0]):
#             row = np.squeeze (np.asarray(base[rowId,:].astype(np.float64).todense()))
#             cnt = row.sum()
#             row += 0.1
#             
#             for offset in range(BagFactor):
#                 dist = rd.dirichlet(row)
#                 #dist = row
#                 newCnt = rd.poisson(cnt)
#                 W[rowId * BagFactor + offset,:] = rd.multinomial(newCnt, dist).astype(np.int32)
#                 
#         W = ssp.csr_matrix(W).astype(dtype)
#         
        #
        # </Debug>
        #
        
        docLens = np.squeeze(np.asarray(W.sum(axis=1)))
        if docLens.min() < 0.5: # docLens should be integers, but in case it's floats which don't add up accurately use 0.5
            print("Input doc-term matrix contains some empty rows. These have been removed.")
            good_rows = (np.where(docLens > 0.5))[0]
            D = len(good_rows)
            order = good_rows
        else:
            D = W.shape[0]
            order = np.linspace(0, D - 1, D)
        
        rd.shuffle(order)
        
        W = W[order,:].astype(dtype)
    if args.feats is None:
        X = None
        F = 0
    else:
        with open(args.feats, 'rb') as f:
            X = pkl.load(f)
            X = X[order,:].astype(dtype)
            F = X.shape[1]
            
    K     = args.K
    P     = args.P
    Q     = args.Q
    folds = args.folds
    
    fv, tv, lfv, ltv = args.feat_var, args.topic_var, args.lat_feat_var, args.lat_topic_var
    
    #
    # Instantiate and configure the model
    #    
    if args.model == CtmBouchard:
        import model.ctm as mdl
        templateModel = mdl.newModelAtRandom(W, K, dtype=dtype)
    elif args.model == CtmBohning:
        import model.ctm_bohning as mdl
        templateModel = mdl.newModelAtRandom(W, K, dtype=dtype)
    elif args.model == StmYvBouchard:
        import model.stm_yv as mdl 
        templateModel = mdl.newModelAtRandom(X, W, P, K, fv, lfv, dtype=dtype)
    elif args.model == StmYvBohning:
        import model.stm_yv_bohning as mdl 
        templateModel = mdl.newModelAtRandom(X, W, P, K, fv, lfv, dtype=dtype)
    elif args.model == LdaCvbZero:
        import model.lda_cvb as mdl
        templateModel = mdl.newModelAtRandom(W, K, dtype=dtype)
    elif args.model == LdaVb:
        import model.lda_vb as mdl
        templateModel = mdl.newModelAtRandom(W, K, dtype=dtype)
    else:
        raise ValueError ("Unknown model identifier " + args.model)
    
    trainPlan = mdl.newTrainPlan(args.iters, args.min_vb_change, args.log_freq, debug=args.debug)
    queryPlan = mdl.newTrainPlan(args.query_iters, args.min_vb_change, args.log_freq, debug=args.debug)
    
    # things to inspect and store for later use
    modelFiles = []

    # Run the model on each fold
    if folds == 1:
        try:
            model = mdl.newModelFromExisting(templateModel)
            query = mdl.newQueryState(W, model)
            
            model, query, (boundItrs, boundVals, boundLikes) = mdl.train (W, X, model, query, trainPlan)
            trainSetLikely = mdl.log_likelihood (W, model, query)
            perp = np.exp (-trainSetLikely / W.data.sum())
            
            print("Train-set Likelihood: %12f" % (trainSetLikely))
            print("Train-set Perplexity: %12f" % (perp))
            print("")
        finally:
            # Write out the end result of the model run.
            if args.out_model is not None:
                modelFile = newModelFile(args.model, args.K, args.P, fold=None, prefix=args.out_model)
                modelFiles.append(modelFile)
                with open(modelFile, 'wb') as f:
                    pkl.dump ((order, boundItrs, boundVals, model, query, None), f)
    else:
        queryLikelySum = 0 # to calculate the overall likelihood and
        queryWordsSum  = 0 # perplexity for the whole dataset
        trainLikelySum = 0
        trainWordsSum  = 0
        finishedFolds  = 0 # count of folds that finished successfully
        
        foldSize  = ceil(D / folds)
        querySize = foldSize
        trainSize = D - querySize
        
        for fold in range(folds):
            try:
                # Split the datasets up for the current fold
                start = fold * foldSize
                end   = start + trainSize
                
                trainSet = np.arange(start,end) % D
                querySet = np.arange(end, end + querySize) % D
                
                W_train, W_query = W[trainSet,:], W[querySet,:]
                if X is not None:
                    X_train, X_query = X[trainSet,:], X[querySet,:]
                else:
                    X_train, X_query = None, None
                
                # Train the model
                modelState  = mdl.newModelFromExisting(templateModel)
                trainTopics = mdl.newQueryState(W_train, modelState)
                modelState, trainTopics, (boundItrs, boundVals, boundLikes) \
                    = mdl.train(W_train, X_train, modelState, trainTopics, trainPlan)
                    
                trainSetLikely = mdl.log_likelihood (W_train, modelState, trainTopics)
                trainWords     = W_train.data.sum()
                trainSetPerp   = np.exp(-trainSetLikely / trainWords)
                
                # Query the model - if there are no features we need to split the text
                W_query_train, W_query_eval = splitInput(X, W_query)
                queryTopics = mdl.newQueryState(W_query_train, modelState)
                modelState, queryTopics = mdl.query(W_query_train, X_query, modelState, queryTopics, queryPlan)
                
                querySetLikely = mdl.log_likelihood(W_query_eval, modelState, queryTopics)
                queryWords     = W_query_eval.data.sum()
                querySetPerp   = np.exp(-querySetLikely / queryWords)
                
                # Keep a record of the cumulative likelihood and query-set word-count
                trainLikelySum += trainSetLikely
                trainWordsSum  += trainWords
                queryLikelySum += querySetLikely
                queryWordsSum  += queryWords
                finishedFolds  += 1
                
                # Write out the output
                print("Fold %d: Train-set Perplexity: %12.3f \t Query-set Perplexity: %12.3f" % (fold, trainSetPerp, querySetPerp))
                print("")
            finally:
                # Write out the end result of the model run.
                if args.out_model is not None:
                    modelFile = newModelFile(args.model, args.K, args.P, fold, args.out_model)
                    modelFiles.append(modelFile)
                    with open(modelFile, 'wb') as f:
                        pkl.dump ((order, boundItrs, boundVals, boundLikes, modelState, trainTopics, queryTopics), f)
        
        print ("Total (%d): Train-set Likelihood: %12.3f \t Train-set Perplexity: %12.3f" % (finishedFolds, trainLikelySum, np.exp(-trainLikelySum / trainWordsSum)))
        print ("Total (%d): Query-set Likelihood: %12.3f \t Query-set Perplexity: %12.3f" % (finishedFolds, queryLikelySum, np.exp(-queryLikelySum / queryWordsSum)))
        
                        
    return modelFiles

def newModelFileFromModel(model, fold=None, prefix="/Users/bryanfeeney/Desktop"):
    return newModelFile (\
                model.name, \
                model.K, \
                None if model.name[:3] == "ctm" else model.P, \
                fold, \
                prefix)

def splitInput(X, W):
    '''
    For traditional topic models, when evaluating on unseen data, we partition
    each document into two parts, using one to estimate topic memberships, and
    the second part to evaluate the log-likelihood given those memberships.
    
    For topic models with side-information, where we don't use words to determine
    topic-memberships, we can use the entire set of words to evaluate the
    likelihood.
    
    So if there are features (X is not None) then we return W for both estimation
    and evaluation. If there are not we partition each document (i.e. row) in W
    into two, and return both versions.
    
    Params:
    X - the DxF matrix of F features for all D documents, used only to see if a 
        split in W is necessary
    W - the DxT matrix of T term-counts for all D documents

    Returns:
    Two DxT matrices of word-counts derived from W, the former for estimation, the
    latter for evaluation
    '''
    if X is not None:
        return W, W
    
    rng = rd.RandomState(0xBADB055)
    
    dat    = W.data
    jitter = rng.normal(scale=0.3, size=len(dat)).astype(dtype=W.dtype)
    evl    = dat + jitter
    est    = np.around(evl / 2.0)
    evl    = dat - est
    
    return \
        ssp.csr_matrix((est, W.indices, W.indptr), shape=W.shape), \
        ssp.csr_matrix((evl, W.indices, W.indptr), shape=W.shape)

def newModelFile(modelName, K, P, fold=None, prefix="/Users/bryanfeeney/Desktop"):
    modelName = modelName.replace('/','_')
    modelName = modelName.replace('-','_')
    timestamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
    
    cfg = "k_" + str(K)
    if P is not None:
        cfg += "_p_" + str(P)
    
    foldDesc = "" if fold is None else "fold_" + str(fold) + "_"
    
    return prefix + '/' \
         + modelName + '_' \
         + cfg + '_' \
         + foldDesc  \
         + timestamp + '.pkl'

if __name__ == '__main__':
    run(args=sys.argv[1:])
    