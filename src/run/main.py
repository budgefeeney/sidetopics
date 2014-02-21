metavar=' ', '''
Created on 26 Nov 2013

@author: bryanfeeney
'''
import argparse as ap
import pickle as pkl
import numpy as np
import sys
import matplotlib as mpl
import time
import os
mpl.use('Agg') # Force everything to be saved to disk, including VB plots
from math import ceil

DTYPE=np.float32


def run(args):
    #
    # Enumerate all possible arguments
    #
    parser = ap.ArgumentParser(description='Execute a topic-modeling run.')
    parser.add_argument('--model', '-m', dest='model', metavar=' ', \
                    help='The type of mode to use, options are uy, uyv and uv_vec_y')
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
    parser.add_argument('--out-model', '-o', dest='out_model', default=os.getcwd(), metavar=' ', \
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
    parser.add_argument('--lat-feat-var', dest='lat_feat_var', type=float, default=0.1, default=0.1, metavar=' ', \
                    help="Scale of the prior isotropic variance over latent features")
    parser.add_argument('--folds', '-f', dest='folds', type=int, default=1, metavar=' ', \
                    help="Number of cross validation folds.")
    parser.add_argument('--debug', '-b', dest='debug', type=bool, default=False, metavar=' ', \
                    help="Display a debug message, with the bound, after every variable update")
    
    #
    # Parse the arguments
    #
    print ("Args are : " + str(args))
    args = parser.parse_args(args)
    
    
    #
    # Load in the files
    #
    with open(args.words, 'rb') as f:
        W = pkl.load(f)
        W = W.astype(DTYPE)
    with open(args.feats, 'rb') as f:
        X = pkl.load(f)
        X = X.astype(DTYPE)
    (D,F) = X.shape
    (_,T) = W.shape
    K     = args.K
    P     = args.P
    Q     = args.Q
    folds = args.folds
    
    fv, tv, lfv, ltv = args.feat_var, args.topic_var, args.lat_feat_var, args.lat_topic_var
    
    #
    # Instantiate and configure the model
    #
    if args.model == 'ctm':
        import model.ctm as mdl
        templateModel = mdl.newModelAtRandom(W, K, dtype=DTYPE)
    elif args.model == 'ctm_bohning':
        import model.ctm_bohning as mdl
        templateModel = mdl.newModelAtRandom(W, K, dtype=DTYPE)
    elif args.model == 'stm_yv':
        import model.stm_yv as mdl 
        templateModel = mdl.newModelAtRandom(X, W, P, K, fv, lfv, dtype=DTYPE)
    elif args.model == 'stm_yv_bohning':
        import model.stm_yv as mdl 
        templateModel = mdl.newModelAtRandom(X, W, P, K, fv, lfv, dtype=DTYPE)
    else:
        raise ValueError ("Unknown model identifier " + args.model)
    
    trainPlan = mdl.newTrainPlan(args.iters, args.min_vb_change, args.log_freq)
    queryPlan = mdl.newTrainPlan(args.query_iters, args.min_vb_change, args.log_freq)
    
    # things to inspect and store for later use
    boundItrses   = [] # array of array of iterations where var-bound was measured
    boundValses   = [] # ditto for the actual values.
    models        = []
    trainTopicses = []
    queryTopicses = []

    try:
        # Run the model on each fold
        if folds == 1:
            model = mdl.newModelFromExisting(templateModel)
            query = mdl.newQueryState(W, model)
            
            model, query, (boundItrs, boundVals) = mdl.train (W, X, model, query, trainPlan)
            trainSetLikely = mdl.log_likelihood (W, model, query)
            perp = mdl.perplexity(W, model, query)
                        
            print("Train-set Likelihood: %12f" % (trainSetLikely))
            print("Train-set Perplexity: %12f" % (perp))
            print("")
            
            models.append (model)
            trainTopicses.append(query)
            boundItrses.append(boundItrs)
            boundValses.append(boundVals)
            queryTopicses.append(None)
        else:
            foldSize  = ceil(D / folds)
            querySize = foldSize
            trainSize = D - querySize
        
            for fold in range(folds):
                # Split the datasets up for hte current fold
                start = fold * foldSize
                end   = start + trainSize
                
                trainSet = np.arange(start,end) % D
                querySet = np.arange(end, end + querySize) % D
                
                X_train, W_train = X[trainSet,:], W[trainSet,:]
                X_query, W_query = X[querySet,:], W[querySet,:]
                
                # Train the model
                modelState  = mdl.newModelFromExisting(templateModel)
                trainTopics = mdl.newQueryState(W_train, modelState)
                modelState, trainTopics, (boundItrs, boundVals) \
                    = mdl.train(W_train, X_train, modelState, trainTopics, trainPlan)
                    
                trainSetLikely = mdl.log_likelihood (W_train, modelState, trainTopics)
                trainSetPerp   = np.exp(-trainSetLikely / W_train.data.sum())
                        
                # Save the training results
                models.append (model)
                trainTopicses.append(query)
                boundItrses.append(boundItrs)
                boundValses.append(boundVals)
                
                # Query the model
                queryTopics = mdl.newQueryState(W_query, modelState)
                modelState, queryTopics = mdl.query(W_query, X_query, modelState, queryTopics, queryPlan)
                
                querySetLikely = mdl.log_likelihood(modelState, X_query, W_query, queryTopics)
                querySetPerp   = np.exp(-querySetLikely / W_query.data.sum())
                
                # Save the query results
                queryTopicses.append(queryTopics)
                
                # Write out the output
                print("Fold %d: Train-set Perplexity: %12.3f \t Query-set Perplexity: %12.3f" % (fold, trainSetPerp, querySetPerp))
                print("")
    finally:
        # Write out the end result of the model run.
        with open(modelFile(args.model, args.K, args.P, args.out_model)) as f:
            pkl.dump ((boundItrses, boundValses, models, trainTopicses, queryTopicses), f)
        

def modelFileFromModel(model, prefix="/Users/bryanfeeney/Desktop"):
    return modelFile (\
                model.name, \
                model.K, \
                None if model[:3] == "ctm" else model.P, \
                prefix)

def modelFile(modelName, K, P, prefix="/Users/bryanfeeney/Desktop"):
    modelName = modelName.replace('/','_')
    modelName = modelName.replace('-','_')
    timestamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
    
    cfg = "k_" + str(K)
    if P is not None:
        cfg += "_p_" + str(P)
    
    return prefix + '/' \
         + modelName + '_' \
         + cfg + '_' \
         + timestamp + '.pkl'

if __name__ == '__main__':
    run(args=sys.argv[1:])
    