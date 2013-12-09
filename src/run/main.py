metavar=' ', '''
Created on 26 Nov 2013

@author: bryanfeeney
'''
import argparse as ap
import pickle as pkl
import numpy as np
import sys
from math import ceil

from model.sidetopic_uy import \
    train as train_uy, \
    newVbModelState as new_uy
from model.sidetopic_uyv import \
    train as train_uyv, \
    newVbModelState as new_uyv, \
    query, \
    newInferencePlan, \
    log_likelihood, \
    DTYPE, \
    VbSideTopicQueryState, VbSideTopicModelState
from model.sidetopic_uv_vecy import \
    train as train_uv_vecy, \
    newVbModelState as new_uv_vecy

  
def selectModel(args):
    '''
    Returns a tuple of three methods for instantiating a model, 
    training it, and querying it, in that order. All methods take the
    same parameters (see sidetopics_uyv for an example). Expects the
    args to contain a 'model' value specifying the appropriate model, values
    are 'uyv', 'uy' and 'uv_vec_y'
    '''
    
    if args.model == 'uyv':
        return new_uyv, train_uyv, query
    elif args.model == 'uy':
        return new_uy, train_uy, query
    elif args.model == 'uv_vec_y' or args.model == 'uv_vecy':
        return new_uv_vecy, train_uv_vecy, query
    else:
        raise ValueError ('Unknown model type ' + args.model)
    
def newTrainPlans(args):
    '''
    Creates a new inference plan for training from the command-line arguments.
    Generates one plan for every cross-validated fold, as the fold names are
    required to modify the filenames for each of the plots
    '''
    return newPlans (args.iters, 'train', args)
   
def newQueryPlans(args):
    '''
    Creates a new inference plan for querying from the command-line arguments
    '''
    queryIters = args.query_iters if args.query_iters is not None else args.iters
    return newPlans (queryIters, 'query', args)
    
def newPlans(iters, plotSuffix, args):
    '''
    Creates a new inference plan for training or querying from the command-line 
    arguments and given parameters.
    Generates one plan for every cross-validated fold, as the fold names are
    required to modify the filenames for each of the plots
    
    iters - the number of training iterations to run
    plotSuffix - either 'train' or 'query'
    args - the rest of the command-line arguments
    '''
    if args.folds == 1:
        return newInferencePlan(args.iters, args.min_vb_change, args.log_freq, plot=args.out_plot is not None, plotFile=args.out_plot, fastButInaccurate=False)
    else:
        plotPrefix = args.out_plot if args.out_plot is not None else ''
        plans = []
        for fold in range(args.folds):
            plotName = plotPrefix + '-' + plotSuffix + '-' + str(fold)
            plans.append(newInferencePlan(iters, args.min_vb_change, args.log_freq, plot=args.out_plot is not None, plotFile=plotName, fastButInaccurate=False))
        return plans
    
def dumpModel (file, modelState, trainTopics, queryTopics):
    '''
    Uses pickle to store the VbSideTopicModelState object summarising the model,
    and the VbSideTopicQueryState bojects summarising the topic assignments for
    the training set and the query set
    '''
    
    megaTuple = (modelState.K, modelState.Q, modelState.F, modelState.P, modelState.T, \
                 modelState.A, modelState.varA, modelState.Y, modelState.omY, modelState.sigY,\
                 modelState.sigT, modelState.U, modelState.V, modelState.vocab, \
                 modelState.topicVar, modelState.featVar, modelState.lowTopicVar, \
                 modelState.lowFeatVar, trainTopics.expLmda, trainTopics.nu, trainTopics.lxi,\
                 trainTopics.s, trainTopics.docLen, queryTopics.expLmda, queryTopics.nu, \
                 queryTopics.lxi, queryTopics.s, queryTopics.docLen)
    
    pkl.dump(megaTuple, file)

def loadModel (file):
    '''
    Loads in the model state and train and query topic assignments created
    by dumpModel()
    '''
    (modelStateK, modelStateQ, modelStateF, modelStateP, modelStateT, \
     modelStateA, modelStatevarA, modelStateY, modelStateomY, modelStatesigY,\
     modelStatesigT, modelStateU, modelStateV, modelStatevocab, \
     modelStatetopicVar, modelStatefeatVar, modelStatelowTopicVar, \
     modelStatelowFeatVar, trainTopicsexpLmda, trainTopicsnu, trainTopicslxi,\
     trainTopicss, trainTopicsdocLen, queryTopicsexpLmda, queryTopicsnu, \
     queryTopicslxi, queryTopicss, queryTopicsdocLen) = pkl.load(file)
     
    return \
         VbSideTopicModelState (modelStateK, modelStateQ, modelStateF, \
              modelStateP, modelStateT, modelStateA, modelStatevarA, \
              modelStateY, modelStateomY, modelStatesigY, modelStatesigT,\
              modelStateU, modelStateV, modelStatevocab, modelStatetopicVar,\
              modelStatefeatVar, modelStatelowTopicVar, modelStatelowFeatVar), \
         VbSideTopicQueryState (trainTopicsexpLmda, trainTopicsnu, \
                 trainTopicslxi, trainTopicss, trainTopicsdocLen), \
         VbSideTopicQueryState (queryTopicsexpLmda, queryTopicsnu, \
                 queryTopicslxi, queryTopicss, queryTopicsdocLen)


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
    parser.add_argument('--eval', '-v', dest='eval', metavar=' ', \
                    help='Evaluation metric, only available is: perplexity or likelihood')
    parser.add_argument('--out-model', '-o', dest='out_model', metavar=' ', \
                    help='Optional output path in which to store the model')
    parser.add_argument('--out-plot', '-t', dest='out_plot', metavar=' ', \
                    help='Optional output path in which to store the plot of variational bound')
    parser.add_argument('--log-freq', '-l', dest='log_freq', type=int, metavar=' ', \
                    help='Log frequency - how many times to inspect the bound while running')
    parser.add_argument('--iters', '-i', dest='iters', type=int, metavar=' ', \
                    help='The maximum number of iterations to run when training')
    parser.add_argument('--query-iters', '-j', dest='query_iters', type=int, metavar=' ', \
                    help='The maximum number of iterations to run when querying, by default same as when training')
    parser.add_argument('--min-vb-change', '-e', dest='min_vb_change', type=float, metavar=' ', \
                    help='The amount by which the variational bound must change at each log-interval to avoid inference being stopped early.')
    parser.add_argument('--topic-var', dest='topic_var', type=float, metavar=' ', \
                    help="Scale of the prior isotropic variance over topics")
    parser.add_argument('--feat-var', dest='feat_var', type=float, metavar=' ', \
                    help="Scale of the prior isotropic variance over features")
    parser.add_argument('--lat-topic-var', dest='lat_topic_var', type=float, metavar=' ', \
                    help="Scale of the prior isotropic variance over latent topics")
    parser.add_argument('--lat-feat-var', dest='lat_feat_var', type=float, metavar=' ', \
                    help="Scale of the prior isotropic variance over latent features")
    parser.add_argument('--folds', '-f', dest='folds', type=int, default=1, metavar=' ', \
                    help="Number of cross validation folds.")
    
    #
    # Parse the arguments
    #
    print ("Args are : " + str(args))
    args = parser.parse_args(args)
    
    #
    # Instantiate and execute the model
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
    
    newModel, trainModel, queryModel = selectModel(args)
    trainPlans = newTrainPlans(args)
    queryPlans = newQueryPlans(args)

        
    
    if folds == 1:
        modelState = newModel(K, Q, F, P, fv, tv, lfv, ltv)
        modelState, queryState = trainModel(modelState, X, W, trainPlans[0])
        trainSetLikely = log_likelihood(modelState, X, W, queryState)
                    
        print("Train-set Likelihood: %12f" % (trainSetLikely))
        print("")
    else:
        foldSize  = ceil(D / folds)
        querySize = foldSize
        trainSize = D - querySize
    
        for fold in range(folds):
            start = fold * foldSize
            end   = start + trainSize
            
            trainSet = np.arange(start,end) % D
            querySet = np.arange(end, end + querySize) % D
            
            X_train, W_train = X[trainSet,:], W[trainSet,:]
            X_query, W_query = X[querySet,:], W[querySet,:]
            
            modelState = newModel(K, Q, F, P, T, fv, tv, lfv, ltv)
            modelState, trainTopics = trainModel(modelState, X_train, W_train, trainPlans[fold])
            trainSetLikely = log_likelihood(modelState, X_train, W_train, trainTopics)
            trainSetPerp   = np.exp (-trainSetLikely / np.sum(trainTopics.docLen))
            
            queryTopics    = queryModel(modelState, X_query, W_query, queryPlans[fold])
            querySetLikely = log_likelihood(modelState, X_query, W_query, queryTopics)
            querySetPerp   = np.exp (-querySetLikely / np.sum(queryTopics.docLen))
            
            if args.out_model is not None:
                with open(args.out_model + "-" + str(fold) + ".pkl", 'wb') as f:
                    dumpModel (f, modelState, trainTopics, queryTopics)
            
            print("Fold %d: Train-set Perplexity: %12.3f \t Query-set Perplexity: %12.3f" % (fold, trainSetPerp, querySetPerp))
            print("")

    
    

if __name__ == '__main__':
    run(args=sys.argv)
    