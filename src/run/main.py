'''
Created on 26 Nov 2013

@author: bryanfeeney
'''
import argparse as ap
import pickle as pkl
import numpy as np
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
    DTYPE
from model.sidetopic_uv_vecy import \
    train as train_uv_vecy, \
    newVbModelState as new_uv_vecy

  
def selectModel(args):
    '''
    Returns a tuple of three methods for instantiating a model, 
    trianing it, and querying it, in that order. All methods take the
    same parameters (see sidetopics_uyv for an example). Expects the
    args to contain a 'model' value specifying the appropriate model, values
    are 'uyv', 'uy' and 'uv_vec_y'
    '''
    
    if args['model'] == 'uyv':
        return new_uyv, train_uyv, query
    elif args['model'] == 'uy':
        return new_uy, train_uy, query
    elif args['model'] == 'uv_vec_y' or args['model'] == 'uv_vecy':
        return new_uv_vecy, train_uv_vecy, query
    else:
        raise ValueError ('Unknown model type ' + args['model'])
    
def newTrainPlan(args):
    '''
    Creates a new inference plan for training from the command-line arguments
    '''
    return newInferencePlan(args['iters'], args['min_vb_change'], args['log_freq'], plot=args['out_plot'] is not None, args['out_plot'], fastButInaccurate=False)

   
def newQueryPlan(args):
    '''
    Creates a new inference plan for querying from the command-line arguments
    '''
    queryIters = args['query_iters'] if args['query_iters'] is not None else args['iters']
    return newInferencePlan(queryIters, args['min_vb_change'], args['log_freq'], plot=args['out_plot'] is not None, args['out_plot'], fastButInaccurate=False)
    
    

if __name__ == '__main__':
    #
    # Enumerate all possible arguments
    #
    parser = ap.ArgumentParser(description='Execute a topic-modeling run.')
    parser.add_argument('--model', dest='model', \
                    help='The type of mode to use, options are uy, uyv and uv_vec_y')
    parser.add_argument('--num-topics', dest='K', type=int, \
                    help='The number of topics to fit')
    parser.add_argument('--num-lat-topics', dest='Q', type=int, \
                    help='The number of latent topics (i.e. rank of the topic covariance matrix)')
    parser.add_argument('--num-lat-feats', dest='P', type=int, \
                    help='The number of latent features (i.e. rank of the features covariance matrix)')
    parser.add_argument('--words', dest='words', \
                    help='The path to the pickle file containing a DxT array or matrix of the word-counts across all D documents')
    parser.add_argument('--feats', dest='feats', \
                    help='The path to the pickle file containing a DxF array or matrix of the features across all D documents')
    parser.add_argument('--eval', dest='eval', \
                    help='Evaluation metric, only available is: perplexity or likelihood')
    parser.add_argument('--out-model', dest='out_model', \
                    help='Optional output path in which to store the model')
    parser.add_argument('--out-plot', dest='out_plot', \
                    help='Optional output path in which to store the plot of variational bound')
    parser.add_argument('--log-freq', dest='log_freq', type=int, \
                    help='Log frequency - how many times to inspect the bound while running')
    parser.add_argument('--iters', dest='iters', type=int, \
                    help='The maximum number of iterations to run when training')
    parser.add_argument('--query-iters', dest='query_iters', type=int, \
                    help='The maximum number of iterations to run when querying, by default same as when training')
    parser.add_argument('--min-vb-change', 'min_vb_change', type=float,\
                    help='The amount by which the variational bound must change at each log-interval to avoid inference being stopped early.')
    parser.add_argument('--topic-var', dest='topic_var', type=float, \
                    help="Scale of the prior isotropic variance over topics")
    parser.add_argument('--feat-var', dest='feat_var', type=float, \
                    help="Scale of the prior isotropic variance over features")
    parser.add_argument('--lat-topic-var', dest='lat_topic_var', type=float, \
                    help="Scale of the prior isotropic variance over latent topics")
    parser.add_argument('--lat-feat-var', dest='lat_feat_var', type=float, \
                    help="Scale of the prior isotropic variance over latent features")
    
    #
    # Parse the arguments
    #
    args = parser.parse_args()
    
    #
    # Instantiate and execute the model
    #
    with open(args['words'], 'rb') as f:
        W = pkl.load(f).astype(DTYPE)
    with open(args['side'], 'rb') as f:
        X = pkl.load(f).astype(DTYPE)
    (D,F) = X.shape
    (_,T) = W.shape
    K     = args['K']
    P     = args['P']
    Q     = args['Q']
    folds = 5
    
    name = args['model']
    fv, tv, lfv, ltv = args['feat_var'], args['topic_var'], args['lat_feat_var'], args['lat_topic_var']
    
    newModel, trainModel, queryModel, DTYPE = selectModel(args)
    trainPlan = newTrainPlan(args)
    queryPlan = newQueryPlan(args)
    
    foldSize  = ceil(D / 5)
    querySize = foldSize
    trainSize = D - querySize
    
    for fold in range(folds):
        start = fold * foldSize
        end   = start + trainSize
        
        trainSet = np.arange(start,end) % D
        querySet = np.arange(end, end + querySize) % D
        
        X_train, W_train = X[trainSet,:], W[trainSet,:]
        X_query, W_query = X[querySet,:], W[querySet,:]
        
        modelState = newModel(K, Q, F, P, fv, tv, lfv, ltv)
        modelState, queryState = trainModel(modelState, X_train, W_train, iterations=100, logInterval=10, plotInterval=100)
        trainSetLikely = log_likelihood(modelState, X_train, W_train, queryState)
        
        queryState = queryModel(modelState, X_query, W_query, iterations=50, epsilon=0.001, logInterval = 10, plotInterval = 100)
        querySetLikely = log_likelihood(modelState, X_query, W_query, queryState)
        
        print("Fold %d: Train-set Likelihood: %12f \t Query-set Likelihood: %12f" % (fold, trainSetLikely, querySetLikely))
        print("")
        
    print("End of Test")
    
    