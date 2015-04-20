'''
Created on 26 Nov 2013

@author: bryanfeeney
'''
import argparse as ap
import pickle as pkl
import numpy as np
import numpy.random as rd
import sys
import time

from model.common import DataSet
from model.evals import perplexity_from_like

DTYPE=np.float32

CtmBouchard   = 'ctm_bouchard'
CtmBohning    = 'ctm_bohning'
StmYvBouchard = 'stm_yv_bouchard'
StmYvBohning  = 'stm_yv_bohning'
LdaCvbZero    = 'lda_cvb0'
LdaVb         = 'lda_vb'
Rtm           = "rtm_vb"

ModelNames = ', '.join([CtmBouchard, CtmBohning, StmYvBouchard, StmYvBohning, LdaCvbZero, LdaVb, Rtm])

FastButInaccurate=False

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
    parser.add_argument('--links', '-c', dest='feats', metavar=' ', \
                    help='The path to the pickle file containing a DxP array or matrix of the links (citations) emanated by all D documents')
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
    print("Args are : " + str(args))
    args = parser.parse_args(args)

    print("Random seed is 0xC0FFEE")
    rd.seed(0xC0FFEE)

    K, P, Q = args.K, args.P, args.Q
    dtype   = np.float32 if args.dtype == 'f4' else np.float64

    data = DataSet(args.words, args.feats, args.links)
    order = data.prune_and_shuffle(min_doc_len=0.5)
    folds = args.folds

    fv, tv, lfv, ltv = args.feat_var, args.topic_var, args.lat_feat_var, args.lat_topic_var

    #
    # Instantiate and configure the model
    #
    if args.model == CtmBouchard:
        import model.ctm as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=dtype)
    elif args.model == CtmBohning:
        import model.ctm_bohning as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=dtype)
    elif args.model == StmYvBouchard:
        import model.stm_yv as mdl
        templateModel = mdl.newModelAtRandom(data, P, K, fv, lfv, dtype=dtype)
    elif args.model == StmYvBohning:
        import model.stm_yv_bohning as mdl
        templateModel = mdl.newModelAtRandom(data, P, K, fv, lfv, dtype=dtype)
    elif args.model == LdaCvbZero:
        import model.lda_cvb as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=dtype)
    elif args.model == LdaVb:
        import model.lda_vb as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=dtype)
    elif args.model == Rtm:
        import model.rtm as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=dtype)
    else:
        raise ValueError ("Unknown model identifier " + args.model)

    trainPlan = mdl.newTrainPlan(args.iters, args.min_vb_change, args.log_freq, fastButInaccurate=FastButInaccurate, debug=args.debug)
    queryPlan = mdl.newTrainPlan(args.query_iters, args.min_vb_change, args.log_freq, debug=args.debug)

    # things to inspect and store for later use
    modelFiles = []

    # Run the model on each fold
    if folds == 1:
        try:
            model = mdl.newModelFromExisting(templateModel)
            query = mdl.newQueryState(data, model)

            model, query, (boundItrs, boundVals, boundLikes) = mdl.train(data, model, query, trainPlan)
            trainSetLikely = mdl.log_likelihood(data, model, query) # TODO Fix me by tupling
            perp = perplexity_from_like(trainSetLikely, data.word_count)

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


        for fold in range(folds):
            try:
                train_data, query_data = data.cross_valid_split(fold, folds)

                # Train the model
                modelState  = mdl.newModelFromExisting(templateModel)
                trainTopics = mdl.newQueryState(train_data, modelState)
                modelState, trainTopics, (boundItrs, boundVals, boundLikes) \
                    = mdl.train(train_data, modelState, trainTopics, trainPlan)

                trainSetLikely = mdl.log_likelihood (train_data, modelState, trainTopics)
                trainWordCount = train_data.word_count
                trainSetPerp   = perplexity_from_like(trainSetLikely, trainWordCount)

                # Query the model - if there are no features we need to split the text
                query_estim, query_eval = query_data.doc_completion_split()
                queryTopics = mdl.newQueryState(query_estim, modelState)
                modelState, queryTopics = mdl.query(query_estim, modelState, queryTopics, queryPlan)

                querySetLikely = mdl.log_likelihood(query_eval, modelState, queryTopics)
                queryWordCount = query_eval.word_count
                querySetPerp   = perplexity_from_like(querySetLikely, queryWordCount)

                # Keep a record of the cumulative likelihood and query-set word-count
                trainLikelySum += trainSetLikely
                trainWordsSum  += trainWordCount
                queryLikelySum += querySetLikely
                queryWordsSum  += queryWordCount
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

        print ("Total (%d): Train-set Likelihood: %12.3f \t Train-set Perplexity: %12.3f" % (finishedFolds, trainLikelySum, perplexity_from_like(trainLikelySum, trainWordsSum)))
        print ("Total (%d): Query-set Likelihood: %12.3f \t Query-set Perplexity: %12.3f" % (finishedFolds, queryLikelySum, perplexity_from_like(queryLikelySum, queryWordsSum)))


    return modelFiles

def newModelFileFromModel(model, fold=None, prefix="/Users/bryanfeeney/Desktop"):
    return newModelFile (\
                model.name, \
                model.K, \
                None if model.name[:3] == "ctm" else model.P, \
                fold, \
                prefix)



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
