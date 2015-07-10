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
from model.evals import perplexity_from_like, mean_average_prec, mean_prec_rec_at, \
    EvalNames, Perplexity, MeanAveragePrecAllDocs, MeanPrecRecAtMAllDocs

DTYPE=np.float32

CtmBouchard   = 'ctm_bouchard'
CtmBohning    = 'ctm_bohning'
StmYvBouchard = 'stm_yv_bouchard'
StmYvBohning  = 'stm_yv_bohning'
LdaCvbZero    = 'lda_cvb0'
LdaVb         = 'lda_vb'
LdaGibbs      = 'lda_gibbs'
Rtm           = "rtm_vb"
Mtm           = "mtm_vb"

ModelNames = ', '.join([CtmBouchard, CtmBohning, StmYvBouchard, StmYvBohning, LdaCvbZero, LdaVb, LdaGibbs, Rtm,Mtm])


DefaultPriorCov = 0.001
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
    parser.add_argument('--links', '-c', dest='links', metavar=' ', \
                    help='The path to the pickle file containing a DxP array or matrix of the links (citations) emanated by all D documents')
    parser.add_argument('--eval', '-v', dest='eval', default=Perplexity, metavar=' ', \
                    help='Evaluation metric, available options are: ' + ','.join(EvalNames))
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
    parser.add_argument('--topic-var', dest='topic_var', type=float, default=DefaultPriorCov, metavar=' ', \
                    help="Scale of the prior isotropic variance over topics")
    parser.add_argument('--feat-var', dest='feat_var', type=float, default=DefaultPriorCov, metavar=' ', \
                    help="Scale of the prior isotropic variance over features")
    parser.add_argument('--lat-topic-var', dest='lat_topic_var', type=float, default=DefaultPriorCov, metavar=' ', \
                    help="Scale of the prior isotropic variance over latent topics")
    parser.add_argument('--lat-feat-var', dest='lat_feat_var', type=float, default=DefaultPriorCov, metavar=' ', \
                    help="Scale of the prior isotropic variance over latent features")
    parser.add_argument('--folds', '-f', dest='folds', type=int, default=1, metavar=' ', \
                    help="Number of cross validation folds.")
    parser.add_argument('--truncate-folds', dest='eval_fold_count', type=int, default=-1, metavar=' ', \
                    help="If set, stop running after the given number of folds had been processed")
    parser.add_argument('--debug', '-b', dest='debug', type=bool, default=False, metavar=' ', \
                    help="Display a debug message, with the bound, after every variable update")
    parser.add_argument('--dtype', '-t', dest='dtype', default="f4:f4", metavar=' ', \
                    help="Datatype to use, values are i4, f4 and f8. Specify two, a data dtype and model dtype, delimited by a colon")
    parser.add_argument('--limit-to', dest='limit', type=int, default=0, metavar=' ', \
                    help="If set, discard all but the initial given number of rows of the input dataset")


    #
    # Parse the arguments
    #
    print("Args are : " + str(args))
    args = parser.parse_args(args)

    print("Random seed is 0xC0FFEE")
    rd.seed(0xC0FFEE)

    K, P, Q = args.K, args.P, args.Q
    (input_dtype, output_dtype)  = parse_dtypes(args.dtype)

    data = DataSet.from_files(args.words, args.feats, args.links, limit=args.limit)
    data.convert_to_dtype(input_dtype)
    data.prune_and_shuffle(min_doc_len=3, min_link_count=2)
    print ("The combined word-count of the %d documents is %.0f, drawn from a vocabulary of %d distinct terms" % (data.doc_count, data.word_count, data.words.shape[1]))
    if data.add_intercept_to_feats_if_required():
        print ("Appended an intercept to the given features")

    fv, tv, lfv, ltv = args.feat_var, args.topic_var, args.lat_feat_var, args.lat_topic_var

    #
    # Instantiate and configure the model
    #
    print ("Building template model... ", end="")
    if args.model == CtmBouchard:
        import model.ctm as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == CtmBohning:
        import model.ctm_bohning as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == StmYvBouchard:
        import model.stm_yv as mdl
        templateModel = mdl.newModelAtRandom(data, P, K, fv, lfv, dtype=output_dtype)
    elif args.model == StmYvBohning:
        import model.stm_yv_bohning as mdl
        templateModel = mdl.newModelAtRandom(data, P, K, fv, lfv, dtype=output_dtype)
    elif args.model == LdaCvbZero:
        import model.lda_cvb as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == LdaVb:
        import model.lda_vb_python as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == LdaGibbs:
        import model.lda_gibbs as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == Rtm:
        import model.rtm as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == Mtm:
        import model.mtm2 as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    else:
        raise ValueError ("Unknown model identifier " + args.model)
    print("Done")


    trainPlan = mdl.newTrainPlan(args.iters, debug=args.debug)
    queryPlan = mdl.newTrainPlan(args.query_iters, debug=args.debug)

    if args.eval == Perplexity:
        return cross_val_and_eval_perplexity(data, mdl, templateModel, trainPlan, queryPlan, args.folds, args.eval_fold_count, args.out_model)
    elif args.eval == MeanAveragePrecAllDocs:
        return link_split_map (data, mdl, templateModel, trainPlan, args.folds, args.out_model)
    elif args.eval == MeanPrecRecAtMAllDocs:
        return link_split_prec_rec (data, mdl, templateModel, trainPlan, args.folds, args.out_model)
    else:
        raise ValueError("Unknown evaluation metric " + args.eval)

    return modelFiles


def parse_dtypes(dtype_str):
    '''
    Parse one or two dtype strings, delimited by a colon if there are two
    '''
    strs = [s.strip() for s in dtype_str.split(':')]

    return (parse_dtype(strs[0]), parse_dtype(strs[0])) \
        if len(strs) == 1 \
        else (parse_dtype(strs[0]), parse_dtype(strs[1]))

def parse_dtype(dtype_str):
    '''
    Parses a dtype string. Accepted values are f4 or f32, f8 or f64
    or i4 or i32. Case is not sensitive. Two may be optionally
    provided, in which case the first is the data dtype and the
    second is the model-dtype
    '''
    if   dtype_str in ['f8', 'F8', 'f64', 'F64']:
        return np.float64
    elif dtype_str in ['f4', 'F4', 'f32', 'F32']:
        return np.float32
    elif dtype_str in ['i4', 'I4', 'i32', 'I32']:
        return np.int32
    else:
        raise ValueError("Can't parse dtype " + dtype_str)


def cross_val_and_eval_perplexity(data, mdl, sample_model, train_plan, query_plan, num_folds, fold_run_count=-1, model_dir= None):
    '''
    Uses cross-validation go get the average perplexity. If folds == 1 a special path is
    triggered where perplexity is evaluated on the training data, and the results are
    not saved to disk, even if model_dir is not none

    :param data: the DataSet object with the data
    :param mdl:  the module with the train etc. functin
    :param sample_model: a preconfigured model which is cloned at the start of each
            cross-validation run
    :param train_plan:  the training plan (number of iterations etc.)
    :param query_plan:  the query play (number of iterations etc.)
    :param num_folds:  the number of folds to cross validation
    :param fold_run_count: for debugging stop early after processing the number
    of the folds
    :param model_dir: if not none, and folds > 1, the models are stored in this
    directory.
    :return: the list of model files stored
    '''
    model_files = []

    if num_folds == 1:
        model = mdl.newModelFromExisting(sample_model)
        query = mdl.newQueryState(data, model)

        model, train_tops, (train_itrs, train_vbs, train_likes) = mdl.train(data, model, query, train_plan)
        likely = mdl.log_likelihood(data, model, train_tops)
        perp   = perplexity_from_like(likely, data.word_count)

        print("Train-set Likelihood: %12f" % (likely))
        print("Train-set Perplexity: %12f" % (perp))

        model_files = save_if_necessary(model_files, model_dir, model, data, 0, train_itrs, train_vbs, train_likes, train_tops, train_tops)
        return model_files

    query_like_sum    = 0 # to calculate the overall likelihood and
    query_wcount_sum  = 0 # perplexity for the whole dataset
    train_like_sum    = 0
    train_wcount_sum  = 0
    folds_finished    = 0 # count of folds that finished successfully

    for fold in range(fold_run_count):
        try:
            train_data, query_data = data.cross_valid_split(fold, num_folds)

            # Train the model
            print ("Duplicating model template... ", end="")
            model      = mdl.newModelFromExisting(sample_model)
            print ("Done.\nCreating query state...")
            train_tops = mdl.newQueryState(train_data, model)

            print ("Starting training")
            model, train_tops, (train_itrs, train_vbs, train_likes) \
                = mdl.train(train_data, model, train_tops, train_plan)

            train_like       = mdl.log_likelihood (train_data, model, train_tops)
            train_word_count = train_data.word_count
            train_perp       = perplexity_from_like(train_like, train_word_count)

            # Query the model - if there are no features we need to split the text
            print ("Starting query.")
            query_estim, query_eval = query_data.doc_completion_split()
            query_tops              = mdl.newQueryState(query_estim, model)
            model, query_tops = mdl.query(query_estim, model, query_tops, query_plan)

            query_like       = mdl.log_likelihood(query_eval, model, query_tops)
            query_word_count = query_eval.word_count
            query_perp       = perplexity_from_like(query_like, query_word_count)

            # Keep a record of the cumulative likelihood and query-set word-count
            train_like_sum += train_like
            train_wcount_sum  += train_word_count
            query_like_sum += query_like
            query_wcount_sum  += query_word_count
            folds_finished  += 1

            # Write out the output
            print("Fold %d: Train-set Perplexity: %12.3f \t Query-set Perplexity: %12.3f" % (fold, train_perp, query_perp))
            print("")

            # Save the model
            model_files = save_if_necessary(model_files, model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, query_tops)
        except Exception as e:
            print("Abandoning fold %d due to the error : %s" % (fold, str(e)))

    print ("Total (%d): Train-set Likelihood: %12.3f \t Train-set Perplexity: %12.3f" % (folds_finished, train_like_sum, perplexity_from_like(train_like_sum, train_wcount_sum)))
    print ("Total (%d): Query-set Likelihood: %12.3f \t Query-set Perplexity: %12.3f" % (folds_finished, query_like_sum, perplexity_from_like(query_like_sum, query_wcount_sum)))

    return model_files


def save_if_necessary (model_files, model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, query_tops):
    if model_dir is not None:
        model_files.append( \
            save_model(\
                model_dir, model, data, \
                fold, train_itrs, train_vbs, train_likes, \
                train_tops, query_tops))
    return model_files


def save_model(model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, query_tops):
    P = 0 if 'P' not in dir(model) else model.P

    model_file = newModelFile(model.name, model.K, P, fold, model_dir)
    with open (model_file, 'wb') as f:
        pkl.dump ((data.order, train_itrs, train_vbs, train_likes, model, train_tops, query_tops), f)

    return model_file


def link_split_map (data, mdl, sample_model, train_plan, folds, model_dir = None):
    '''
    Train on all the words and half the links. Predict the remaining links.
    Evaluate using mean average-precision.

    Cross validation may be used, but note we're always evaluating on training
    data.

    :param data: the DataSet object with the data
    :param mdl:  the module with the train etc. functin
    :param sample_model: a preconfigured model which is cloned at the start of each
            cross-validation run
    :param train_plan:  the training plan (number of iterations etc.)
    :param folds:  the number of folds to cross validation
    :param model_dir: if not none, and folds > 1, the models are stored in this
    directory.
    :return: the list of model files stored
    '''
    model_files = []
    assert folds > 1, "Need at least two folds for this to make any sense whatsoever"
    def prepareForTraining(data):
        if mdl.is_undirected_link_predictor():
            result = data.copy()
            result.convert_to_undirected_graph()
            result.convert_to_binary_link_matrix()
            return result
        else:
            return data


    for fold in range(folds):
        model = mdl.newModelFromExisting(sample_model)
        train_data, query_data = data.link_prediction_split(symmetric=False)
        train_data = prepareForTraining(train_data) # make symmetric, if necessary, after split, so we
                                                    # can compare symmetric with non-symmetric models
        train_tops = mdl.newQueryState(train_data, model)
        model, train_tops, (train_itrs, train_vbs, train_likes) = \
            mdl.train(train_data, model, train_tops, train_plan)

        print ("Training perplexity is %.2f " % perplexity_from_like(mdl.log_likelihood(train_data, model, train_tops), train_data.word_count))

        min_link_probs       = mdl.min_link_probs(model, train_tops, query_data.links)
        predicted_link_probs = mdl.link_probs(model, train_tops, min_link_probs)

        map = mean_average_prec (query_data.links, predicted_link_probs)
        print ("Fold %2d: Mean-Average-Precision %6.3f" % (fold, map))

        model_files = save_if_necessary(model_files, model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, train_tops)

    return model_files


def link_split_prec_rec (data, mdl, sample_model, train_plan, folds, model_dir = None):
    '''
    Train on all the words and half the links. Predict the remaining links.
    Evaluate using precision at m using as values of m 50, 100, 250, and 500,
    and additionally recall at m

    Cross validation may be used, but note we're always evaluating on training
    data.

    :param data: the DataSet object with the data
    :param mdl:  the module with the train etc. functin
    :param sample_model: a preconfigured model which is cloned at the start of each
            cross-validation run
    :param train_plan:  the training plan (number of iterations etc.)
    :param folds:  the number of folds to cross validation
    :param model_dir: if not none, and folds > 1, the models are stored in this
    directory.
    :return: the list of model files stored
    '''
    ms = [10, 25, 50, 100, 250, 500]
    model_files = []
    assert folds > 1, "Need at least two folds for this to make any sense whatsoever"
    def prepareForTraining(data):
        if mdl.is_undirected_link_predictor():
            result = data.copy()
            result.convert_to_undirected_graph()
            result.convert_to_binary_link_matrix()
            return result
        else:
            return data


    for fold in range(folds):
        model = mdl.newModelFromExisting(sample_model)
        train_data, query_data = data.link_prediction_split(symmetric=False)
        train_data = prepareForTraining(train_data) # make symmetric, if necessary, after split, so we
                                                    # can compare symmetric with non-symmetric models
        train_tops = mdl.newQueryState(train_data, model)
        model, train_tops, (train_itrs, train_vbs, train_likes) = \
            mdl.train(train_data, model, train_tops, train_plan)

        print ("Training perplexity is %.2f " % perplexity_from_like(mdl.log_likelihood(train_data, model, train_tops), train_data.word_count))

        min_link_probs       = mdl.min_link_probs(model, train_tops, query_data.links)
        predicted_link_probs = mdl.link_probs(model, train_tops, min_link_probs)

        precs, recs, doc_counts = mean_prec_rec_at (query_data.links, predicted_link_probs, at=ms, groups=[(0,3), (3,5), (5,10), (10,1000)])
        print ("Fold %2d: Mean-Precisions at \n" % fold, end="")

        printTable("Precision", precs, doc_counts, ms)
        printTable("Recall",    recs, doc_counts, ms)


        model_files = save_if_necessary(model_files, model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, train_tops)

    return model_files


def printTable(title, scores, doc_counts, ms):
    print(title)
    print("| Group    | Doc Count | " + " | ".join("%5d" % m for m in ms)    + " |")
    print("|----------|-----------|-" + "-|-".join("-----" for _ in len(ms)) + "-|")

    groups = [g for g in scores.keys()]
    groups.sort()

    for g in groups:
        print("| %2d,%5d " % (g[0], g[1]), end="")
        print("| %9d " % doc_counts[g], end="")
        print("| " + " | ".join("%.3f" % m for m in scores[g]), end="")
        print(" |")



def newModelFileFromModel(model, fold=None, prefix="/Users/bryanfeeney/Desktop"):
    return newModelFile (
        model.name,
        model.K,
        None if model.name[:3] == "ctm" else model.P,
        fold,
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
