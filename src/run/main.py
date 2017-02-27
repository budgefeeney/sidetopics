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
import traceback

from model.common import DataSet
from model.evals import perplexity_from_like, mean_average_prec, \
    mean_reciprocal_rank, mean_prec_rec_at, \
    EvalNames, Perplexity, MeanAveragePrecAllDocs,  \
    MeanPrecRecAtMAllDocs, LroMeanPrecRecAtMAllDocs, \
    LroMeanPrecRecAtMFeatSplit, HashtagPrecAtM
from util.sigmoid_utils import rowwise_softmax

DTYPE=np.float32

CtmBouchard   = 'ctm_bouchard'
CtmBohning    = 'ctm_bohning'
StmYvBouchard = 'stm_yv_bouchard'
StmYvBohning  = 'stm_yv_bohning'
LdaCvb        = 'lda_cvb'
LdaCvbZero    = 'lda_cvb0'
LdaVb         = 'lda_vb'
LdaSvb        = 'lda_svb'
LdaGibbs      = 'lda_gibbs'
Rtm           = "rtm_vb"
Mtm           = "mtm_vb"
Mtm2          = "mtm2_vb"
Lro           = "lro_vb"
Dmr           = "dmr"
SimLda        = "sim_lda_vb"
SimTfIdf      = "sim_tfidf"
MomEm         = "mom_em"
MomGibbs      = "mom_gibbs"


StmYvBohningFakeOnline = "stm_yv_bohning_fake_online"

ModelNames = ', '.join([CtmBouchard, CtmBohning, StmYvBouchard, StmYvBohning, StmYvBohningFakeOnline, LdaCvb, LdaCvbZero, LdaVb, LdaSvb, LdaGibbs, Rtm, Mtm, Lro, Dmr, MomEm, MomGibbs])

from model.lda_vb_python import pruneQueryState as pruneLdaVbQueryState
from model.lda_vb_python import MODEL_NAME as LDA_VB_MODEL_NAME
from model.lda_gibbs import pruneQueryState as pruneLdaGibbsQueryState
from model.lda_gibbs import MODEL_NAME as LDA_GIBBS_MODEL_NAME

from model.lro_vb import MODEL_NAME as LRO_MODEL_NAME
from model.sim_based_rec import MODEL_NAME_PREFIX as SIM_MODEL_NAME_PREFIX
from model.sim_based_rec import LDA as SIM_MODEL_NAME_SUFFIX
LDA_SIM_MODEL_NAME = SIM_MODEL_NAME_PREFIX + SIM_MODEL_NAME_SUFFIX

DefaultPriorCov = 0.001
FastButInaccurate=False

MinLinkCountPrune=0 # 2
MinLinkCountEval=5

def model_supports_sgd(model_name):
    return model_name == LdaSvb


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
    parser.add_argument('--vocab-prior', dest='vocabPrior', type=float, default=1.1, metavar=' ', \
                    help="Symmetric prior over the vocabulary")
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
    parser.add_argument('--word-dict', dest='word_dict', default=None, metavar=' ', \
                    help='A dictionary of all words. Used to identify hashtag indices')
    parser.add_argument('--lda-model', dest='ldaModel', default=None, metavar=' ', \
                    help='A trained LDA model, used with the LRO model')
    parser.add_argument('--feats-mask', dest='features_mask_str', default=None, metavar=' ', \
                    help='Feature mask to use with FeatSplit runs, comma-delimited list of colon-delimited pairs')
    parser.add_argument('--gradient-batch-size', dest='sgd_batch_size', type=int, default=0, metavar=' ', \
                    help='What batch size should be employed when training using gradient descent')
    parser.add_argument('--gradient-rate-retardation', dest='sgd_retardation_rate', type=float, default=0.6, metavar=' ', \
                    help='A non-negative number, the higher this value, the smaller the learning rate is in early iterations')
    parser.add_argument('--gradient-forgetting-rate', dest='sgd_forget_rate', type=float, default=0.6, metavar=' ', \
                    help='A number in the range 0.5 < f <= 1, the higher this value, the faster the learning rate collapses to almost zero.')

    # Initialization of the app: first parse the arguments
    #
    print("Random seed is 0xC0FFEE")
    rd.seed(0xC0FFEE)

    print("Args are : " + str(args))
    args = parser.parse_args(args)
    K, P, Q = args.K, args.P, args.Q

    features_mask = parse_features_mask(args)
    (input_dtype, output_dtype)  = parse_dtypes(args.dtype)

    fv, tv, lfv, ltv = args.feat_var, args.topic_var, args.lat_feat_var, args.lat_topic_var

    #
    #  Load and prune the data
    #
    data = DataSet.from_files(args.words, args.feats, args.links, limit=args.limit)
    data.convert_to_dtype(input_dtype)
    data.prune_and_shuffle(min_doc_len=3, min_link_count=MinLinkCountPrune)

    print ("The combined word-count of the %d documents is %.0f, drawn from a vocabulary of %d distinct terms" % (data.doc_count, data.word_count, data.words.shape[1]))
    if data.add_intercept_to_feats_if_required():
        print ("Appended an intercept to the given features")


    #
    # Instantiate and configure the model
    #
    if (args.ldaModel is not None) and (args.model == Lro or args.model == SimLda):
        ldaModel, ldaTopics = load_and_adapt_lda_model(args.ldaModel, data.order)
    else:
        ldaModel, ldaTopics = None, None

    print ("Building template model... ", end="")
    if args.model == CtmBouchard:
        import model.ctm as mdl
        templateModel = mdl.newModelAtRandom(data, K, args.vocabPrior, dtype=output_dtype)
    elif args.model == CtmBohning:
        import model.ctm_bohning as mdl
        templateModel = mdl.newModelAtRandom(data, K, args.vocabPrior, dtype=output_dtype)
    elif args.model == StmYvBouchard:
        import model.stm_yv as mdl
        templateModel = mdl.newModelAtRandom(data, P, K, fv, lfv, args.vocabPrior, dtype=output_dtype)
    elif args.model == StmYvBohning:
        import model.stm_yv_bohning as mdl
        templateModel = mdl.newModelAtRandom(data, P, K, fv, lfv, args.vocabPrior, dtype=output_dtype)
    elif args.model == StmYvBohningFakeOnline:
        import model.stm_yv_bohning_fake_online as mdl
        templateModel = mdl.newModelAtRandom(data, P, K, fv, lfv, args.vocabPrior, dtype=output_dtype)
    elif args.model == MomEm:
        import model.mom_em as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == MomGibbs:
        import model.mom_gibbs as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == LdaCvb:
        import model.lda_cvb as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == LdaCvbZero:
        import model.lda_cvb0 as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == LdaVb:
        import model.lda_vb_python as mdl
        templateModel = mdl.newModelAtRandom(data, K, args.vocabPrior, dtype=output_dtype)
    elif args.model == LdaSvb:
        import model.lda_vb_python as mdl
        templateModel = mdl.newModelAtRandom(data, K, args.vocabPrior, dtype=output_dtype)
    elif args.model == LdaGibbs:
        import model.lda_gibbs as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == Rtm:
        import model.rtm as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == Mtm:
        import model.mtm2 as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == Mtm2:
        import model.mtm3 as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == Dmr:
        import model.dmr as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == Lro:
        import model.lro_vb as mdl
        templateModel = mdl.newModelAtRandom(data, K, dtype=output_dtype)
    elif args.model == SimLda:
        import model.sim_based_rec as mdl
        templateModel = mdl.newModelAtRandom(data, K, method=mdl.LDA, dtype=output_dtype)
    elif args.model == SimTfIdf:
        import model.sim_based_rec as mdl
        templateModel = mdl.newModelAtRandom(data, K, method=mdl.TF_IDF, dtype=output_dtype)
    else:
        raise ValueError ("Unknown model identifier " + args.model)
    print("Done")

    if args.sgd_batch_size > 0 and model_supports_sgd(args.model):
        trainPlan = mdl.newTrainPlan(
                        args.iters,
                        batchSize=args.sgd_batch_size,
                        rate_retardation=args.sgd_retardation_rate,
                        forgetting_rate=args.sgd_forget_rate,
                        debug=args.debug)
    else:
        trainPlan = mdl.newTrainPlan(args.iters, debug=args.debug)


    queryPlan = mdl.newTrainPlan(args.query_iters, debug=args.debug)

    if args.eval == Perplexity:
        return cross_val_and_eval_perplexity(data, mdl, templateModel, trainPlan, queryPlan, args.folds, args.eval_fold_count, args.out_model)
    elif args.eval == HashtagPrecAtM:
        return cross_val_and_eval_hashtag_prec_at_m(data, mdl, templateModel, trainPlan, load_dict(args.word_dict), args.folds, args.eval_fold_count, args.out_model)
    elif args.eval == MeanAveragePrecAllDocs:
        return link_split_map (data, mdl, templateModel, trainPlan, args.folds, args.out_model)
    elif args.eval == MeanPrecRecAtMAllDocs:
        return link_split_prec_rec (data, mdl, templateModel, trainPlan, args.folds, args.eval_fold_count, args.out_model, ldaModel, ldaTopics)
    elif args.eval == LroMeanPrecRecAtMAllDocs:
        return insample_lro_style_prec_rec (data, mdl, templateModel, trainPlan, args.folds, args.eval_fold_count, args.out_model, ldaModel, ldaTopics)
    elif args.eval == LroMeanPrecRecAtMFeatSplit:
        return outsample_lro_style_prec_rec (data, mdl, templateModel, trainPlan, features_mask, args.out_model, ldaModel, ldaTopics)
    else:
        raise ValueError("Unknown evaluation metric " + args.eval)

    return modelFiles


def load_and_adapt_lda_model(path, desired_order):
    '''
    Loads the given LDA model, and ensure it follows the given order
    :param path: the path to the LDA model files (see save_model())
    :param desired_order: re-arrange the given elements to respect
    the given order
    :return: an LDA model and the trained-document topics
    '''
    with open(path, "rb") as f:
        (saved_order, _, _, _, model, train_tops, _) = pkl.load(f)

    if np.all(saved_order == desired_order):
        return model, train_tops
    else:
        raise ValueError("Have not implemented code to re-order saved models")


def load_dict(dict_path):
    with open(dict_path, 'rb') as f:
        return pkl.load(f)


def popular_hashtag_indices(data, word_dict, count=50):
    '''
    Use the word_dict to identify which indices in the given words dictionary
    in the data correspond to
    :param data: the dataset, particularly the words
    :param word_dict: the words dictionary, just a list of words
    :param count: how many hashtag indices should we return
    :return: the indices of the most popular hashtags
    '''
    hashtags        = [w for w in word_dict if w[0] == '#']
    hashtag_indices = [word_dict.index(h) for h in hashtags]
    hashtag_counts  = np.squeeze(np.array(data.words[:,hashtag_indices].sum(axis=0)))

    popular_hashtag_count_indices = hashtag_counts.argsort()[-count:][::-1]
    popular_hashtag_indices = [hashtag_indices[i] for i in popular_hashtag_count_indices]
    # popular_hashtags        = [word_dict[i] for i in popular_hashtag_indices]

    return popular_hashtag_indices


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
    i4 or i32, or u2 or u16, corresponding to 32 and 64 bit floating
    point numbers, 32-bit signed integers, and 16-bit unsigned
    integers respectively. Case is not sensitive.
    '''
    if   dtype_str in ['f8', 'F8', 'f64', 'F64']:
        return np.float64
    elif dtype_str in ['f4', 'F4', 'f32', 'F32']:
        return np.float32
    elif dtype_str in ['i4', 'I4', 'i32', 'I32']:
        return np.int32
    elif dtype_str in ['u2', 'U2', 'u16', 'U16']:
        return np.uint16
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
    :param model_dir: if not none, the models are stored in this directory.
    :return: the list of model files stored
    '''
    model_files = []
    if fold_run_count < 1:
        fold_run_count = num_folds

    if num_folds == 1:
        model = mdl.newModelFromExisting(sample_model)
        query = mdl.newQueryState(data, model)

        model, train_tops, (train_itrs, train_vbs, train_likes) = mdl.train(data, model, query, train_plan)
        likely = mdl.log_likelihood(data, model, train_tops)
        perp   = perplexity_from_like(likely, data.word_count)

        print("Train-set Likelihood: %12f" % (likely))
        print("Train-set Perplexity: %12f" % (perp))

        model_files = save_if_necessary(model_files, model_dir, model, data, 0, train_itrs, train_vbs, train_likes, train_tops, train_tops, mdl)
        return model_files

    query_like_sum    = 0 # to calculate the overall likelihood and
    query_wcount_sum  = 0 # perplexity for the whole dataset
    train_like_sum    = 0
    train_wcount_sum  = 0
    folds_finished    = 0 # count of folds that finished successfully

    fold = 0
    while fold < num_folds and folds_finished < fold_run_count:
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

            print ("DEBUG Train perplexity is " + str(train_perp))

            # Query the model - if there are no features we need to split the text
            print ("Starting query.")
            query_estim, query_eval = query_data.doc_completion_split()
            query_tops              = mdl.newQueryState(query_estim, model)
            _, query_tops = mdl.query(query_estim, model, query_tops, query_plan)

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
            model_files = save_if_necessary(model_files, model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, query_tops, mdl)
        # except Exception as e:
        #     traceback.print_exc()
        #     print("Abandoning fold %d due to the error : %s" % (fold, str(e)))
        finally:
            fold += 1

    print ("Total (%d): Train-set Likelihood: %12.3f \t Train-set Perplexity: %12.3f" % (folds_finished, train_like_sum, perplexity_from_like(train_like_sum, train_wcount_sum)))
    print ("Total (%d): Query-set Likelihood: %12.3f \t Query-set Perplexity: %12.3f" % (folds_finished, query_like_sum, perplexity_from_like(query_like_sum, query_wcount_sum)))

    return model_files


def cross_val_and_eval_hashtag_prec_at_m(data, mdl, sample_model, train_plan, word_dict, num_folds, fold_run_count=-1, model_dir= None):
    '''
    Evaluate the precision at M for the top 50 hash-tags. In the held-out set, the hashtags
    are deleted. We train on all, both training and held-out, then evaluate the precision
    at M for the hashtags

    For values of M we use 10, 50, 100, 150, 250, 500


    :param data: the DataSet object with the data
    :param mdl:  the module with the train etc. functin
    :param sample_model: a preconfigured model which is cloned at the start of each
            cross-validation run
    :param train_plan:  the training plan (number of iterations etc.)
    :param word_dict the word dictionary, used to identify hashtags and print them
    out when the run is completed.
    :param num_folds:  the number of folds to cross validation
    :param fold_run_count: for debugging stop early after processing the number
    of the folds
    :param model_dir: if not none, the models are stored in this directory.
    :return: the list of model files stored
    '''
    MS = [10, 50, 100, 150, 200, 250, 1000, 1500, 3000, 5000, 10000]
    Precision, Recall = "precision", "recall"

    model_files = []
    if fold_run_count < 1:
        fold_run_count = num_folds
    if num_folds <= 1:
        raise ValueError ("Number of folds must be greater than 1")

    hashtag_indices = popular_hashtag_indices (data, word_dict, 50)

    folds_finished = 0 # count of folds that finished successfully
    fold = 0
    while fold < num_folds and folds_finished < fold_run_count:
        try:
            train_range, query_range = data.cross_valid_split_indices(fold, num_folds)

            segment_with_htags             = data.words[train_range, :]
            held_out_segment_with_htags    = data.words[query_range, :]
            held_out_segment_without_htags = data.words[query_range, :]
            held_out_segment_without_htags[:, hashtag_indices] = 0

            train_words = ssp.vstack((segment_with_htags, held_out_segment_without_htags))
            train_data  = data.copy_with_changes(words=train_words)

            # Train the model
            print ("Duplicating model template... ", end="")
            model      = mdl.newModelFromExisting(sample_model)
            train_tops = mdl.newQueryState(train_data, model)

            print ("Starting training")
            model, train_tops, (train_itrs, train_vbs, train_likes) \
                = mdl.train(train_data, model, train_tops, train_plan)

            # Predict hashtags
            dist = rowwise_softmax(train_tops.means)

            # For each hash-tag, for each value of M, evaluate the precision
            results = {Recall : dict(), Precision : dict()}
            for hi in hashtag_indices:
                h_probs = dist[query_range,:].dot(model.vocab[:,hi])
                h_count = held_out_segment_with_htags[:, hi].sum()

                results[Recall][word_dict[hi]]    = { -1 : h_count }
                results[Precision][word_dict[hi]] = { -1 : h_count }
                for m in MS:
                    top_m = h_probs.argsort()[-m:][::-1]

                    true_pos = held_out_segment_with_htags[top_m, hi].sum()
                    rec_denom = min(m, h_count)
                    results[Precision][word_dict[hi]][m] = true_pos / m
                    results[Recall][word_dict[hi]][m]    = true_pos / rec_denom

            print ("%10s\t%20s\t%6s\t" % ("Metric", "Hashtag", "Count") + "\t".join("%5d" % m for m in MS))
            for htag, prec_results in results[Precision].items():
                print ("%10s\t%20s\t%6d\t%s" % ("Precision", htag, prec_results[-1], "\t".join(("%0.3f" % prec_results[m] for m in MS))))
            for htag, prec_results in results[Recall].items():
                print ("%10s\t%20s\t%6d\t%s" % ("Recall", htag, prec_results[-1], "\t".join(("%0.3f" % prec_results[m] for m in MS))))


            # Save the model
            model_files = save_if_necessary(model_files, model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, None, mdl)
        except Exception as e:
            traceback.print_exc()
            print("Abandoning fold %d due to the error : %s" % (fold, str(e)))
        finally:
            fold += 1


    return model_files


def save_if_necessary (model_files, model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, query_tops, mdl):
    if model_dir is not None:
        model_files.append( \
            save_model(\
                model_dir, model, data, \
                fold, train_itrs, train_vbs, train_likes, \
                train_tops, query_tops, \
                mdl))
    return model_files


def save_model(model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, query_tops, mdl):
    P = 0 if 'P' not in dir(model) else model.P

    model_file = newModelFile(model.name, model.K, P, fold, model_dir)

    if model.name == "stm-yv/bohning":
        train_tops = mdl.QueryState(
            train_tops.means,
            None,
            train_tops.varcs,
            train_tops.docLens
        )
        if query_tops is not None:
            query_tops = mdl.QueryState(
                query_tops.means,
                None,
                query_tops.varcs,
                query_tops.docLens
            )

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

        print("Training perplexity is %.2f " % perplexity_from_like(mdl.log_likelihood(train_data, model, train_tops), train_data.word_count))

        min_link_probs       = mdl.min_link_probs(model, train_tops, query_data.links)
        predicted_link_probs = mdl.link_probs(model, train_tops, min_link_probs)

        map = mean_average_prec (query_data.links, predicted_link_probs)
        print ("Fold %2d: Mean-Average-Precision %6.3f" % (fold, map))

        model_files = save_if_necessary(model_files, model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, train_tops, mdl)

    return model_files


def link_split_prec_rec (data, mdl, sample_model, train_plan, folds, target_folds=None, model_dir=None, ldaModel=None, ldaTopics=None):
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
    :param target_folds: the number of folds to complete before finishing. Set
    to folds by default
    :param model_dir: if not none, and folds > 1, the models are stored in this
    directory.
    :param ldaModel: for those models that utilise and LDA component, a pre-trained
    LDA model can be supplied.
    :param ldaTopics: the topics of all documents in the corpus as given by the ldaModel
    :return: the list of model files stored
    '''
    ms = [10, 20, 30, 40, 50, 75, 100, 150, 250, 500]
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

    if ldaModel is not None:
        (_, _, _, _, ldaModel, ldaTopics, _) = ldaModel
    if target_folds is None:
        target_folds = folds

    combi_precs, combi_recs, combi_dcounts = None, None, None
    mrr_sum, mrr_doc_count = 0, 0
    map_sum, map_doc_count = 0, 0
    for fold in range(target_folds):
        model = mdl.newModelFromExisting(sample_model, withLdaModel=ldaModel) \
                if sample_model.name == LRO_MODEL_NAME \
                else mdl.newModelFromExisting(sample_model)

        train_data, query_data = data.link_prediction_split(symmetric=False)
        train_data = prepareForTraining(train_data) # make symmetric, if necessary, after split, so we
                                                    # can compare symmetric with non-symmetric models
        train_tops = mdl.newQueryState(train_data, model)
        model, train_tops, (train_itrs, train_vbs, train_likes) = \
            mdl.train(train_data, model, train_tops, train_plan)

        print ("Training perplexity is %.2f " % perplexity_from_like(mdl.log_likelihood(train_data, model, train_tops), train_data.word_count))

        min_link_probs       = mdl.min_link_probs(model, train_tops, query_data.links)
        predicted_link_probs = mdl.link_probs(model, train_tops, min_link_probs)
        expected_links       = query_data.links

        precs, recs, doc_counts = mean_prec_rec_at (expected_links, predicted_link_probs, at=ms, groups=[(0,3), (3,5), (5,10), (10,1000)])
        print ("Fold %2d: Mean-Precisions at \n" % fold, end="")

        printTable("Precision", precs, doc_counts, ms)
        printTable("Recall",    recs,  doc_counts, ms)

        mrr = mean_reciprocal_rank(expected_links, predicted_link_probs)
        print ("Mean reciprocal-rank : %f" % mrr)
        mrr_sum       += mrr * expected_links.shape[0]
        mrr_doc_count += expected_links.shape[0]

        map = mean_average_prec (expected_links, predicted_link_probs)
        print ("Mean Average Precision : %f" % map)
        map_sum       += map * expected_links.shape[0]
        map_doc_count += expected_links.shape[0]

        combi_precs, _             = combine_map(combi_precs, combi_dcounts, precs, doc_counts)
        combi_recs,  combi_dcounts = combine_map(combi_recs,  combi_dcounts, recs,  doc_counts)

        model_files = save_if_necessary(model_files, model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, train_tops, mdl)

    print ("-" * 80 + "\n\n Final Results\n\n")
    printTable("Precision", combi_precs, combi_dcounts, ms)
    printTable("Recall",    combi_recs,  combi_dcounts, ms)
    print("Mean reciprocal-rank: %f" % (mrr_sum / mrr_doc_count))

    return model_files


def insample_lro_style_prec_rec (data, mdl, sample_model, train_plan, folds, target_folds=None, model_dir=None, ldaModel=None, ldaTopics=None):
    '''
    For documents with > 5 links remove a portion. The portion is determined by
    the number of folds (e.g. five-fold implied remove one fifth of links, three
    fold implies remove a third, etc.)

    Train on all documents and all remaining links.

    Predict remaining links.

    Evaluate using precision@m, recall@m, mean reciprocal-rank and
    mean average-precision

    Average all results.

    :param data: the DataSet object with the data
    :param mdl:  the module with the train etc. functin
    :param sample_model: a preconfigured model which is cloned at the start of each
            cross-validation run
    :param train_plan:  the training plan (number of iterations etc.)
    :param folds:  the number of folds to cross validation
    :param target_folds: the number of folds to complete before finishing. Set
    to folds by default
    :param model_dir: if not none, and folds > 1, the models are stored in this
    directory.
    :param ldaModel: for those models that utilise and LDA component, a pre-trained
    LDA model can be supplied.
    :param ldaTopics: the topics of all documents in the corpus as given by the ldaModel
    :return: the list of model files stored
    '''
    ms = [10, 20, 30, 40, 50, 75, 100, 150, 250, 500]
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

    if target_folds is None:
        target_folds = folds

    combi_precs, combi_recs, combi_dcounts = None, None, None
    mrr_sum, mrr_doc_count = 0, 0
    map_sum, map_doc_count = 0, 0
    fold_count = 0
    for fold in range(folds):
        # try:
        # Prepare the training and query data
        train_data, query_data, docSubset = data.folded_link_prediction_split(MinLinkCountEval, fold, folds)
        train_data = prepareForTraining(train_data) # make symmetric, if necessary, after split, so we
                                                    # can compare symmetric with non-symmetric models

        print ("\n\nFold %d\n" % fold + ("-" * 80))

        # Train the model
        if model_uses_lda(sample_model):
            model      = mdl.newModelFromExisting(sample_model, withLdaModel=ldaModel)
            train_tops = mdl.newQueryState(train_data, model, withLdaTopics=ldaTopics)
        else:
            model = mdl.newModelFromExisting(sample_model)
            train_tops = mdl.newQueryState(train_data, model)

        model, train_tops, (train_itrs, train_vbs, train_likes) = \
            mdl.train(train_data, model, train_tops, train_plan)

        print ("Training perplexity is %.2f " % perplexity_from_like(mdl.log_likelihood(train_data, model, train_tops), train_data.word_count))

        # Infer the expected link probabilities
        min_link_probs       = mdl.min_link_probs(model, train_tops, train_tops, query_data.links, docSubset)
        predicted_link_probs = mdl.link_probs(model, train_tops, train_tops, min_link_probs, docSubset)
        expected_links       = query_data.links[docSubset, :]

        # Evaluation 1/3: Precision and Recall at M
        precs, recs, doc_counts = mean_prec_rec_at (expected_links, predicted_link_probs, at=ms, groups=[(0,3), (3,5), (5,10), (10,1000)])
        print ("Fold %2d: Mean-Precisions at \n" % fold, end="")

        printTable("Precision", precs, doc_counts, ms)
        printTable("Recall",    recs,  doc_counts, ms)

        combi_precs, _             = combine_map(combi_precs, combi_dcounts, precs, doc_counts)
        combi_recs,  combi_dcounts = combine_map(combi_recs,  combi_dcounts, recs,  doc_counts)

        # Evaluation 2/3: Mean Reciprocal-Rank
        mrr = mean_reciprocal_rank(expected_links, predicted_link_probs)
        print ("Mean reciprocal-rank : %f" % mrr)
        mrr_sum       += mrr * expected_links.shape[0]
        mrr_doc_count += expected_links.shape[0]

        # Evaluation 3/3: Mean Average-Precision
        map = mean_average_prec (expected_links, predicted_link_probs)
        print ("Mean Average Precision : %f" % map)
        map_sum       += map * expected_links.shape[0]
        map_doc_count += expected_links.shape[0]

        # Save the files if necessary and move onto the next fold if required
        model_files = save_if_necessary(model_files, model_dir, model, data, fold, train_itrs, train_vbs, train_likes, train_tops, train_tops, mdl)
        fold_count += 1
        if fold_count == target_folds:
            break
        # except Exception as e:
        #     print("Fold " + str(fold) + " failed: " + str(e))

    print ("-" * 80 + "\n\n Final Results\n\n")
    printTable("Precision", combi_precs, combi_dcounts, ms)
    printTable("Recall",    combi_recs,  combi_dcounts, ms)
    print("Mean reciprocal-rank: %f" % (mrr_sum / mrr_doc_count))
    print("Mean average-precision: %f" % (map_sum / map_doc_count))

    return model_files


def outsample_lro_style_prec_rec (data, mdl, sample_model, train_plan, feature_mask, model_dir=None, ldaModel=None, ldaTopics=None):
    '''
    Take a feature list. Train on all documents where none of those features
    are set. Remove the first element from the feature list, query all documents
    with that feature set, and then evaluate link prediction. Repeat until
    feature-list is empty.

    :param data: the DataSet object with the data
    :param mdl:  the module with the train etc. functin
    :param sample_model: a preconfigured model which is cloned at the start of each
            cross-validation run
    :param train_plan:  the training plan (number of iterations etc.)
    :param feature_mask:  the list of features used to separate training from query
    This is a list of tuples, the left side is the feature label, the right side
    is the
    :param model_dir: if not none, the models are stored in this directory.
    :param ldaModel: for those models that utilise and LDA component, a pre-trained
    LDA model can be supplied.
    :param ldaTopics: the topics of all documents in the corpus as given by the ldaModel
    :return: the list of model files stored
    '''
    def prepareForTraining(data):
        if mdl.is_undirected_link_predictor():
            result = data.copy()
            result.convert_to_undirected_graph()
            result.convert_to_binary_link_matrix()
            return result
        else:
            return data

    ms = [10, 20, 30, 40, 50, 75, 100, 150, 250, 500]
    model_files = []

    combi_precs, combi_recs, combi_dcounts = None, None, None
    mrr_sum, mrr_doc_count = 0, 0
    map_sum, map_doc_count = 0, 0
    while len(feature_mask) > 0:
        # try:
        # Prepare the training and query data
        feature_mask_indices = [i for _,i in feature_mask]
        train_data, query_data, train_indices = data.split_on_feature(feature_mask_indices)
        (feat_label, feat_id) = feature_mask.pop(0)
        print ("\n\nFeature: %s\n" % (feat_label,) + ("-" * 80))

        train_data = prepareForTraining(train_data) # make symmetric, if necessary, after split, so we
                                                    # can compare symmetric with non-symmetric models

        # Train the model
        if model_uses_lda(sample_model):
            ldaModelSubset, ldaTopicsSubset = subsetLda(ldaModel, ldaTopics, train_indices)
            model      = mdl.newModelFromExisting(sample_model, withLdaModel=ldaModelSubset)
            train_tops = mdl.newQueryState(train_data, model, withLdaTopics=ldaTopicsSubset)
        else:
            model = mdl.newModelFromExisting(sample_model)
            train_tops = mdl.newQueryState(train_data, model)

        model, train_tops, (train_itrs, train_vbs, train_likes) = \
            mdl.train(train_data, model, train_tops, train_plan)

        print ("Training perplexity is %.2f " % perplexity_from_like(mdl.log_likelihood(train_data, model, train_tops), train_data.word_count))

        # Infer the expected link probabilities
        query_tops    = mdl.newQueryState(query_data, model)
        _, query_tops = mdl.query(query_data, model, query_tops, train_plan)

        min_link_probs       = mdl.min_link_probs(model, train_tops, query_tops, query_data.links)
        predicted_link_probs = mdl.link_probs(model, train_tops, query_tops, min_link_probs)
        expected_links       = query_data.links

        # Evaluation 1/3: Precision and Recall at M
        precs, recs, doc_counts = mean_prec_rec_at (expected_links, predicted_link_probs, at=ms, groups=[(0,3), (3,5), (5,10), (10,1000)])
        print (" Mean-Precisions for feature %s (#%d)" % (feat_label, feat_id), end="")

        printTable("Precision", precs, doc_counts, ms)
        printTable("Recall",    recs,  doc_counts, ms)

        combi_precs, _             = combine_map(combi_precs, combi_dcounts, precs, doc_counts)
        combi_recs,  combi_dcounts = combine_map(combi_recs,  combi_dcounts, recs,  doc_counts)

        # Evaluation 2/3: Mean Reciprocal-Rank
        mrr = mean_reciprocal_rank(expected_links, predicted_link_probs)
        print ("Mean reciprocal-rank : %f" % mrr)
        mrr_sum       += mrr * expected_links.shape[0]
        mrr_doc_count += expected_links.shape[0]

        # Evaluation 3/3: Mean Average-Precision
        map = mean_average_prec (expected_links, predicted_link_probs)
        print ("Mean Average Precision : %f" % map)
        map_sum       += map * expected_links.shape[0]
        map_doc_count += expected_links.shape[0]

        # Save the files if necessary and move onto the next fold if required
        model_files = save_if_necessary(model_files, model_dir, model, data, feat_id, train_itrs, train_vbs, train_likes, train_tops, train_tops, mdl)
        # except Exception as e:
        #     print("Fold " + str(fold) + " failed: " + str(e))

    print ("-" * 80 + "\n\n Final Results\n\n")
    printTable("Precision", combi_precs, combi_dcounts, ms)
    printTable("Recall",    combi_recs,  combi_dcounts, ms)
    print("Mean reciprocal-rank: %f" % (mrr_sum / mrr_doc_count))
    print("Mean average-precision: %f" % (map_sum / map_doc_count))

    return model_files


def model_uses_lda(model):
    return model.name == LRO_MODEL_NAME or model.name == LDA_SIM_MODEL_NAME


def subsetLda(ldaModel, ldaTopics, doc_indices):
    '''
    Take a fully trainined LDA model, and then return only those rows
    corresponding to the given document indices
    :param ldaModel: the model, returned as is
    :param ldaTopics: the topics
    :param doc_indices: a simple list of integers in the range 0..D where
    D is the number of documents used to train the LDA model.
    :return:
    '''
    if ldaModel is None:
        return None
    else:
        if ldaModel.name == LDA_GIBBS_MODEL_NAME:
            return ldaModel, pruneLdaGibbsQueryState(ldaTopics, doc_indices)
        elif ldaModel.name == LDA_VB_MODEL_NAME:
            return ldaModel, pruneLdaVbQueryState(ldaTopics, doc_indices)
        else:
            raise ValueError("Can't prune queries for LDA models of type " + ldaModel.name)


def combine_map (old_avgs, old_counts, new_avgs, new_counts):
    '''
    Given two maps of averages, organised as group-key -> [avg], we use
    the document counts map (organised in the same fashion) to combine the averages
    and counts in the "old" maps with the new values. The updated averages and counts
    maps are returned
    :return: a tuple with updated avgs and updated counts in that order
    '''
    if old_avgs is None and old_counts is None:
        # create a copy
        return { g:lst[:] for g,lst in new_avgs.items() }, \
               { g:cnt for g,cnt in new_counts.items() }

    upd_counts = { g:old_count + new_counts[g] for g,old_count in old_counts.items() }
    upd_avgs   = { g:[(o * old_counts[g] + n * new_counts[g]) / upd_counts[g] \
                      for o,n in zip(lst,new_avgs[g])] \
                   for g,lst in old_avgs.items() }

    return upd_avgs, upd_counts



def printTable(title, scores, doc_counts, ms):
    print(title)
    print("| Group    | Doc Count | " + " | ".join("%5d" % m for m in ms)    + " |")
    print("|----------|-----------|-" + "-|-".join("-----" for _ in range(len(ms))) + "-|")

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

def parse_features_mask(args):
    if args.features_mask_str is not None:
        features_mask = []
        parts = args.features_mask_str.split(',')
        for part in parts:
            subparts = part.split(':')
            features_mask.append((subparts[0], int(subparts[1])))
        return features_mask
    else:
        return None


if __name__ == '__main__':
    run(args=sys.argv[1:])
