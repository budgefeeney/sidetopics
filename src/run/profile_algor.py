__author__ = 'bryanfeeney'


import cProfile

from run.main import run, Rtm, LdaGibbs, LdaVb, Mtm
from model.evals import Perplexity, MeanAveragePrecAllDocs

AclPath = "/Users/bryanfeeney/iCloud/Datasets/ACL/ACL.100/"
AclWordPath = AclPath + "words-freq.pkl"
AclCitePath = AclPath + "ref.pkl"

NipsPath = "/Users/bryanfeeney/iCloud/Datasets/NIPS-from-pryor-Sep15/"
NipsWordPath = NipsPath + "W_ar.pkl"
NipsFeatPath = NipsPath + "X_ar.pkl"


def runAlgorithm():
    K,P = 10, 75
    for modelName in [ Rtm ]: #ModelNames:
        cmdline = '' \
                + ' --debug '          + "False" \
                + ' --model '          + modelName \
                + ' --dtype '          + 'f8:f8'      \
                + ' --num-topics '     + str(K)    \
                + ' --log-freq '       + '5'       \
                + ' --eval '           + 'perplexity'  \
                + ' --iters '          + '5'      \
                + ' --query-iters '    + '5'      \
                + ' --folds '          + '2'      \
                + ' --words '          + AclWordPath \
                + ' --links '          + AclCitePath \
                + ' --limit-to '       + '100000' \
                + ' --eval '           + MeanAveragePrecAllDocs \
                + ' --out-model '      + '/Users/bryanfeeney/Desktop/acl-out'
#                     + ' --words '          + '/Users/bryanfeeney/Dropbox/Datasets/ACL/words.pkl' \
#                     + ' --words '          + '/Users/bryanfeeney/Desktop/NIPS-from-pryor-Sep15/W_ar.pkl'
#                      + ' --words '          + '/Users/bryanfeeney/Desktop/Dataset-Sep-2014/words.pkl' \
#                      + ' --feats '          + '/Users/bryanfeeney/Desktop/Dataset-Sep-2014/side.pkl'
#                    + ' --words '          + wordsFile \
#                    + ' --feats '          + featsFile
#                    + ' --words '          + '/Users/bryanfeeney/Desktop/Tweets600/words-by-author.pkl' \
#                     + ' --out-model '      + modelFileDir \

        run(cmdline.strip().split(' '))



if __name__ == "__main__":
    cProfile.run ('runAlgorithm()')