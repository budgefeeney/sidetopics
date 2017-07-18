#!/bin/bash


DATA_DIR="/Users/bryanfeeney/iCloud/Datasets"
NIPS_DIR="$DATA_DIR/NIPS-from-pryor-Sep15"
NIPS_FEATS="$NIPS_DIR/X_ar.pkl"
NIPS_WORDS="$NIPS_DIR/W_ar.pkl"

REUTERS_DIR="$DATA_DIR/reuters"
REUTERS_FEATS="$REUTERS_DIR/NO_FEATURES_EXIST_FOR_REUTERS"
REUTERS_WORDS="$REUTERS_DIR/W.pkl"

TNEWS_DIR="$DATA_DIR/20news4"
TNEWS_FEATS="$TNEWS_DIR/NO_FEATURES_EXIST_FOR_TNEWS"
TNEWS_WORDS="$TNEWS_DIR/W.pkl"

if [ "nips" == $1 ];
then
	WORDS=$NIPS_WORDS
	FEATS=$NIPS_FEATS
elif [ "reut" == $1 ];
then
	WORDS=$REUTERS_WORDS
	FEATS=$REUTERS_FEATS
elif [ "tnews" == $1 ];
then
	WORDS=$TNEWS_WORDS
	FEATS=$TNEWS_FEATS
else
	echo "Don't know that dataset: $1"
	exit;
fi

if [ "big" == $2 ];
then
	TOPIC_COUNTS="100 150 200"
else
	TOPIC_COUNTS="5 10 25 50"
fi

ALGORS=$3

FOLDS=5
MIN_FOLDS=3
EVAL="perplexity"

TRAIN_ITERS=500
QUERY_ITERS=100

OUT_DIR=`dirname $0`
BACKUP=$PWD
cd $OUT_DIR
OUT_DIR=$PWD
cd $BACKUP 

STM_EXEC=`which python3`
SRC_DIR="/Users/bryanfeeney/Workspace/sidetopics/src"
cd $SRC_DIR
STM_EXEC="$STM_EXEC run/main.py "
export PYTHONPATH=$SRC_DIR:$PYTHONPATH

OUT_DIR="$OUT_DIR/output"
test -d $OUT_DIR || mkdir -v $OUT_DIR
OUT_MODELS="$OUT_DIR/models"
test -d $OUT_MODELS || mkdir -v $OUT_MODELS
OUT_LOGS="$OUT_DIR/logs"
test -d $OUT_LOGS || mkdir -v $OUT_LOGS



for ALGOR in $ALGORS
do
	for TOPIC_COUNT in $TOPIC_COUNTS
	do
		LOG_FILE="$OUT_LOGS/Job-$1-$ALGOR-K-$TOPIC_COUNT"
	
		echo "$STM_EXEC \
		--model $ALGOR \
		--num-topics $TOPIC_COUNT  \
		--folds $FOLDS \
		--truncate-folds $MIN_FOLDS \
		--eval $EVAL \
		--iters $TRAIN_ITERS \
		--query-iters $QUERY_ITERS \
		--out-model $OUT_MODELS \
		--words $WORDS \
		>$LOG_FILE.out 2>$LOG_FILE.err"
		
		$STM_EXEC \
		--model $ALGOR \
		--num-topics $TOPIC_COUNT  \
		--folds $FOLDS \
		--truncate-folds $MIN_FOLDS \
		--eval $EVAL \
		--iters $TRAIN_ITERS \
		--query-iters $QUERY_ITERS \
		--out-model $OUT_MODELS \
		--words $WORDS \
		>$LOG_FILE.out 2>$LOG_FILE.err
	done
done

