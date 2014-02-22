#!/bin/bash

HOURS=8

HOME="/home/ucabbfe/Tweets"
FEATS_FILE="$HOME/side-noisy.pkl"
WORDS_FILE="$HOME/words-noisy.pkl"
AUTHOR_FILE="$HOME/words-by-author.pkl"
OUT_PATH="$HOME/out"
STM_EXEC="$HOME/stm.sh"

FOLDS=5
EVAL="perplexity"

TOPIC_COUNTS="5 10 25 50 100 150 250"
LATENT_SIZES="5 10 25 50 75 100"

OBS_FEAT_VAR=0.01
LAT_FEAT_VAR=0.1
OBS_TOPIC_VAR=0.01
LAT_TOPIC_VAR=0.1

TRAIN_ITERS=500
QUERY_ITERS=100


for ALGOR in ctm_bouchard ctm_bohning
do
	for TOPIC_COUNT in $TOPIC_COUNTS
	do
		echo "$STM_EXEC \
		--model $ALGOR \
		--num-topics $TOPIC_COUNT  \
		--folds $FOLDS \
		--eval $EVAL \
		--iters $TRAIN_ITERS \
		--query-iters $QUERY_ITERS \
		--out-model $OUT_PATH \
		--words $AUTHOR_FILE" | qsub -N "Job-$ALGOR-K-$TOPIC_COUNT" -l h_rt=$HOURS:0:0 -l mem_free=8G,h_vmem=12G,tmem=12G -S /bin/bash
	done
done

for ALGOR in stm_yv_bouchard stm_yv_bohning
do
	for TOPIC_COUNT in $TOPIC_COUNTS
	do
		for LATENT_SIZE in $LATENT_SIZES
		do
			echo "$STM_EXEC \
			--model $ALGOR \
			--num-topics $TOPIC_COUNT  \
			--num-lat-feats $LATENT_SIZE \
			--folds $FOLDS \
			--eval $EVAL \
			--iters $TRAIN_ITERS \
			--query-iters $QUERY_ITERS \
			--out-model $OUT_PATH \
			--feats $FEATS_FILE \
			--words $WORDS_FILE" | qsub -N "Job-$ALGOR-K-$TOPIC_COUNT-P-$LATENT_SIZE" -l h_rt=$HOURS:0:0 -l mem_free=8G,h_vmem=12G,tmem=12G -S /bin/bash
		done
	done
done