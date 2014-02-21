#!/bin/bash

HOME="/home/ucabbfe/Tweets"
FEATS_FILE="$HOME/side-noisy.pkl"
WORDS_FILE="$HOME/words-noisy.pkl"
AUTHOR_FILE="$HOME/words-by-author.pkl"
OUT_PATH="$HOME/out"
STM_EXEC="$HOME/stm.sh"

FOLDS=5
EVAL="perplexity"

NUM_TOPICS="5 10 25 50 100 150 250"
LATENT_SIZES="5 10 25 50 75 100"

OBS_FEAT_VAR=0.01
LAT_FEAT_VAR=0.1
OBS_TOPIC_VAR=0.01
LAT_TOPIC_VAR=0.1

TRAIN_ITERS=500
QUERY_ITERS=100


for ALGOR in ctm_bouchard ctm_bohning
do
	for NUM_TOPICS in $NUM_TOPICS
	do
		echo "$STM_EXEC \
		--model $ALGOR \
		--num-topics $NUM_TOPICS  \
		--folds $FOLDS \
		--eval $EVAL \
		--iters $TRAIN_ITERS \
		--query-iters $QUERY_ITERS \
		--out-model $OUT_PATH \
		--words $AUTHOR_FILE" | qsub -N "Job-$ALGOR-K-$NUM_TOPICS" -l h_rt=$hours:0:0 -l mem_free=4G,h_vmem=8G,tmem=8G -S /bin/bash
	done
done

for ALGOR in stm_yv_bouchard stm_yv_bohning
do
	for NUM_TOPICS in $NUM_TOPICS
	do
		for LATENT_SIZE in $LATENT_SIZES
		do
			echo "$STM_EXEC \
			--model $ALGOR \
			--num-topics $NUM_TOPICS  \
			--num-lat-feats $LATENT_SIZE \
			--folds $FOLDS \
			--eval $EVAL \
			--iters $TRAIN_ITERS \
			--query-iters $QUERY_ITERS \
			--out-model $OUT_PATH \
			--feats $FEATS_FILE \
			--words $WORDS_FILE" | qsub -N "Job-$ALGOR-K-$NUM_TOPICS-P-$LATENT_SIZE" -l h_rt=$hours:0:0 -l mem_free=4G,h_vmem=8G,tmem=8G -S /bin/bash
		done
	done
done

