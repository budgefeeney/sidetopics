#!/bin/bash
PYTHON_EXEC=`which python3.3`

repo=/Users/bryanfeeney/Workspace/sidetopics

dataDir=/Users/bryanfeeney/Desktop/SmallerDB-20
outDir=$dataDir/out
codeDir=$dataDir/code

K=40
Q=10
P=50
algor="uy"

test -d $outDir || mkdir $outdir
if [ ! -d $codeDir ];
then
	mkdir $codeDir
	cd $codeDir
	git clone $repo $codeDir
fi

featFile="$dataDir/side.pkl"
wordFile="$dataDir/words.pkl"
modelFile="$outDir/model"
plotFile="$outDir/plot"
 
cmdline=""
cmdline="$cmdline --model $algor"
cmdline="$cmdline --num-topics $K"
cmdline="$cmdline --num-lat-topics $Q"
cmdline="$cmdline --num-lat-feats $P"
cmdline="$cmdline --eval likely"
cmdline="$cmdline --out-model $modelFile"
cmdline="$cmdline --out-plot $plotFile"
cmdline="$cmdline --log-freq 100"
cmdline="$cmdline --iters 500"
cmdline="$cmdline --query-iters 50"
cmdline="$cmdline --min-vb-change 0.00001"
cmdline="$cmdline --topic-var 0.01"
cmdline="$cmdline --feat-var 0.01"
cmdline="$cmdline --lat-topic-var 1"
cmdline="$cmdline --lat-feat-var 1"
cmdline="$cmdline --folds 5"
cmdline="$cmdline --feats $featFile"
cmdline="$cmdline --words $wordFile"

export PYTHONPATH=$PYTHONPATH:$codeDir
echo "$PYTHON_EXEC $codeDir/run/main.py $cmdline"
$PYTHON_EXEC $codeDir/run/main.py $cmdline


