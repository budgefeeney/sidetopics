#!/bin/sh

PYTHON="/share/apps/python-3.3.2/bin/python3.3"
HOME=`dirname $0`
SRC_BUNDLE="$HOME/sidetopics.tgz"

TIMESTAMP=`date +%s%N`
test -d $HOME/$HOSTNAME || mkdir $HOME/$HOSTNAME

SRC_COPY="$HOME/$HOSTNAME/sidetopics-${TIMESTAMP}${RANDOM}"

# Clearly there's a race condition here, but it's the best of a bad bunch
while [ -d $SRC_COPY ]
do
	TIMESTAMP=`date +%s%N`
	SRC_COPY="$HOME/$HOSTNAME/sidetopics-${TIMESTAMP}${RANDOM}"
done
mkdir $SRC_COPY
tar xvfz $HOME/sidetopics.tgz -C $SRC_COPY

touch $SRC_COPY/configuration.txt
echo "$@" >> $SRC_COPY/configuration.txt
export PYTHONPATH=$SRC_COPY/src:$PYTHONPATH

$PYTHON $SRC_COPY/src/run/main.py $@