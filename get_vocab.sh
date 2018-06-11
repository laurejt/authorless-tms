#! /bin/bash

IN_SEQ=$1
VOCAB=$2

mallet info --input $IN_SEQ --print-feature-counts | cut -f 1 | sort -k 1 > $VOCAB
