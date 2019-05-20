#!/usr/bin/env bash

clear
projdir=..
datadir=$projdir/data
indir=$datadir/odf-data-2019-05-20-11-04-47

level=$1

/Users/ilandi/anaconda3/bin/python -u $projdir/src/create-vocabulary.py $indir $level