#!/usr/bin/env bash

clear
projdir=..
datadir=$projdir/data
indir=$datadir/odf-data-2019-06-13-12-28-37

level=$1

/Users/ilandi/anaconda3/bin/python -u $projdir/src/create-vocabulary.py $indir $level