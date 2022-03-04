#!/usr/bin/env bash
set -x
DATAPATH="/home/mz/llreda/Stereo_Dataset"
CSV_ROOT="./csvfiles"

python ./processing/get_csvfiles.py --datapath $DATAPATH  --savedir $CSV_ROOT --train_sets [1,2,3,4,5,6] --validation_sets \
[7,] --test_sets [8,9]