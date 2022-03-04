#!/usr/bin/env bash
set -x
DATAPATH="/home/mz/llreda/Stereo_Dataset"
CSV_ROOT="./csvfiles"
LOAD_MODEL="/home/mz/llreda/Stereo_Matching/exp_gwc/checkpoint_2.tar"

python test.py --model gwc_gc --datapath $DATAPATH  --dataset_csv_root $CSV_ROOT --loadmodel $LOAD_MODEL