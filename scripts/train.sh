#!/usr/bin/env bash
set -x
DATAPATH="/home/mz/llreda/Stereo_Dataset"
CSV_ROOT="./csvfiles"

python main.py --model gwc_gc --datapath $DATAPATH  --dataset_csv_root $CSV_ROOT