#!/usr/bin/env bash
set -x
DATAPATH="/home/mz/llreda/Stereo_Dataset"

python ./processing/dataset_processing.py --datapath $DATAPATH