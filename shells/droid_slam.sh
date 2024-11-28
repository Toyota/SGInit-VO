#!/bin/bash
# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 

PLOT_ARG="--plot --zero_origin"
CALIB="--calib 535.4 539.2 320.1 247.6"
SEQ="rgbd_dataset_freiburg3_cabinet"

WEIGHT=/data/models/papers/droid.pth

python evaluation_scripts/launch.py \
  --stride 3 \
  --weights ${WEIGHT} \
  --src_rgb_path "/data/datasets/droid-slam/{}/rgb" \
  --dataset "UncalibratedFolder" \
  --rgb_ext ".png" \
  --resize 240 320 \
  ${PLOT_ARG} \
  ${CALIB} \
  --seqs ${SEQ}