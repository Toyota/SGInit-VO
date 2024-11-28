#!/bin/bash
# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 

PLOT_ARG="--plot "
OPTIM_ARG="--vidar_config demo_configs/save_zerodepth.yaml --vidar_depth_type perceiver --vidar_pose_type None"

SEQ=000150 # TODO: finally, to be replaced by 000187,  as a failure demonstration

WEIGHT=/data/models/papers/droid.pth

python evaluation_scripts/launch.py \
  --weights ${WEIGHT} \
  --src_rgb_path "/data/datasets/DDAD/{}/rgb_1216_1936/CAMERA_01" \
  --ref_traj_path "/data/datasets/DDAD/{}/pose/CAMERA_01" \
  --dataset "DDAD" \
  ${PLOT_ARG} \
  ${REMOVE_SEQS} \
  ${OPTIM_ARG} --seqs ${SEQ}
  # ${OPTIM_ARG} # For full experiment
