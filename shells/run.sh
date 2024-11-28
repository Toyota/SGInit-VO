#!/bin/bash
# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 

FIRST_GRAPH_OPTIM="--posenet_init_first_graph"
PLOT_ARG="--plot --zero_origin"  # Remove --zero_origin to replicate the paper's experiment
OPTIM_ARG="--vidar_config configs/papers/sginit/inference_resnet18s.yaml"  # Inference speed test
# FLOW_DEBUG_CFG=" --show_gru --VKP 7 8 --show_gru_path tests/intermediates/{}" # Viz intermediate representations

SEQ=000191

WEIGHT=/data/models/papers/droid.pth

python evaluation_scripts/launch.py \
  --weights ${WEIGHT} \
  --src_rgb_path "/data/datasets/DDAD/{}/rgb_1216_1936/CAMERA_01" \
  --ref_traj_path "/data/datasets/DDAD/{}/pose/CAMERA_01" \
  --dataset "DDAD" \
  ${PLOT_ARG} \
  ${FLOW_DEBUG_CFG} \
  ${FIRST_GRAPH_OPTIM} \
  ${OPTIM_ARG} --seqs ${SEQ}
  # ${OPTIM_ARG} # For full experiment
