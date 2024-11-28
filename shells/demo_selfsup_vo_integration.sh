#!/bin/bash
# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 
# Set SELFSUP_CKPT_OVERRIDE to your pre-trained model by vidar
# (e.g.) SELFSUP_CKPT_OVERRIDE=/data/checkpoints/vo_demo/2024-08-27_09h44m17s/models/best.ckpt

SELFSUP_CKPT_OVERRIDE=
RGB_FOLDER="/data/datasets/DDAD/{}/rgb_1216_1936/CAMERA_01"
FIRST_GRAPH_OPTIM="--posenet_init_first_graph"
PLOT_ARG="--plot --zero_origin"
OPTIM_ARG="--vidar_config demo_configs/selfsup_resnet18_vo_calib.yaml"
SEQ="000150"
WEIGHT=/data/models/papers/droid.pth

python evaluation_scripts/launch.py \
  --override_ckpt ${SELFSUP_CKPT_OVERRIDE} \
  --weights ${WEIGHT} \
  --src_rgb_path ${RGB_FOLDER} \
  --dataset "UncalibratedFolder" \
  --use_pred_intrinsics \
  ${PLOT_ARG} \
  ${FIRST_GRAPH_OPTIM} \
  ${OPTIM_ARG} --seqs ${SEQ}