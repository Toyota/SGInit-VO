# Copyright 2024 Toyota Motor Corporation.  All rights reserved.
# This source code is partially derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM.git)

import argparse
from typing import Any, List


def sginit_argparse():
    """ Handle the parser to feed the all arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", default=1, type=int, help="frame stride")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--show_cli", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument('--seqs', nargs='+', default=[],
                        help="Specify sequences to track (e.g. --seq 000150 000151 ...)"
                        )
    parser.add_argument("--posenet_init_first_graph", action="store_true")
    parser.add_argument("--skip_post_ba_for_first_graph", action="store_true")
    parser.add_argument("--posenet_in_update_ba", type=str, default=None)
    parser.add_argument("--gt_pose_prior", action="store_true")

    parser.add_argument("--beta", type=float, default=0.5, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.0,
                        help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=3.5, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0,
                        help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=16, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=1, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=0, help="non-maximal supression of edges")

    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--zero_origin", action="store_true",
                        help="Set t=0's Pose as identity matrix forcibly. "
                             "For replicate, DO NOT SET.  "
                             "because this is just for visualization so the paper's result is not from here")
    parser.add_argument("--pkl_suffix", type=str, default="temp/show_gru/{}")  # {} will be replaced by seq

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    # optical-flow analysis
    parser.add_argument("--show_gru", action="store_true")
    parser.add_argument("--show_gru_path", type=str, default="temp/show_gru/{}")  # {} will be replaced by seq
    parser.add_argument('--viz_kf_pairs', '--VKP', nargs='+', default=[],
                        help="Keyframe pairs for visualization (e.g `--VKP 7 8 12 9 `) means 7->8 and 12->9 is tracked"
                        )
    parser.add_argument("--show_gru_ba_steps_upd1", nargs='+', default=[-1], )  # 3 == self.iters1 - 1
    parser.add_argument("--show_gru_ba_steps_upd2", nargs='+', default=[1], )  # 1 == self.iters2 - 1

    # Self-sup integrated operations
    parser.add_argument('--vidar_config', type=str, default=None,
                        help='Path to TRI-VIDAR configuration path to mount Self-Sup priors onto GPU')
    parser.add_argument('--vidar_depth_type', type=str, default='depth',
                        help='Networks predicting depth estimation (default is `depth`, and `perceiver` is supported)')
    parser.add_argument('--vidar_pose_type', type=str, default='pose',
                        help='Networks predicting depth estimation (default is `pose`, and `None` is supported)')
    parser.add_argument('--override_ckpt', type=str, default=None,
                        help='Specify the depth/pose prior .ckpt to load via VIDAR')
    parser.add_argument('--use_pred_intrinsics', action="store_true",
                        help='Use learned Intrinsics loaded from VIDAR')
    parser.add_argument('--resize', nargs='+', default=[384, 640],
                        help='Reshape image size to feed into the self-sup models, [H,W]')

    # Dataset implementations
    parser.add_argument('--src_rgb_path', type=str, default=None,
                        help='Path to RGB stream stored')
    parser.add_argument('--remove_seqs', nargs='+', default=[],
                        help="Skip seqeunces (e.g. --remove_seqs 000150 000151 ...)"
                        )
    parser.add_argument('--ref_traj_path', type=str, default=None,
                        help='Path to RGB stream stored')
    parser.add_argument('--rgb_ext', type=str, default='.jpg', help='Extension for RGB file')
    parser.add_argument('--dataset', type=str, default=None,
                        help='DatasetClass name to be load (see droid_slam/datasets/Dataset###.py)')
    parser.add_argument('--raw_rgb_size', nargs='+', default=[1216, 1936],
                        help='Raw image shape [H,W]')
    parser.add_argument('--calib', nargs='+', default=None, help='Pinhole camera model parameter: (fx,fy,cx,cy)')
    return parser.parse_args()
