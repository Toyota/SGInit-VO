# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 
# The implementation is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)

from argparse import Namespace

from collections import OrderedDict
import os
import time
from typing import Any, List, Callable
import warnings

import cv2
import numpy as np
import matplotlib
import pandas as pd
from PIL.Image import Image
from termcolor import cprint
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import EvalDataSet
from datasets.utils import calib2intrinics_mapper
from datasets.DatasetBase import DatasetBase
from data_readers.augmentation import def_transforms
from droid import Droid
from droid_utils.parsers import sginit_argparse
from droid_utils.config import arg_has_false as arg_has
from droid_utils.config import arg_has as arg_has_settable
from droid_utils.config import args2dict, args_override
from geom.pose_utils import mat_homogeneous2pose_vec
from geom.projective_ops import kmat2pinhole
from modules.selfsup_wrapper import VidarSelfSupWrapper

matplotlib.use('TkAgg')
warnings.simplefilter('ignore', UserWarning)


def image_stream(image_list: List[str], transform_method: Callable[[np.ndarray], Image],
                 intrinsics_mapper: Callable[[str], np.ndarray],
                 use_depth=False, terminate_phase=False,
                 dnn_priors: VidarSelfSupWrapper = None
                 ) -> List[Any]:
    """ image generator ... Multiple type of output is dependent on its purpose

    Parameters
    ----------
    image_list: List[str]
        Sorted filepath to .png or .jpg images
    transform_method: Callable[[np.ndarray], Image]
        RGB transforming method to obtain the downsize image if needed
    intrinsics_mapper: intrinsics_mapper: Callable[[str], np.ndarray]
        Intrinsics mapping function given the corresponding filename.
    use_depth: bool
        Flag for depth-use.
    terminate_phase: bool
        Flag for termination (it's for motion_only BA for interporation.)
    dnn_priors: VidarSelfSupWrapper
        Wrapper for

    Returns
    -------
    List[Any]
        Per-frame processed outputs as follows: (`O` indicates optional)
        - t: int ... serial number of rgb files
        - image: torch.Tensor of the image [3,h,w] that fed into the DROID-SLAM module
        - depth (`O`): torch.Tensor of the depth map [h,w]
        - flatten_intrinsics (`O`): torch.Tensor of the intrinsics parameter, [4,] (JUST Pinhole model is supported)
        - rel_pose: torch.Tensor of the camera transformation [7,], representing [xyz]+[quat_xyzw]
    """

    for t, image_file in enumerate(image_list):
        image = cv2.imread(image_file)  # Raw, full sized image
        h_original, w_original, _ = image.shape  # [H,W,3]
        raw_intrinsics = intrinsics_mapper(image_file)  # Just pinhole model can be handled

        rgb_tensor = rgb_tensor_prev = None
        if not dnn_priors is None:
            # Feeding original size of RGB into the prior models
            rgb_tensor = dnn_priors.transform_image_tensor(image)  # [1,3,h',w']
            # intrinsics
            dnn_intrinsics = dnn_priors.transform_intrinsics(intrinsics_ndarr=raw_intrinsics)  # [1,3,3]
            if t != 0:
                prev_image = None if t == 0 else cv2.imread(image_list[t - 1])
                rgb_tensor_prev = dnn_priors.transform_image_tensor(prev_image)

        # To replicate the original experiment condition, once image is compressed into 384x640, then resampled
        image = np.asarray(transform_method(image))
        h2model, w2model, _ = image.shape
        h1 = int(h2model * np.sqrt((384 * 512) / (h2model * w2model)))
        w1 = int(w2model * np.sqrt((384 * 512) / (h2model * w2model)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        if not terminate_phase and dnn_priors is not None:
            depth = dnn_priors.fwd_depth(rgb_tensor=rgb_tensor, intrinsics=dnn_intrinsics)[0, 0, ...].cpu()  # (h,w)
            depth = F.interpolate(depth[None, None], (h1, w1)).squeeze()
            depth = depth[:h1 - h1 % 8, :w1 - w1 % 8]

        flatten_intrinsics = torch.as_tensor(kmat2pinhole(raw_intrinsics))
        flatten_intrinsics[0::2] *= (w1 / w_original)
        flatten_intrinsics[1::2] *= (h1 / h_original)

        if terminate_phase or (dnn_priors is None) or (not dnn_priors.has_pose):
            rel_pose = torch.as_tensor([0, 0, 0, 0, 0, 0, 1])
        else:
            if t == 0:
                rel_pose = torch.as_tensor([0, 0, 0, 0, 0, 0, 1])
            else:
                rel_pose_mat = dnn_priors.fwd_pose(
                    tgt_frame=rgb_tensor_prev, ctx_frame=rgb_tensor, ctx_is=1
                ).cpu()
                rel_pose = mat_homogeneous2pose_vec(rel_pose_mat).squeeze(0)  # (7,)
                pass

        if use_depth:
            yield t, image[None], depth, flatten_intrinsics, rel_pose

        else:
            if terminate_phase:
                yield t, image[None], flatten_intrinsics
            else:
                yield t, image[None], None, flatten_intrinsics, rel_pose


def eval_single_traj(droid_args: Namespace, dataset: DatasetBase, seq: str = "000150",
                     log_suffix: str = "") -> OrderedDict:
    """Run SG-Init Visual Odometry by setting the configuration based on the dataset, sequence ID, and log information.

    Parameters
    ----------
    droid_args: Namespace
        All configurations for hyperparameter, model path, etc.
    dataset: Inheritance of the DatasetBase.
        (e.g. DatasetDDAD, DatasetUncalibratedFolder, etc.)
    seq: str
        Index for sequence
    log_suffix:
        Provide the suffix for log csv, png etc. naming.

    Returns
    -------
    OrderedDict[str, float]
        Evaluation metrics of visual odometry, such as ATE.
    """
    error = None  # Result of this function. See README to modify this

    # Read directories
    image_list = dataset.get_rgb_lists(seq)[::arg_has_settable(droid_args, 'stride', 1)]
    args_curr = droid_args

    # Load prior via TRI-VIDAR models
    prior_net = None
    if arg_has(args_curr, 'vidar_config'):
        cprint('### [INFO] Read TRI-VIDAR YAML: {}'.format(args_curr.vidar_config), 'yellow', 'on_blue')
        prior_net = VidarSelfSupWrapper(cfg_path=args_curr.vidar_config,
                                        resize=args_curr.resize,
                                        raw_imagesize=args_curr.raw_rgb_size,
                                        ckpt=arg_has_settable(args_curr, 'override_ckpt', None),
                                        depth_key=arg_has_settable(args_curr, 'vidar_depth_type', 'depth'),
                                        pose_key=arg_has_settable(args_curr, 'vidar_pose_type', 'pose',
                                                                  none_str_is_none=True),
                                        )
        use_depth = True
    else:
        use_depth = False

    # If show_gru, monitor the intermediate representations
    if arg_has(args_curr, 'show_gru') and ('{}' in args_curr.show_gru_path):
        args_curr = args_override(args_curr,
                                  key='show_gru_path',
                                  value=args2dict(args_curr)['show_gru_path'].format(seq) + log_suffix,
                                  override_warning=True
                                  )

    # Define intrinsics mapper
    get_intrinsics_from: Callable[[str], np.ndarray] = dataset.map_intrinsics
    if arg_has_settable(droid_args, 'use_pred_intrinsics', False):
        cprint("### [INFO] Use learned Intrinsics ... ", 'red')
        get_intrinsics_from = prior_net.fwd_intrinsics
    if arg_has_settable(droid_args, 'calib', None):
        cprint("### [INFO] Intrinsics is EXTERNALLY override ... {}".format(droid_args.calib), 'red')
        get_intrinsics_from = calib2intrinics_mapper(param=droid_args.calib)

    # Define RGB interpolation protocol
    rgb_transform = def_transforms(dst_hw=tuple([int(elem) for elem in args.resize]))

    # Run droid
    for (t, image, depth, intrinsics, pose) in tqdm(
            image_stream(image_list, transform_method=rgb_transform,
                         intrinsics_mapper=get_intrinsics_from, use_depth=use_depth, dnn_priors=prior_net)):

        if t == 0:
            args_curr.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args_curr)

        droid.track(t, image, depth, intrinsics=intrinsics, pose=pose)

    traj_est = droid.terminate(
        image_stream(image_list, transform_method=rgb_transform, intrinsics_mapper=get_intrinsics_from,
                     use_depth=False, terminate_phase=True))  # (B,7)
    tstamps = dataset.get_timestamps(seq)
    gt_pose = dataset.get_gt_poses(seq)

    # ===== Ground-truth pose & timestamp for evaluation =====
    if (tstamps is not None) and (gt_pose is not None):
        ### <<Quantitative evaluation is missing here!!!>>> ###
        #######################################################
        pass

    return {'ATE': error}


if __name__ == '__main__':
    strt_time = time.time()

    args = sginit_argparse()

    torch.multiprocessing.set_start_method('spawn')

    target_dataset: DatasetBase = EvalDataSet.get_class_by_name("Dataset" + args.dataset)(
        rgb_root=args.src_rgb_path,
        rgb_ext=args.rgb_ext,
        remove_sequences=args.remove_seqs,
        gt_traj_path=args.ref_traj_path,
    )

    if not args.seqs:
        seqs = target_dataset.get_all_sequence()
    else:
        seqs = args.seqs

    csv_name = "no_depth"
    if args.vidar_config is not None:
        csv_name = os.path.basename(args.vidar_config).split('.')[0]  # Use yaml base-filename as prefix
        cprint("### [INFO] Init by:\t{}".format(csv_name), 'yellow')
    else:
        cprint("### [INFO] No init input", 'blue')
    res_lst = []

    # Per sequence evaluation
    for scenario in seqs:
        cprint("### [INFO] Sequence:\t{}".format(scenario), 'yellow')
        ret = eval_single_traj(seq=scenario, dataset=target_dataset, droid_args=args, log_suffix='-' + csv_name)
        if ret['ATE'] is not None:
            res_df = pd.DataFrame(ret, index=[scenario])
            res_lst.append(res_df)
        else:
            cprint('### [WARN] Quantitative evaluation is skipped as NO implementation.', 'yellow', 'on_white')
        pass

    # CLI visualization
    if (res_lst != []):
        if (len(seqs) != 1):
            merged_df = pd.concat(res_lst, axis=0)
            print(merged_df.mean(axis='index'))
            merged_df.to_csv(csv_name + "_num" + str(len(seqs)) + ".csv")
        else:
            print(res_df)

    cprint("Duration\t{}".format(time.time() - strt_time), 'red')
