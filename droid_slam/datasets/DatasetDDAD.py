# Copyright 2024 Toyota Motor Corporation.  All rights reserved.

import glob
import os
from typing import List

import numpy as np
from termcolor import cprint
import torch

from datasets.DatasetBase import DatasetBase
from geom.pose_utils import invert_pose_ndmat, mat_homogeneous2pose_vec


class DatasetDDAD(DatasetBase):
    """ Parsing class of DDAD data structure, given the preprocessed dataset. """

    def __init__(self, rgb_root: str = '/data/datasets/DDAD/{}/rgb_1216_1936/CAMERA_01',
                 rgb_ext: str = '.jpg', remove_sequences=None, **kwargs):
        super().__init__(rgb_root, rgb_ext, remove_sequences, **kwargs)

    def get_rgb_lists(self, sequence: str = '000000'):
        """ Read all images with sorting """
        rgbs = self.__get_rgb_with_sort(sequence)
        assert rgbs != []
        return rgbs

    def get_all_sequence(self):
        """ Read sequences in the validation split."""
        seqs = [str(i).zfill(6) for i in range(150, 200)]  # validation set
        cprint("### [INFO] FULL DDAD's val sequence start ...", 'blue')
        if self.remove_sequences != []:
            cprint("#### Removed Seq: {}".format(self.remove_sequences), 'red', 'on_white')
            seqs = sorted(list(set(seqs) - set(self.remove_sequences)))
            cprint("#### Start {} sequences ...".format(len(seqs)), 'red', 'on_white')
        return [str(i).zfill(6) for i in range(150, 200)]  # validation set

    def get_timestamps(self, sequence: str = '000000'):
        """ Read timestamps from RGB filenames """
        rgbs = self.__get_rgb_with_sort(sequence)
        tstamps: List[float] = [float(x.split('/')[-1][:-4]) * 10e-8 for x in rgbs]
        return tstamps

    def get_gt_poses(self, seq):
        """
        Load GT Poses from preprocessed DDAD Dataset released from `TRI-ML/efm_datasets.`
        Please refer to the following link: https://github.com/TRI-ML/efm_datasets
        """
        pose_gt_list = sorted(glob.glob(os.path.join(self.gt_traj_path.format(seq), '*.npy')))
        bx4x4 = self.__gt_npys2traj(pose_gt_list)
        traj_gt = mat_homogeneous2pose_vec(torch.tensor(bx4x4)).cpu().numpy()  # (B,7)
        return traj_gt

    def map_intrinsics(self, filename: str) -> np.ndarray:
        """ Mapper function from rgb filename to intrinsics with pinhole model [3,3] """
        rgb_dirname = self.rgb_root.split('/')[-2]
        corresp_intrinsics_dirname = rgb_dirname.replace('rgb_', 'intrinsics_')
        intrinsics_filename = filename.replace(rgb_dirname, corresp_intrinsics_dirname)[:-4] + '.npy'
        return np.load(intrinsics_filename)

    # %%% Internal methods %%%
    def __get_rgb_with_sort(self, seq):
        """ Read RGB filenames."""
        return sorted(glob.glob(os.path.join(self.rgb_root.format(seq), '*' + self.rgb_ext)))

    @staticmethod
    def __gt_npys2traj(pose_list: List[str]) -> np.ndarray:
        """ Read npy poses pose[k]={t=k}^T_{w}, then create GT odometry that repesents {t=0}^T_{t=k} """
        ret_trajectory = [np.eye(4)]
        trajectory_length = len(pose_list)
        all_poses = [np.load(pose_file) for pose_file in pose_list]
        for j in range(1, trajectory_length):
            odom_j = np.matmul(all_poses[0], invert_pose_ndmat(all_poses[j]))
            ret_trajectory.append(odom_j)
        return np.array(ret_trajectory)
