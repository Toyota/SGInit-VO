# Copyright 2024 Toyota Motor Corporation.  All rights reserved.
# The implementation is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)

from argparse import Namespace
from typing import List

import lietorch
import numpy as np
import torch
from torch.multiprocessing import Value

import droid_backends
from droid_net import cvx_upsample
from droid_utils.config import arg_has
import geom.projective_ops as pops
from geom.pose_utils import mat_homogeneous2pose_vec

class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0", args: Namespace = None):
                
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.tstamp_prev = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.poses_prior = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.ts2instant_pose = torch.zeros(8 * buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2  # False if the SLAM with -RGB and -Depth

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")

        # initialize pose priors as well
        # self.poses_prior is odometry composed of the poses and poses_instantaneous: poses_prior[idx] = {idx}^T_{0}
        # self.poses_instantaneous is per-frame relative poses: poses_instantaneous[idx] = {idx}^T_{idx-1}
        self.poses_prior[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
        self.ts2instant_pose[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")

        # for logging
        if type(args) is Namespace:
            self.pkl_suffix = arg_has(args, "pkl_suffix", None)

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        else:
            pass  # Come here to interpolate the trajectory

        # self.dirty[index] = True
        self.tstamp[index] = item[0]  # Only appended: The value is appended and increase gradually (NEVER decrease)

        # The following variables are dynamically shifted because they define the area for dense BA ...
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

        if len(item) > 9:
            if item[9] is not None:
                self.poses_prior[index] = item[9]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index],
                self.poses_prior[index],
                )

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.dirty[:self.counter.value] = True

    def reproject(self, ii, jj, initialization_by_relpose: bool = False):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])
        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii: torch.Tensor = None, jj: torch.Tensor = None, beta=0.3, bidirectional=True):
        """ frame distance metric

        Parameters
        ----------
        ii: torch.Tensor(n)
            Indices for running BA (input1)
            Generally `younger` are used for ID (e.g. torch.tensor([24]) )
        jj: torch.Tensor(n)
            Indices for running BA (input2)
            For the final BA, full meshgrid is used for exhaustive matching thus None is applied
        beta
        bidirectional

        Returns
        -------

        """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value # Get keyframes
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

            self.disps.clamp_(min=0.001)

    def compose_idx2pose_from_instpose(self, indices: torch.Tensor, base_keyframe_id: int = None) -> torch.Tensor:
        """
        Convert relative poses to global coordinated poses with [N,7]
         by keyframe with `base_keyframe_id` as its base coordinate.

        Parameters
        ----------
        indices: torch.Tensor
            [N,] shaping tensors of keyframe indices (such ii or jj implemented in FactorGraph() class): 0,1,2,3,...
        base_keyframe_id: int
            Integer of the keyframe index: 0,1,2,3, ...

        Returns
        -------
        torch.Tensor
            [N,7] tensors of the SE(3)
        """
        # Get unique indices
        unique_timestamps = indices.unique()
        unique_ts_lst = sorted(list(unique_timestamps.cpu().numpy()))  # len(unique)
        assert self._check_continuous(unique_ts_lst), "Jumping unique ID is NOT supporting!! Check track ID{}".format(
            unique_ts_lst)

        # Define base pose
        orig_coord_id = unique_ts_lst[0]
        if base_keyframe_id is not None:
            if base_keyframe_id <= unique_ts_lst[0]:
                orig_coord_id = base_keyframe_id
            else:
                print('[WARN] base_keyframe_id (={}) is '
                      'ignored as it is larger than indices.min() (={})'.format(base_keyframe_id, unique_ts_lst[0]))
            pass

        orig_pose = lietorch.SE3(self.poses[orig_coord_id].unsqueeze(0)).matrix()  # {tau}^T_[ts=0}, 4x4
        # Tile the keyframe indices
        orig_with_uniq_keyframes: list = list(np.arange(orig_coord_id, np.array(unique_ts_lst).min())) + unique_ts_lst
        orig_with_ts = self.tstamp[orig_with_uniq_keyframes]

        # Tile the pose from the world to a specific timestamp
        transforms_from_wlrd: List[torch.Tensor] = []

        prev_ts = None
        for loop, keyframe_id in enumerate(orig_with_uniq_keyframes):
            curr_ts = self.tstamp[keyframe_id]
            if loop == 0:
                transforms_from_wlrd.append(orig_pose)
            else:
                prev_pose_w = transforms_from_wlrd[loop - 1]
                for iter in range(int(prev_ts.cpu().numpy()), int(curr_ts.cpu().numpy())):
                    timestamp = torch.tensor([iter + 1]).to(curr_ts.device)
                    curr_pose_prev = lietorch.SE3(self.ts2instant_pose[timestamp]).matrix()
                    curr_pose_w = curr_pose_prev @ prev_pose_w

                    if timestamp == curr_ts:
                        transforms_from_wlrd.append(curr_pose_w)
                        pass
                    prev_pose_w = curr_pose_w
            # update timestamp
            prev_ts = curr_ts
            pass

        unique_poses = torch.cat(transforms_from_wlrd, dim=0)  # [len(unique),4,4]
        timestamps_from_keyframe_id = torch.index_select(self.tstamp, 0, indices.to(torch.int32))

        # Tile all poses that base coordinate is "world" of the odometry
        poses_from_wlrd = self._get_values(input_index=timestamps_from_keyframe_id,  # Timestamp indices
                                           keys=orig_with_ts,  # Timestamp keys including origin and unique timestamps
                                           values=unique_poses,  # poses corresponding to the unique timestamps
                                           )

        return mat_homogeneous2pose_vec(poses_from_wlrd)

    @staticmethod
    def _get_values(input_index: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """Given the integer keys and corresponding values, sample the series of  values based on the `input_index`.
        =======================
        # (e.g.)
        # input_index = torch.tensor([22, 23, 20])
        # keys = torch.tensor([20, 21, 22, 23, 24])
        # values1 = torch.tensor([-8, -60, -3, -2, -4])
        # values2 = torch.tensor([[-8, -3],[-60, 90],[-3, 87],[-2, 90],[-4,1]])
        # values3 = torch.randn(5, 4, 4)
        ====================

        Parameters
        ----------
        input_index: torch.Tensor
            [L,] tensor for what the user wants to replace them by the value
        keys: torch.Tensor
            [D, ] tensor for indices
        values: torch.Tensor
            [D, ?] tensor for tensors (intended for the pose tensors of [D,4,4])
        Returns
        -------
        torch.Tensor
            [L, ?] tensor the values of which is sampled from values
        """
        # Create the result placeholder
        result_shape = (input_index.shape[0],) + values.shape[1:]
        result = torch.empty(result_shape, dtype=values.dtype).to(values.device)

        # Obtains the result based on the input_index
        for i, index in enumerate(input_index):
            key_index = (keys == index).nonzero(as_tuple=True)[0]
            result[i] = values[key_index]

        return result

    @staticmethod
    def _check_continuous(input_list: List[int]) -> bool:
        """ Sanity-check the indices whether .unique() has skip IDs
        (e.g.)
            if input_list == [5, 0, 3, 4, 1] -> False
            if input_list == [5, 0, 3, 4, 2, 1] -> True
        Returns
        -------
        bool
            Return True if the integer in input_list is NOT duplicated nor skip
        """
        sorted_list = sorted(input_list)
        for i in range(len(sorted_list) - 1):
            if sorted_list[i + 1] - sorted_list[i] >= 2:
                return False
        return True