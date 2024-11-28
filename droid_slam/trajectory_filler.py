# Copyright 2024 Toyota Motor Corporation.  All rights reserved.
# The implementation is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)

from argparse import Namespace
from typing import List

from termcolor import cprint
import torch
import lietorch

from lietorch import SE3
from factor_graph import FactorGraph
from depth_video import DepthVideo
from droid_utils.config import arg_has

class PoseTrajectoryFiller:
    """ This class is used to fill in non-keyframe poses """

    def __init__(self, net, video: DepthVideo, device="cuda:0", args:Namespace = None):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update
        self.confmask_flag = arg_has(args, 'conf_mask', None)
        cprint('### [INFO] <TrajFilter> override_conf ... ', 'green') if self.confmask_flag else True

        self.count = 0
        self.video = video
        self.device = device

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image)

    def __fill(self, tstamps: List[int], images: List[torch.Tensor], intrinsics: List[torch.Tensor]) -> List[SE3]:
        """ fill operator """

        tt = torch.as_tensor(tstamps, device="cuda")  # the latest sixteen serial number corresponding to the filename
        images = torch.stack(images, 0)
        intrinsics = torch.stack(intrinsics, 0)
        inputs = images[:,:,[2,1,0]].to(self.device) / 255.0
        
        ### linear pose interpolation ###
        N = self.video.counter.value  # tracked keyframes, decided by the ego-motion distance
        M = len(tstamps)  # (0~15, 16~31, ..., )

        ts = self.video.tstamp[:N]  # actual timestamps of tracked keyframes: [0.,  1.,  2.,  3.,  4.,  ..., 89.]
        Ps = SE3(self.video.poses[:N])

        t0 = torch.as_tensor([ts[ts<=t].shape[0] - 1 for t in tstamps])  # lower-bound INDEX for interpolation
        t1 = torch.where(t0<N-1, t0+1, t0)  # upper-bound  INDEX for interpolation

        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()

        v = dP.log() / dt.unsqueeze(-1)
        w = v * (tt - ts[t0]).unsqueeze(-1)
        Gs = SE3.exp(w) * Ps[t0]

        # extract features (no need for context features)
        # [N, 1, ch, ch, h, w]. N is almost always 16 and the image is original size
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        self.video.counter.value += M

        # In this implementation, once registration of all variables into the item list self.video[] is required.
        # Thus, the void area of them (id= self.video.counter.value or later) is temporally borrowed to run BA.
        # Then cleared by `self.video.counter.value -= M`
        self.video[N:N+M] = (tt, images[:,0], Gs.data, 1, None, intrinsics / 8.0, fmap)

        graph = FactorGraph(self.video, self.update, override_conf_mask=self.confmask_flag)
        graph.add_factors(t0.cuda(), torch.arange(N, N+M).cuda())
        graph.add_factors(t1.cuda(), torch.arange(N, N+M).cuda())

        for itr in range(6):
            # Since this process reuse the ``ii'' which is registered while frontend process,
            # feeding ii to torch.index_select() is still available for tsstamp or conf_mask reusing
            graph.update(N, N+M, motion_only=True)
    
        Gs = SE3(self.video.poses[N:N+M].clone())
        self.video.counter.value -= M  # Memory cleared

        return [ Gs ]  # Sixteen or fewer poses which are linearly interpolated and optimized via photometric BA

    @torch.no_grad()
    def __call__(self, image_stream):
        """ fill in poses of non-keyframe images """

        # store all camera poses
        pose_list = []

        tstamps = []
        images = []
        intrinsics = []
        
        for (tstamp, image, intrinsic) in image_stream:
            tstamps.append(tstamp)
            images.append(image)
            intrinsics.append(intrinsic)

            if len(tstamps) == 16:
                # Set the latest 16 frames for interpolation
                pose_list += self.__fill(tstamps, images, intrinsics)
                tstamps, images, intrinsics = [], [], []

        if len(tstamps) > 0:
            # Finally, full interpolation will go
            pose_list += self.__fill(tstamps, images, intrinsics)

        # stitch pose segments together
        return lietorch.cat(pose_list, 0)  # camera_trajectory.matrix() provides [B,4,4]

