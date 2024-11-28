# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 
# The implementation is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
# Just refactoring is applied.

import lietorch
from termcolor import cprint
import torch

from depth_video import DepthVideo
from droid_net import DroidNet
import geom.projective_ops as pops
from modules.corr import CorrBlock


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net: DroidNet, video: DepthVideo, thresh=2.5, device="cuda:0", **kwargs):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        # For ablation study
        cprint('--filter_thres ablation: {}'.format(thresh), 'blue', 'on_cyan') if thresh != 2.0 else True
        self.device = device

        self.skip_cnt_to_video_append = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

        # Add variables for temporal-automask
        self.prev_normed_image = None
        self.automask_thres = 0.02  # if abs(a-b)/(a+b) gets smaller than this, recognized as 'No Difference'  # FIX IS RECOMMENDED
        self.textureless_thres = 0.00  # if remained area after automask ratio gets below this, decided as Textureless  # 0.05 is GREAT
        self.automask_flatten = 'and'
        defined_flattens = ('', 'or', 'and')
        assert self.automask_flatten in defined_flattens, \
            "Automask flatten method {} is not defined".format(self.automask_flatten)

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None, pose=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        ### Always append relative ego-motion if available: {idx}^T_{idx-1}
        if pose is not None:
            self.video.ts2instant_pose[tstamp] = pose  # 0,1,2,... len(seq)-1

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0], pose)

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]  # [1,1,ht,wd,2]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)  # [1,1,ch,ht,wd]

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.skip_cnt_to_video_append = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0], pose)

            else:
                self.skip_cnt_to_video_append += 1

        self.prev_normed_image = inputs.squeeze(1)  # Reshape to [B,3,H,W]

    def _get_automask(self, rgb1: torch.Tensor, rgb2: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        rgb1: torch.Tensor
            Normalized image tensor with [B,3,H,W]. Generally `current` frame is set here

        rgb2: torch.Tensor
            `Previous` tensor with  [B,3,H,W]

        Returns
        -------
        torch.Tensor
            Binary tensor with the [B,1,H,W]

        """

        if rgb2 is None:
            return torch.ones_like(rgb1)[:, 0, :, :].unsqueeze(1)  # All area is non mask-out

        mask: torch.Tensor = (rgb1 - rgb2).abs() / (rgb1 + rgb2 + 10e-7)
        mask[mask >= self.automask_thres] = 1.  # Non-masked area
        mask[mask < self.automask_thres] = 0.  # masked-area

        if self.automask_flatten == 'or':
            mask[mask != 0] = 1.
            mask = torch.sum(mask, dim=1).unsqueeze(0) / 3.  # As the sum of RGB
        elif self.automask_flatten == 'and':
            new_ = mask[:, 0] * mask[:, 1] * mask[:, 2]
            mask = new_.unsqueeze(0)
        elif self.automask_flatten == '':  # output 3-channel mask
            pass
        else:
            raise NotImplementedError()
        return mask

    def get_automask_removing_ratio(self, input1: torch.Tensor, input2: torch.Tensor):
        mask = self._get_automask(input1, input2)
        _, _, *hw = mask.shape
        return mask.sum() / (hw[0] * hw[1])