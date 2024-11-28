# Copyright 2024 Toyota Motor Corporation.  All rights reserved.
# The implementation is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)

import torch
from termcolor import cprint

from droid_utils.config import arg_has
from depth_video import DepthVideo
from droid_net import DroidNet
from factor_graph import FactorGraph


class DroidBackend:
    def __init__(self, net: DroidNet, video: DepthVideo, args):
        self.video = video
        self.update_op = net.update

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = arg_has(args, 'upsample', False)
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms
        self.override_conf = arg_has(args, 'conf_mask', False)
        cprint('### [INFO] <BACKEND> override_conf ... ', 'green') if self.override_conf else True

    @torch.no_grad()
    def __call__(self, steps=12):
        """ main update """

        t = self.video.counter.value
        if not self.video.stereo and not torch.any(self.video.disps_sens):
             self.video.normalize()

        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=16*t, upsample=self.upsample,
                            override_conf_mask=self.override_conf)

        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta,
                                    )

        unique_ids = graph.ii.unique()
        th_skip = 2
        if len(unique_ids) < th_skip:
            print('Skip Global BA ... ')
            return -1  # As BA is impossible to find a proper pair
        else:
            pass

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        self.video.dirty[:t] = True
