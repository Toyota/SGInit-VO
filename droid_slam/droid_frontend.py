# Copyright 2024 Toyota Motor Corporation.  All rights reserved.
# The implementation is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)

import copy

from termcolor import cprint
import torch

from depth_video import DepthVideo
from droid_utils.config import arg_has
from droid_net import DroidNet, UpdateModule
from factor_graph import FactorGraph


class DroidFrontend:
    def __init__(self, net: DroidNet, video: DepthVideo, args):
        __IMPLE_POSENET_MODE = ('force', 'textureless', 'with_ba', None)

        self.video = video
        self.update_op: UpdateModule = net.update
        conf_mask_ = arg_has(args, 'conf_mask', None)
        cprint('### [INFO] <FRONTEND> override_conf ... ', 'green') if conf_mask_ else True
        self.graph = FactorGraph(video, net.update, max_factors=48, upsample=arg_has(args, 'upsample', False),
                                 override_conf_mask=conf_mask_, args=args)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

        self.posenet_init_first_graph = arg_has(args, key='posenet_init_first_graph', else_return=False)
        cprint('\n### [INFO] PoseNet for first BA graph ... ', 'red') if self.posenet_init_first_graph else True

        self.skip_post_ba_for_first_graph = arg_has(args, key='skip_post_ba_for_first_graph', else_return=False)
        cprint('\n### [INFO] Skip all BA for the first graph ... ', 'red') if self.skip_post_ba_for_first_graph else True

        self.posenet_in_update_ba = arg_has(args, key='posenet_in_update_ba', else_return=None)
        cprint('\n### [INFO] PoseNet MODE: {}'.format(
            self.posenet_in_update_ba), 'red') if self.posenet_in_update_ba is not None else True
        assert self.posenet_in_update_ba in __IMPLE_POSENET_MODE, "Set PoseNet Mode from {}".format(
            __IMPLE_POSENET_MODE)

        self.skip_post_update_ba = arg_has(args, key='skip_post_update_ba', else_return=False)
        cprint('### [INFO] Post Local BA for PoseNet Mode ... ', 'red') if (
                self.posenet_in_update_ba and self.skip_post_update_ba) else True

        # visualization for Dense-BA operators
        self.show_gru = False
        self.diagnose_upd_steps_1 = None
        self.diagnose_upd_steps_2 = None
        if arg_has(args, 'show_gru', False):
            self.show_gru = True
            self.diagnose_upd_steps_1 = arg_has(args, 'show_gru_ba_steps_upd1', [-1,])
            self.diagnose_upd_steps_2 = arg_has(args, 'show_gru_ba_steps_upd2', [self.iters2-1])
            pass


    def __update(self):
        """ add edges, perform update 
        """

        self.t1 += 1  # current == self.t1 - 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(t0=self.t1-5, t1=max(self.t1-self.frontend_window, 0),
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0, 
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        print_debug = False
        print("print_debug") if print_debug else True
        if self.posenet_in_update_ba in ('force', 'with_ba'):
            # To follow the index-like behavior
            update_index = torch.tensor([self.t1 - 1]).to(torch.int32).to(self.video.tstamp.device)
            # Estimate the current pose by multiplying the instantaneous ego-motion
            # by the pose of the previous keyframe (self.t1-2)
            pose_t_minus_1 = self.video.compose_idx2pose_from_instpose(
                indices=update_index, base_keyframe_id=self.t1 - 2)
            self.video.poses[self.t1 - 1] = pose_t_minus_1[0]
            self.video.disps[self.t1 - 1] = self.video.disps_sens[self.t1 - 1]
            if self.posenet_in_update_ba == 'with_ba':
                # print('Go BA')
                for itr in range(self.iters1):
                    self.graph.update(None, None, use_inactive=True, itr=itr)
        else:
            # Normal worker mode as DROID-SLAM
            for itr in range(self.iters1):
                self.graph.update(None, None, use_inactive=True, itr=itr,
                                  diagnose_step=self.diagnose_upd_steps_1)

        # set initial pose for next frame
        # comparing `current-2` and `current-1` to decide whether keep 
        d = self.video.distance(ii=[self.t1-3], jj=[self.t1-2], beta=self.beta, bidirectional=True)

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            if self.posenet_in_update_ba is not None:
                # As once poses are optimized via outside the DROID, chose whether rerun optimization inside or not
                if not self.skip_post_update_ba:
                    for itr in range(self.iters2):
                        self.graph.update(None, None, use_inactive=True)
                else:
                    # Apply no BA anymore inside Droid
                    pass
                pass
            else:
                for itr in range(self.iters2):
                    self.graph.update(None, None, use_inactive=True, itr=itr,
                                  diagnose_step=self.diagnose_upd_steps_2)

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True
        self.video.tstamp_prev[:self.t1] = copy.deepcopy(self.video.tstamp[:self.t1])

    def __initialize(self):
        """ initialize the SLAM system
        i=0,1,2,...,"self.t1-1" elements of the `self.video.poses` and `self.video.disps` are updated

        """

        self.t0 = 0
        self.t1 = self.video.counter.value  # same with the self.warmup

        # 1st and LAST time to call add_neighborhood_factors(): t0=0, t1=8. meshgrid is applied between those indices
        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        print_debug = False
        print("== STRT GRPH ==") if print_debug else True
        init_indices = torch.arange(start=self.t0, end=self.t1).to(self.video.poses.device)
        if self.posenet_init_first_graph:
            self.video.poses[init_indices] = self.video.compose_idx2pose_from_instpose(indices=init_indices)
            self.video.disps[init_indices] = self.video.disps_sens[init_indices]
        else:
            for itr in range(8):
                self.graph.update(1, use_inactive=True, itr=itr)

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        if self.skip_post_ba_for_first_graph:
            # Forcebly ignore the BA step
            pass
        else:
            for itr in range(8):
                self.graph.update(1, use_inactive=True, itr=itr)

        print("-- Fin  __initialize() --") if print_debug else True

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)
        self.video.tstamp_prev[init_indices] = copy.deepcopy(self.video.tstamp[init_indices])

    def __call__(self):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()
