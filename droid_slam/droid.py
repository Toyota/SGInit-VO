# Copyright 2024 Toyota Motor Corporation.  All rights reserved.
# The implementation is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)

from collections import OrderedDict
import os

from termcolor import cprint
import torch
from torch.multiprocessing import Process

from depth_video import DepthVideo
from droid_backend import DroidBackend
from droid_frontend import DroidFrontend
from droid_net import DroidNet
from droid_utils.config import arg_has_false as arg_has
from geom.pose_utils import to_zero_origin
from motion_filter import MotionFilter
from trajectory_filler import PoseTrajectoryFiller


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.args = args
        self.load_weights(args.weights)
        self.disable_vis = args.disable_vis
        self.t_zero_is_origin = arg_has(args, 'zero_origin')
        if self.t_zero_is_origin:
            cprint('### [INFO] Pose|t=0 is set as identity', 'green')

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo, args=args)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video, args=self.args)


    def load_weights(self, weights):
        """ load trained model weights """

        print(weights) if arg_has(self.args, 'show_cli') else True
        self.net = DroidNet()

        if os.path.basename(weights) == 'droid.pth':
            # download: https://github.com/princeton-vl/DROID-SLAM?tab=readme-ov-file#demos
            loaded_state_dict = torch.load(weights)
            replace_key = "module."
            pass
        elif os.path.basename(weights) == 'r3d3_finetuned.ckpt':
            # download: https://github.com/SysCV/r3d3?tab=readme-ov-file#vkitti2-finetuned-feature-matching
            cprint("\n### [INFO] R3D3 weight:{}".format(weights), 'yellow')
            loaded_state_dict = torch.load(weights)['state_dict']
            replace_key = 'module.networks.droid_net.'
            pass
        else:
            raise NotImplementedError('Invalid droid weight:{}'.format(weights))
        
        state_dict = OrderedDict([
            (k.replace(replace_key, ""), v) for (k, v) in loaded_state_dict.items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None, pose=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics, pose)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            # self.backend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """
        if not self.frontend.is_initialized:
            cprint('### [WARN] Frontend NEVER called, so just global-BA is executed', 'yellow')

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32) if arg_has(self.args, 'show_cli') else True
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32) if arg_has(self.args, 'show_cli') else True
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        odom_putput = camera_trajectory.inv().data.cpu().numpy()

        if self.t_zero_is_origin:
            forced2zero_origin = to_zero_origin(odom_putput)
            return forced2zero_origin
        else:
            return odom_putput

