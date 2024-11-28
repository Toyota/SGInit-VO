# Copyright 2024 Toyota Motor Corporation. All rights reserved.
# The implementation is derived from TRI-VIDAR -- 2023 Toyota Research Institute.

import copy
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms

from vidar.utils.config import read_config, Config, update_from_kwargs
from vidar.utils.setup import setup_arch

# Consistently use the same interpolation for both model input and droid-slam model
INTERPOLATION = transforms.InterpolationMode.LANCZOS


class VidarSelfSupWrapper:
    def __init__(self, cfg_path: str, depth_key='depth', ckpt: str = None, verbose=False, intrinsics_key='intrinsics',
                 pose_key='pose', device='cuda:0', resize: Tuple[int, int] = None,
                 raw_imagesize: Tuple[int, int] = None):
        """
        TRI-ML/VIDAR compatible DepthNet, PoseNet, and IntrinsicsNet wrapper

        Parameters
        ----------
        cfg_path: str
            Path to a YAML file that provides the learning configuration (only `arch:` and it's below are imported).
        depth_key: str
            Key to identify the depth predictor under `arch/networks`
        ckpt: str
            (Default:None) Override the checkpoint path to load models, defined at `arch/model/checkpoint`
        verbose: bool
            Print the load model inf oor not
        intrinsics_key: str
            Key to identify the intrinsics predictor under `arch/networks`
        pose_key: str
            Key to identify the ego-motion predictor under `arch/networks`
        device: str
            Specify the Cuda device that is used for default.
        resize: Tuple[int,int]
            RGB's HW to feed the model.
        raw_imagesize: Tuple[int,int]
            Raw RGB size loaded from your local storage.
        """
        super().__init__()
        cfg: Config = read_config(cfg_path)
        if ckpt is not None:
            # override the ckpt to be load
            cfg.arch.model = update_from_kwargs(cfg.arch.model, **{'checkpoint': ckpt})

        self.has_depth = False
        self.has_pose = False
        self.has_intrinsics = False

        self.__model = setup_arch(cfg.arch, verbose=verbose).to(device).eval()
        self.depth_module_key = depth_key
        self.pose_module_key = pose_key
        self.intrinsics_module_key = intrinsics_key
        self.default_dev = device
        self.hw_raw = [int(elem) for elem in raw_imagesize]
        self.hw_finally = [int(elem) for elem in resize]
        self.__resize_flag = False if resize is None else True
        self.__camera_model = None

        # mount models onto GPU
        network_keys = self.__model.networks.keys()
        self.__transform = self.__define_transforms(hw=resize)
        if self.depth_module_key is not None and (self.depth_module_key in network_keys):
            self.__depth_net = self.__model.networks[self.depth_module_key]
            self.has_depth = True
        if self.pose_module_key is not None and (self.pose_module_key in network_keys):
            self.__pose_net = self.__model.networks[self.pose_module_key]
            self.has_pose = True
        if self.intrinsics_module_key is not None and (self.intrinsics_module_key in network_keys):
            self.__intrinsics_net = self.__model.networks[self.intrinsics_module_key]
            self.__camera_model = cfg.arch.networks.intrinsics.has('camera_model', 'pinhole')
            assert self.__camera_model == 'pinhole', \
                'Only camera_model == pinhole is supported. Now `{}` is applied.'.format(self.__camera_model)
            self.has_intrinsics = True

    def __define_transforms(self, hw: Tuple[int, int] = None, interpolation=INTERPOLATION):
        """Define the Transformation from raw RGB to resize the image to feed into the models."""
        transform = transforms.Compose([transforms.Resize(hw, interpolation=interpolation),
                                        transforms.ToTensor()])
        return transform

    def transform_image_tensor(self, image, tensor_type='torch.FloatTensor', revert_rgb=True):
        """Casts an image read by cv2 to a torch.Tensor (revert_rgb should be TRUE for PIL compatiblity) [1,3,h,w]"""
        inpt_ndarray = image[:, :, ::-1].copy() if revert_rgb else image
        if not self.__resize_flag:
            return self.__transform(inpt_ndarray).type(tensor_type)
        else:
            pil_image = transforms.ToPILImage()(inpt_ndarray)
            return self.__transform(pil_image).type(tensor_type).unsqueeze(0)

    def transform_intrinsics(self, intrinsics_ndarr: np.ndarray, tensor_type='torch.FloatTensor',
                             hw: Tuple[int, int] = None) -> torch.Tensor:
        """Given the K matrix corresponding to RAW image, adopt it to a resize image size and return [1,3,3] """
        raw_hw: Tuple[int, int] = hw if hw is not None else self.hw_raw
        assert raw_hw is not None, 'Set argument `hw` or `raw_imagesize` of this class instance'
        if not self.__resize_flag:
            intrinsics = intrinsics_ndarr
        else:
            h0, w0 = raw_hw
            h1, w1 = self.hw_finally
            intrinsics = copy.copy(intrinsics_ndarr)
            intrinsics[0, :] *= (w1 / w0)
            intrinsics[1, :] *= (h1 / h0)
        return torch.tensor(intrinsics).type(tensor_type).to(self.default_dev).unsqueeze(0)

    def fwd_depth(self, rgb_tensor: torch.Tensor, intrinsics: torch.Tensor = None, device: str = None) -> torch.Tensor:
        """
        Forward depth network depending on the implemented depth estimation type (depth, perceiver, etc.)

        Parameters
        ----------
        rgb_tensor: torch.Tensor
            [b,3,h,w] of RGB WITHOUT NORMALIZATION
        intrinsics: torch.Tensor
            [b,3,3]
        device: str
            Device to launch if override is needed
        Returns
        -------
        torch.Tensor
            [b,1,h,w] of depth tensor output
        """
        dev = device if device is not None else self.default_dev
        with torch.no_grad():
            depth_out = self.__depth_net(rgb=rgb_tensor.to(dev), intrinsics=intrinsics.to(dev))
            if self.depth_module_key == 'depth':
                pred_depth = depth_out['depths'][0]
            elif self.depth_module_key == 'perceiver':
                pred_depth = depth_out
            else:
                raise NotImplementedError()
            return pred_depth

    def fwd_pose(self, tgt_frame: torch.Tensor, ctx_frame: torch.Tensor,
                 ctx_is: int, device: str = None) -> torch.Tensor:
        """
        Forward ego-motion estimation network

        Parameters
        ----------
        tgt_frame: torch.Tensor
            [b,3,h,w]
        ctx_frame: torch.Tensor
            [b,3,h,w]
        ctx_is: int
            Specify the temporal position of the context frame from the target frame, like -1 or 1
            (-1 ... just previous frame from tgt, 1 is just forward, ...etc.)
        device: str
            override cuda device (default is None)

        Returns
        -------
        torch.Tensor
            Ego-motion transformation of [b,4,4], or None
        """
        dev = device if device is not None else self.default_dev
        with torch.no_grad():
            if self.pose_module_key == 'pose':
                bx4x4 = self.__pose_net([tgt_frame.to(dev), ctx_frame.to(dev)],
                                        invert=(0 < ctx_is))['transformation']
            elif self.pose_module_key == None:
                bx4x4 = None
            else:
                raise NotImplementedError()
            return bx4x4

    def fwd_intrinsics(self, filename: str = '') -> np.ndarray:
        """
        Forward ego-motion intrinsics parameters.
        See `dataset.DatasetBase.map_intrinsics()` or
            `dataset.utils.calib2intrinics_mapper()`, to fit this method's I/O of them

        Parameters
        ----------
        filename: str
            RGB filename that provides corresponding intrinsics output (NOT used)
        Returns
        -------
        np.ndarray
            K matrix with [3,3]
        """
        dummy_rgb = torch.randn((1, 3, self.hw_finally[0], self.hw_finally[1])).to(self.default_dev)
        with torch.no_grad():
            fx, fy, cx, cy = self.__intrinsics_net(
                dummy_rgb).cpu().numpy()[0]  # flatten intrinsics that shapes (4,)
            if self.__camera_model == 'pinhole':
                return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
            else:
                raise NotImplementedError()
