# Copyright 2024 Toyota Motor Corporation.  All rights reserved.

from typing import Tuple

import numpy as np
import cv2

import flow_vis
import torch


def viz_high_intensity(image_like_1d: np.ndarray, rgb_img: np.ndarray) -> np.ndarray:
    """
    Visualize high-intensity area given the 1D image and corresponding RGB.

    Parameters
    ----------
    image_like_1d : np.ndarray
        Any type 1D tensor to create heatmap shapes [H, W]
    rgb_img : np.ndarray
        uint8 Image tensor shapes [H,W,3]

    Returns
    -------
    np.ndarray
        Alpha-blended image to visualize the heatmap.

    """
    image_like_1d = cv2.resize(image_like_1d, rgb_img.shape[:2][::-1])
    normalized_gray = image_like_1d / np.max(image_like_1d)
    alpha_channel = np.expand_dims(normalized_gray, axis=2)
    overlay = (alpha_channel * rgb_img).astype(np.uint8)
    return overlay


def viz_flow_resizing(flow: torch.Tensor, hw: Tuple[int] = None) -> np.ndarray:
    """

    Parameters
    ----------
    flow: torch.Tensor
        Flow tensor that shapes []
    hw: Tuple[int]
        Resize image
        **NOTE** This is just for visualization on RGB image, so not support the flow correctness

    Returns
    -------
    np.ndarray
        Optical-flow that can be visualized.
    """
    flow = flow_vis.flow_to_color(flow.cpu().numpy())
    if hw is not None:
        flow = cv2.resize(flow, (hw[1], hw[0]))
    return flow


def viz_weight_on_image(img_ndarray: np.ndarray, w_ndarray: np.ndarray,
                        heat_lower_thres: float = 0.15, ctype=cv2.COLORMAP_JET, heat_blend: float = 0.5) -> np.ndarray:
    """
    Visualize high-intensity area given the 1D image and corresponding RGB.

    Parameters
    ----------
    img_ndarray : np.ndarray
        uint8 Image tensor shapes [H,W,3]
    w_ndarray : np.ndarray
        Any type 1D tensor to create heatmap shapes [H, W]
    heat_lower_thres: float
        Lower boundary visualize the heatmap
        If the intensity below this, original RGB is depicted
    ctype :
        cv2 color type (see: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html)
    heat_blend : float
        Ratio to overlay heatmap to fuse original RGB

    Returns
    -------
    np.ndarray
        uint8 Image tensor shapes [H, W, 3] that represent the original image overlayed by the heatmap
    """
    gray_arr = cv2.resize(w_ndarray, img_ndarray.shape[:2][::-1])
    gray_arr = gray_arr / np.max(gray_arr)
    heatmap_rgb = cv2.applyColorMap((gray_arr * 255).astype(np.uint8), ctype)  # [h,w,3]

    # Apply the mask given the threshold.
    mask = gray_arr <= heat_lower_thres
    rgb_highval_replaced = np.array(
        [np.where(mask, img_ndarray[:, :, i], heatmap_rgb[:, :, i]) for i in range(3)]).transpose(1, 2, 0)

    blended_img = cv2.addWeighted(rgb_highval_replaced, 0.5, img_ndarray, heat_blend, 2.2)

    return blended_img
