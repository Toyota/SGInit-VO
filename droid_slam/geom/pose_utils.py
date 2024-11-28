# Copyright 2024 Toyota Motor Corporation. All rights reserved.
# The implementation is derived from TRI-VIDAR -- 2023 Toyota Research Institute.

import numpy as np
import torch

from vidar.geometry.pose_utils import rot2quat, mat2euler, quat2rot


def mat_homogeneous2pose_vec(mat: torch.Tensor, mode='quat') -> torch.Tensor:
    """
    Convert homogeneous matrices [B,4,4] to pose [B,3+#] (translation + rotation)
    Default style [B,7] is compatible with the argument of a lietorch.SE3()
    """
    _supporting = ('euler', 'quat')
    b = mat.shape[0]
    if mode == 'quat':
        r_vec = rot2quat(mat[:, :3, :3])  # [B,xyzw]
    elif mode == 'euler':
        r_vec = torch.concat([mat2euler(mat[i, :3, :3]) for i in range(b)])  # [B,3] (xyz)
    else:
        NotImplementedError('mode {} is not supported. Set it from {}'.format(mode, _supporting))
    transl = mat[:, :3, 3]  # B,3
    return torch.concat([transl, r_vec], 1)


def to_zero_origin(poses: np.ndarray):
    """ The origin of [b,7] camera trajectory ({t}^T_{w}) is forcibly reset to the origin that t=0, R=0"""
    rot_vidar = quat2rot(torch.tensor(poses[:, 3:]))  # [b,3,3]
    bx3x4 = torch.cat([rot_vidar, torch.tensor(poses[:, :3]).unsqueeze(-1)], dim=2)
    bx4x4 = torch.concat([bx3x4, torch.tensor([0, 0, 0, 1]).repeat([bx3x4.shape[0], 1]).unsqueeze(1)], 1).numpy()

    to_be_origin = bx4x4[0]
    coord_shifter = np.linalg.inv(to_be_origin)

    ret = []
    for i in range(poses.shape[0]):
        ret.append(bx4x4[i] @ coord_shifter)
        pass

    mats = np.array(ret)

    return mat_homogeneous2pose_vec(torch.tensor(mats)).numpy()  # [translation + xyzw]


def invert_pose_ndmat(pose: np.ndarray) -> np.ndarray:
    """
    Inverts a transformation matrix (pose)

    Parameters
    ----------
    pose : np.array
        Input pose [4, 4]

    Returns
    -------
    inv_pose : np.array
        Inverted pose [4, 4]
    """
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = np.transpose(pose[:3, :3])
    inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
    return inv_pose
