# Copyright 2024 Toyota Motor Corporation.  All rights reserved.

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class DatasetBase(ABC):
    """Base dataset class, with functionalities shared across all subclasses."""

    def __init__(self, rgb_root: str, rgb_ext: str, remove_sequences: List[str], gt_traj_path: str):
        """
        Parameters
        ----------
        rgb_root : str
            Path to an RGB source folder.
        rgb_ext : str
            Extension for an image file.
        remove_sequences : List[str]
            Ignore the sequence path for evaluation.
        gt_traj_path :
            Path to an RGB source folder contains GT Trajectory.
        """
        super().__init__()

        self.rgb_root = rgb_root
        self.rgb_ext = rgb_ext
        self.remove_sequences = remove_sequences
        self.gt_traj_path = gt_traj_path

    @abstractmethod
    def get_all_sequence(self) -> List[str]:
        """ See each dataset implementation """
        pass

    @abstractmethod
    def get_rgb_lists(self, seq: str) -> List[str]:
        """ See each dataset implementation """
        pass

    @abstractmethod
    def get_timestamps(self, seq: str) -> List[str]:
        """ See each dataset implementation """
        pass

    @abstractmethod
    def get_gt_poses(self, seq: str) -> np.ndarray:
        """ See each dataset implementation """
        pass

    def map_intrinsics(self, filename: str) -> np.ndarray:
        """ See each dataset implementation. No mandatory for implementation"""
        pass
