# Copyright 2024 Toyota Motor Corporation.  All rights reserved.

import glob
import os

from datasets.DatasetBase import DatasetBase


class DatasetUncalibratedFolder(DatasetBase):
    """
    RGB dataset class for demonstration code that assumes:
    - rgb_root: MUST contain the One '{}' to search the sequences.
    - NO ground-truth intrinsic parameters are available, and they are shared across all sequences.
    - self.rgb_root.format(seq) provides the one-level-higher folder name to reach raw RGB images.
    """

    def __init__(self, rgb_root: str = '',
                 rgb_ext: str = '.jpg', remove_sequences=None, **kwargs):
        super().__init__(rgb_root, rgb_ext, remove_sequences, **kwargs)

    def get_rgb_lists(self, sequence: str):
        """ Read all images with sorting. """
        rgbs = self.__get_rgb_with_sort(sequence)
        assert rgbs != []
        return rgbs

    def get_all_sequence(self):
        """Read all folder names fitting to the template."""
        return self.get_dir_list_from_template(self.rgb_root)

    def get_timestamps(self, sequence: str):
        """ Implement functions depending on the dataset structure """
        return None

    def get_gt_poses(self, seq):
        """ Implement functions depending on the dataset structure """
        return None

    @staticmethod
    def get_dir_list_from_template(base_path: str) -> list:
        """ given the path-like such as `hoge/huga/{}/piyo/puka`, obtain all dirname under hoge/huga/ """
        base_path_parts = base_path.split("{}")

        if len(base_path_parts) != 2:
            raise ValueError("Template must contain exactly one '{}' placeholder.")

        base_path = base_path_parts[0]

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"The directory '{base_path}' does not exist.")

        dir_list = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

        return dir_list

    def __get_rgb_with_sort(self, seq):
        """ Read RGB filenames."""
        return sorted(glob.glob(os.path.join(self.rgb_root.format(seq), '*' + self.rgb_ext)))
