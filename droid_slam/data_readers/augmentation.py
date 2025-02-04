# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 
# The implementation is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Consistently use the same interpolation for both model input and droid-slam model
INTERPOLATION = transforms.InterpolationMode.LANCZOS


def def_transforms(dst_hw=(384, 640), interpolation=INTERPOLATION) -> transforms.Compose:
    """ RGB transformation to map the cv2-load image to PIL stuff """
    rgb_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(dst_hw, interpolation=interpolation)])
    return rgb_transform


class RGBDAugmentor:
    """ perform augmentation on RGB-D video """

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.4/3.14),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor()])

        self.max_scale = 0.25

    def spatial_transform(self, images, depths, poses, intrinsics):
        """ cropping and resizing """
        ht, wd = images.shape[2:]

        max_scale = self.max_scale
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd)))

        scale = 2 ** np.random.uniform(min_scale, max_scale)
        intrinsics = scale * intrinsics
        depths = depths.unsqueeze(dim=1)

        images = F.interpolate(images, scale_factor=scale, mode='bilinear', 
            align_corners=False, recompute_scale_factor=True)
        
        depths = F.interpolate(depths, scale_factor=scale, recompute_scale_factor=True)

        # always perform center crop (TODO: try non-center crops)
        y0 = (images.shape[2] - self.crop_size[0]) // 2
        x0 = (images.shape[3] - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        images = images[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depths = depths[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        depths = depths.squeeze(dim=1)
        return images, poses, depths, intrinsics

    def color_transform(self, images):
        """ color jittering """
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2,1,0]] / 255.0)
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(self, images, poses, depths, intrinsics):
        images = self.color_transform(images)
        return self.spatial_transform(images, depths, poses, intrinsics)
