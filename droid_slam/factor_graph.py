# Copyright 2024 Toyota Motor Corporation.  All rights reserved.
# The implementation is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)

from argparse import Namespace
import os.path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

import geom.projective_ops as pops
from depth_video import DepthVideo
from droid_net import UpdateModule
from droid_utils.viz import viz_weight_on_image, viz_flow_resizing, viz_high_intensity
from droid_utils.config import arg_has
from modules.corr import CorrBlock, AltCorrBlock


class FactorGraph:
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1, upsample=False,
                 override_conf_mask: bool = None, args: Namespace = None):
        self.video: DepthVideo = video
        self.update_op: UpdateModule = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl
        self.upsample = upsample

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)  # [h/8,w/8,2]
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # For visualization inside BA
        self.update_called_time = 0
        self.show_gru = False
        self.track_ii2jj = torch.tensor([[7, 8], [19, 20]])
        if args is not None and arg_has(args, 'show_gru', False):
            self.show_gru = True
            self.viz_path = arg_has(args, 'show_gru_path', 'temp/show_gru')
            if '{}' in self.viz_path:
                self.viz_path = self.viz_path.replace('{}', '')  # If no sequence ID is specified
            if not os.path.exists(self.viz_path):
                os.makedirs(self.viz_path)
                print('\n### [INFO] {} is created to save the NN outputs'.format(self.viz_path))
            viz_kf_pairs = arg_has(args, 'viz_kf_pairs', [7, 8, 19, 20])
            assert len(viz_kf_pairs) % 2 == 0, \
                'The length of the tracking frame must be even to clarify ' \
                'the start and the end of the BA-flow ' \
                '(e.g. [0,1,5,6] for track 0->1, 5->6 flow)'
            viz_kf_pairs = [int(elem) for elem in viz_kf_pairs]
            self.track_ii2jj = torch.tensor(viz_kf_pairs).reshape(-1, 2)
        
        # confmask
        self.override_conf_mask = override_conf_mask if override_conf_mask is not None else override_conf_mask

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)


        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj)
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix: int):
        """ drop edges from factor graph

        Parameters
        ----------
        ix :int
            Target index to remove from already appended variables and ii & jj pairs
            a) self.video.items
                variables of the VIDEO INSTANCE which is NOW registered in `ix` will be swapped by `ix+1`
            b) self.##_inacs
                Remove `ix` for pairs to run the consequent BA
            c) self.ii & self.jj
                Remove `ix` for pairs to run the consequent BA
            d) self.other_variables (e.g. self.target, self.weight, ....)
                remove actually variables for the BA target
        """

        with self.video.get_lock():
            self.video.tstamp[ix] = self.video.tstamp[ix+1]  # Appended 
            self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.poses_prior[ix] = self.video.poses_prior[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_sens[ix] = self.video.disps_sens[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]

            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)


    @torch.cuda.amp.autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False, itr: int = None,
               diagnose_step: List[int] = None):
        """ run update operator on factor graph with additional visualization;

        Appended Parameters
        ----------
        itr : int
            Current step number
        diagnose_step : List[int]
            Steps to save the flow estimation and weight
            -1      : Initial optical-flow just calculated from depth and pose init values
            0,1,2...: corresponding to the BA step
        """
        diagnose_step = [] if diagnose_step is None else diagnose_step
        viz_hw = (self.video.ht, self.video.wd)
        if self.track_ii2jj.shape[0] != 0:
            # Obtain the indices for visualization in its current BA target.
            # Not that it does NOT corresponding to the actual keyframe ID: thus max will be len(self.ii)-1
            # `-1` means 
            viz_candidates, _ = self.__find_index(
                keys=self.track_ii2jj,  # representing the VKP
                src=torch.stack([self.ii, self.jj]).permute(1, 0).cpu()) if len(self.track_ii2jj) != 0 else [-1]
            save_pair_indices = [elem for elem in viz_candidates if
                                 elem != -1]  # Indicies for self.ii or self.jj e.g. [13, 30, .. len(self.ii)-1]
        else:
            viz_candidates = save_pair_indices = []

        viz_strs = ['-ts-{}-{}-'.format(
            self.video.tstamp[self.ii[pair_id]].to(torch.uint8),
            self.video.tstamp[self.jj[pair_id]].to(torch.uint8)) for pair_id in save_pair_indices]

        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)  # [1,len(self.ii),/8,w,8,2]
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)  # [1,len(self.ii),hd,wd,8,2+2]
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)  # ignore the large flow, [1,len_ii,2+2,hd,wd]
        # correlation features
        corr = self.corr(coords1)

        if (-1 in diagnose_step) and (itr == 0) and (save_pair_indices != []) and self.show_gru:
            mtp = coords1 - self.coords0
            flows = [viz_flow_resizing(mtp[0, pair_id], viz_hw) for pair_id in save_pair_indices]
            for idx, prefix in enumerate(viz_strs):
                cv2.imwrite(
                    os.path.join(self.viz_path, 'f' + prefix + str(self.update_called_time).zfill(4) + '-0-' + '.png'),
                    flows[idx])

        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            ht, wd = self.coords0.shape[0:2]
            self.damping[torch.unique(self.ii)] = damping

            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)
            else:
                # for backend FULL global bundle adjustment
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

            # Start diagnose operation
            diagnose_step = [] if diagnose_step is None else diagnose_step
            viz_hw = (self.video.ht, self.video.wd)

            if self.track_ii2jj.shape[0] != 0:
                viz_candidates, extracted_keys = self.__find_index(
                    keys=self.track_ii2jj,
                    src=torch.stack([ii, jj]).permute(1, 0).cpu()) if len(self.track_ii2jj) != 0 else [-1]
                save_pair_indices = [elem for elem in viz_candidates if elem != -1]  # [e.g. 13, 30, .. len(self.ii)-1]
            else:
                viz_candidates = extracted_keys = []

            viz_strs = ['-ts-{}-{}-'.format(
                self.video.tstamp[ii[pair_id]].to(torch.uint8),
                self.video.tstamp[jj[pair_id]].to(torch.uint8)) for pair_id in save_pair_indices]
            
            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()  # [len_ii,2,hd,wd]
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # dense bundle adjustment
            self.video.ba(target, weight, damping, ii, jj, t0, t1, 
                itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
        
            if self.upsample:
                self.video.upsample(torch.unique(self.ii), upmask)

            if itr in diagnose_step and save_pair_indices != [] and self.show_gru:
                # Flow viz
                flow_after_ba: torch.Tensor = target.unsqueeze(0).permute(0, 1, 3, 4, 2
                                                                          ) - self.coords0  # [1,l_ii,hd,wd,2]
                flows = [viz_flow_resizing(flow_after_ba[0, pair_id], viz_hw) for pair_id in save_pair_indices]

                # Weight viz
                weigh_after_ba: torch.Tensor = weight.unsqueeze(0).permute(0, 1, 3, 4, 2)  # [1,l_ii,hd,wd,2]
                weight_1d = torch.norm(weigh_after_ba,
                                       dim=-1) / torch.sqrt(torch.tensor([2.])).to(weigh_after_ba.device)
                raw_imgs = [self.video.images[ii[pair_id]].permute(1, 2, 0).cpu().numpy() for pair_id in
                            save_pair_indices]
                weights = [viz_weight_on_image(raw_imgs[k], weight_1d[0, pair_id].cpu().numpy()) for k, pair_id in
                           enumerate(save_pair_indices)]
                fws = [viz_high_intensity(weight_1d[0, pair_id].cpu().numpy(), flows[k]) for k, pair_id in
                       enumerate(save_pair_indices)]
                corr_inv_disp = [1. / self.video.disps[ii[pair_id]].cpu().numpy() for pair_id in save_pair_indices]
                vkp_to_be_extracted = [elem for elem in extracted_keys if elem]  # [e.g. 13, 30, .. len(self.ii)-1]

                # save all
                for idx, prefix in enumerate(viz_strs):
                    vkp_from_key, vkp_to_key = vkp_to_be_extracted[idx]
                    vkp_strs = '_vkp-{}-{}_'.format(vkp_from_key, vkp_to_key)
                    path2flow_saving = os.path.join(self.viz_path, 'FVIZ' + prefix + str(self.update_called_time).zfill(
                        4) + vkp_strs + '.png')
                    cv2.imwrite(
                        path2flow_saving,
                        flows[idx])
                    cv2.imwrite(
                        path2flow_saving.replace('FVIZ', 'WVIZ'),
                        weights[idx])
                    cv2.imwrite(
                        path2flow_saving.replace('FVIZ', 'MFLW'),
                        fws[idx])
                    np.savez(path2flow_saving.replace('FVIZ', 'DEP').replace('.png', '.npz'), corr_inv_disp)

        self.age += 1
        self.update_called_time += 1

    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        bar = tqdm(total = steps)
        bar.set_description('Global BA Iteration')

        for step in range(steps):
            # print("Global BA Iteration #{}".format(step+1))
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            s = 8
            for i in range(0, self.jj.max()+1, s):
                # Register all outputs of the neural networks (e.g. targets, weights, etc.)
                # to run Global BA by splitting all ii and jj into N//8 grid for considering the memory consumption
                v = (self.ii >= i) & (self.ii < i + s)
                iis = self.ii[v]
                jjs = self.jj[v]

                ht, wd = self.coords0.shape[0:2]
                if coords1[:,v].shape[1] == 0:
                    continue
                corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                with torch.cuda.amp.autocast(enabled=True):
                 
                    net, delta, weight, damping, upmask = \
                        self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)

                    if self.upsample:
                        self.video.upsample(torch.unique(iis), upmask)

                # target variables to be acquired in this for`loop
                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            damping = .2 * self.damping[torch.arange(self.video.counter.value).to(self.ii.device)].contiguous() + EP
            target = self.target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = self.weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous() # self.weight.shape == [1,nm,h,w,2], will be reshaped latter

            # dense bundle adjustment
            self.video.ba(target, weight, damping, self.ii, self.jj, 1, t, 
                itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False)

            self.video.dirty[:t] = True
            bar.update(1)

    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r
        Just only one-time use called by DroidFrontend.__initialize(self):

        Parameters
        ----------
        t0: int
            == 0,
        t1: int
            == 8, or self.video.counter.value
        r: int
            ==3 is set as the default value to define the reserving time step to run the BA

        Returns
        -------

        """
        assert t0 == 0, "t0 == 0 because this is only for first time use to init the BA pairs"

        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1))
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = 1 if self.video.stereo else 0

        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx)
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf


        es = []
        for i in range(t0, t):
            if self.video.stereo:
                es.append((i, i))
                d[(i-t0)*(t-t1) + (i-t1)] = np.inf

            for j in range(max(i-rad-1,0), i):
                es.append((i,j))
                es.append((j,i))
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                # ignore if the distance is not properly for the BA paring
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)

        self.add_factors(ii, jj, remove)

    @staticmethod
    def __find_index(keys: torch.Tensor, src: torch.Tensor) -> Tuple[List[int], List[int]]:
        """ Get keys to the visualization.

        Parameters
        ----------
        keys: torch.Tensor
            Key tensor with [b,2]
        src: torch.Tensor
            source tensor to search: shapes [B,2]

        Returns
        -------
        Tuple of List[int]
            [0] Pickup scalr index If not found, return -1
            [1] List of keys that appended into the return

        """
        indices = []
        loc_on_keys = []
        for key in keys:
            index = torch.nonzero(torch.all(src == key, dim=1)).squeeze(1)
            indices.append(index.cpu().item() if index.numel() else -1)
            loc_on_keys.append(key.cpu().numpy().tolist() if index.numel() else [])
        return indices, loc_on_keys
