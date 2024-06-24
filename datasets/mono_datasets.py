# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import time
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms
import pdb


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDatasets(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
    """
    def __init__(self,
                 opt,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False):
        super(MonoDatasets, self).__init__()

        self.opt = opt
        self.height = height
        self.width = width
        #self.num_scales = num_scales
        self.num_scales = 1
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        
    def preprocess(self, inputs, color_aug, height_re_HiS, width_re_HiS, height_re_LoS, width_re_LoS, dx_HiS, dy_HiS, do_crop_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        self.resize_HiS = transforms.Resize((height_re_HiS, width_re_HiS), interpolation=self.interp)
        self.resize_MiS = transforms.Resize((self.height, self.width), interpolation=self.interp)
        self.resize_LoS = transforms.Resize((height_re_LoS, width_re_LoS), interpolation=self.interp)
        box_HiS = (dx_HiS, dy_HiS, dx_HiS + self.width, dy_HiS + self.height)
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                #print('k', k)
                #inputs[(n + "_aug", im, -1)] = []

                for i in range(self.num_scales):
                    #print('i', i)

                    inputs[(n + "_HiS", im, i)]=self.resize_HiS(inputs[(n, im, i - 1)][0]).crop(box_HiS)
                    inputs[(n + "_MiS", im, i)]=self.resize_MiS(inputs[(n, im, i - 1)][0])
                    inputs[(n + "_LoS", im, i)]=self.resize_LoS(inputs[(n, im, i - 1)][0])
                    #print('over!')

        for k in list(inputs):
            f = inputs[k]
            if "color_HiS" in k:
                n, im, i = k

                inputs[(n, im, i)] = self.to_tensor(f)
                #inputs[(n, im, i)] = torch.stack(inputs[(n, im, i)], dim=0)
            if "color_MiS" in k:
                n, im, i = k

                inputs[(n, im, i)] = self.to_tensor(f)# [3,192,640]
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

            if "color_LoS" in k:
                n, im, i = k

                LoS_part = self.to_tensor(f)
                # point1 = int(2*width_re_LoS-self.width)
                # point2 = int(2*height_re_LoS-self.height)
                Tensor_LoS = torch.zeros(3, self.height, self.width)
                Tensor_LoS[:, 0:height_re_LoS, 0:width_re_LoS] = LoS_part
                # Tensor_LoS[:, height_re_LoS:self.height, 0:width_re_LoS] = LoS_part[:, point2:height_re_LoS, 0:width_re_LoS]
                Tensor_LoS[:, height_re_LoS:self.height, 0:width_re_LoS] = 0
                # Tensor_LoS[:, 0:height_re_LoS, width_re_LoS:self.width] = LoS_part[:, 0:height_re_LoS, point1:width_re_LoS]
                Tensor_LoS[:, 0:height_re_LoS, width_re_LoS:self.width] = 0
                # Tensor_LoS[:, height_re_LoS:self.height, width_re_LoS:self.width] = LoS_part[:, point2:height_re_LoS, point1:width_re_LoS]
                Tensor_LoS[:, height_re_LoS:self.height, width_re_LoS:self.width] = 0
                inputs[(n, im, i)] = Tensor_LoS


    def __len__(self):
        return len(self.filenames)
        #return self.num_frames


    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        do_crop_aug = self.is_train

        frame_index = self.filenames[index].strip().split()[0]
        self.get_info(inputs, frame_index, do_flip)

        # High-Scale
        ra_HiS = 1.1
        rb_HiS = 2.0
        resize_ratio_HiS = (rb_HiS - ra_HiS) * random.random() + ra_HiS
        if do_crop_aug:
            height_re_HiS = int(self.height * resize_ratio_HiS)
            width_re_HiS = int(self.width * resize_ratio_HiS)
        else:
            height_re_HiS = self.height
            width_re_HiS = self.width

        height_d_HiS = height_re_HiS - self.height
        width_d_HiS = width_re_HiS - self.width
        if do_crop_aug:
            dx_HiS = int(width_d_HiS * random.random())
            dy_HiS = int(height_d_HiS * random.random())
        else:
            dx_HiS = 0
            dy_HiS = 0

        # Middle-Scale
        dx_MiS = 0
        dy_MiS = 0

        # Low-Scale
        ra_LoS = 0.7
        rb_LoS = 0.9
        resize_ratio_LoS = (rb_LoS - ra_LoS) * random.random() + ra_LoS
        height_re_LoS = int(self.height * resize_ratio_LoS)
        width_re_LoS = int(self.width * resize_ratio_LoS)

        dx_LoS = 0
        dy_LoS = 0

        inputs[("dxy_HiS")] = torch.Tensor((dx_HiS, dy_HiS))
        inputs[("dxy_MiS")] = torch.Tensor((dx_MiS, dy_MiS))
        inputs[("dxy_LoS")] = torch.Tensor((dx_LoS, dy_LoS))
        inputs[("resize_HiS")] = torch.Tensor((width_re_HiS, height_re_HiS))
        inputs[("resize_LoS")] = torch.Tensor((width_re_LoS, height_re_LoS))



        # adjusting intrinsics to match each scale in the pyramid
        if not self.is_train:
            self.frame_idxs = [0]

        '''
        
        '''
        for scale in range(self.num_scales):
            for frame_id in self.frame_idxs:
                inputs[("K_HiS", frame_id, scale)] = []
                inputs[("inv_K_HiS", frame_id, scale)] = []

                inputs[("K_MiS", frame_id, scale)] = []
                inputs[("inv_K_MiS", frame_id, scale)] = []

                inputs[("K_LoS", frame_id, scale)] = []
                inputs[("inv_K_LoS", frame_id, scale)] = []
    
        for index_spatial in range(1):
            for scale in range(self.num_scales):
                for frame_id in self.frame_idxs:
                    K_HiS = inputs[('K_ori', frame_id)][index_spatial].copy()
                    K_MiS = inputs[('K_ori', frame_id)][index_spatial].copy()
                    K_LoS = inputs[('K_ori', frame_id)][index_spatial].copy()

                    #print('judge???',inputs[('K_ori', 0)][index_spatial] == inputs[('K_ori', -1)][index_spatial])


                    K_HiS[0, :] *= (width_re_HiS // (2 ** scale)) / inputs['width_ori'][index_spatial]
                    K_HiS[1, :] *= (height_re_HiS // (2 ** scale)) / inputs['height_ori'][index_spatial]
                    inv_K_HiS = np.linalg.pinv(K_HiS)
                    inputs[("K_HiS", frame_id, scale)].append(torch.from_numpy(K_HiS))
                    inputs[("inv_K_HiS", frame_id, scale)].append(torch.from_numpy(inv_K_HiS))

                    K_MiS[0, :] *= (self.width // (2 ** scale)) / inputs['width_ori'][index_spatial]
                    K_MiS[1, :] *= (self.height // (2 ** scale)) / inputs['height_ori'][index_spatial]
                    inv_K_MiS = np.linalg.pinv(K_MiS)
                    inputs[("K_MiS", frame_id, scale)].append(torch.from_numpy(K_MiS))
                    inputs[("inv_K_MiS", frame_id, scale)].append(torch.from_numpy(inv_K_MiS))

                    K_LoS[0, :] *= (width_re_LoS // (2 ** scale)) / inputs['width_ori'][index_spatial]
                    K_LoS[1, :] *= (height_re_LoS // (2 ** scale)) / inputs['height_ori'][index_spatial]
                    inv_K_LoS = np.linalg.pinv(K_LoS)
                    inputs[("K_LoS", frame_id, scale)].append(torch.from_numpy(K_LoS))
                    inputs[("inv_K_LoS", frame_id, scale)].append(torch.from_numpy(inv_K_LoS))

        for scale in range(self.num_scales):
            for frame_id in self.frame_idxs:
                inputs[("K_HiS",frame_id, scale)] = torch.stack(inputs[("K_HiS",frame_id, scale)], dim=0)
                inputs[("inv_K_HiS",frame_id, scale)] = torch.stack(inputs[("inv_K_HiS", frame_id,scale)], dim=0)

                inputs[("K_MiS", frame_id, scale)] = torch.stack(inputs[("K_MiS", frame_id, scale)], dim=0)
                inputs[("inv_K_MiS", frame_id, scale)] = torch.stack(inputs[("inv_K_MiS", frame_id, scale)], dim=0)

                inputs[("K_LoS", frame_id, scale)] = torch.stack(inputs[("K_LoS", frame_id, scale)], dim=0)
                inputs[("inv_K_LoS", frame_id, scale)] = torch.stack(inputs[("inv_K_LoS", frame_id, scale)], dim=0)

        if do_color_aug:
            #color_aug = transforms.ColorJitter.get_params(
            #    self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = lambda x: x

        #self.preprocess(inputs, color_aug)
        self.preprocess(inputs, color_aug, height_re_HiS, width_re_HiS, height_re_LoS, width_re_LoS, dx_HiS, dy_HiS,
                        do_crop_aug)

        del inputs[("color", 0, -1)]
        if self.is_train:
            for i in self.frame_idxs[1:]:
                del inputs[("color", i, -1)]
                #del inputs[("color_aug", i, -1)]
            for i in self.frame_idxs:
                del inputs[('K_ori', i)]
        else:
            del inputs[('K_ori', 0)]
            
        del inputs['width_ori']
        del inputs['height_ori']




        return inputs

    def get_info(self, inputs, index, do_flip):
        raise NotImplementedError

