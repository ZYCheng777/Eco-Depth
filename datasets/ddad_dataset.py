# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import pickle
import pdb
import cv2

from .mono_dataset import MonoDataset
from .mono_datasets import MonoDatasets


class DDADDataset(MonoDatasets):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(DDADDataset, self).__init__(*args, **kwargs)

        self.split = 'train' if self.is_train else 'val'
        self.rgb_path = '/data1/czy/DDAD_dataset/raw_data'
        self.depth_path = '/data1/czy/DDAD_dataset/depth'
        #self.match_path = '/home/ace/data/DDAD_dataset/match'
        #self.mask_path = '/home/ace/data/DDAD_dataset/mask'

        with open('datasets/ddad/{}.txt'.format(self.split), 'r') as f:
            self.filenames = f.readlines()

        with open('datasets/ddad/info_{}.pkl'.format(self.split), 'rb') as f:
            self.info = pickle.load(f)

        #self.camera_ids = ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right']
        #self.camera_names = ['CAMERA_01', 'CAMERA_05', 'CAMERA_07', 'CAMERA_09', 'CAMERA_08','CAMERA_06']

        self.camera_ids = ['front']
        self.camera_names = ['CAMERA_01']
        
    
    def get_info(self, inputs, index_temporal, do_flip):
        inputs[("color", 0, -1)] = []
        if self.is_train:
            for idx, i in enumerate(self.frame_idxs[1:]):
                inputs[("color", i, -1)] = []

            for idx, i in enumerate(self.frame_idxs):
                inputs[('K_ori', i)] = []
            
            #inputs['mask_ori'] = []
            #inputs["pose_spatial"] = []
        else:
            inputs[('K_ori', 0)] = []
            inputs['depth'] = []


        #inputs['width_ori'], inputs['height_ori'], inputs['id'] = [], [], []
        inputs['width_ori'], inputs['height_ori'] = [], []

        scene_id = self.info[index_temporal]['scene_name']

        for index_spatial in range(1):
            #inputs['id'].append(self.camera_ids[index_spatial])
            color = self.loader(os.path.join(self.rgb_path, scene_id, 'rgb', 
                                self.camera_names[index_spatial], index_temporal + '.png'))
            inputs['width_ori'].append(color.size[0])
            inputs['height_ori'].append(color.size[1])
            

            if not self.is_train:
                depth = np.load(os.path.join(self.depth_path, scene_id, 'depth',
                                             self.camera_names[index_spatial], index_temporal + '.npy'))
                inputs['depth'].append(depth.astype(np.float32))

            
            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
            inputs[("color", 0, -1)].append(color)
    
    
            K = np.eye(4).astype(np.float32)
            K[:3, :3] = self.info[index_temporal][self.camera_names[index_spatial]]['intrinsics']
            inputs[('K_ori', 0)].append(K)
            if self.is_train:
                for idx, i in enumerate(self.frame_idxs[1:]):
                    index_temporal_i = self.info[index_temporal]['context'][idx]

                    K = np.eye(4).astype(np.float32)
                    K[:3, :3] = self.info[index_temporal_i][self.camera_names[index_spatial]]['intrinsics']
                    inputs[('K_ori', i)].append(K)

                    color = self.loader(os.path.join(self.rgb_path, scene_id, 'rgb',
                                                     self.camera_names[index_spatial], index_temporal_i + '.png'))

                    if do_flip:
                        color = color.transpose(pil.FLIP_LEFT_RIGHT)

                    inputs[("color", i, -1)].append(color)



    
        if self.is_train:
            for idx, i in enumerate(self.frame_idxs):
                inputs[('K_ori', i)] = np.stack(inputs[('K_ori', i)], axis=0)
        else:
            inputs[('K_ori', 0)] = np.stack(inputs[('K_ori', 0)], axis=0)
            inputs['depth'] = np.stack(inputs['depth'], axis=0)

        for key in ['width_ori', 'height_ori']:
            inputs[key] = np.stack(inputs[key], axis=0)








