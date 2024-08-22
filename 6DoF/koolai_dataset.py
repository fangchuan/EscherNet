# Copyright (c) 2023-2024, Chuan Fang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
sys.path.append("./")
sys.path.append("../")
import json
from abc import ABC, abstractmethod
import os.path as osp
# from megfile import smart_open, smart_path_join, smart_exists

import numpy as np
import torch
import trimesh

import math
from pathlib import Path
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import webdataset as wds
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from icecream import ic

from utils.typing import *

class KoolAIDataLoader():
    def __init__(self, root_dir: str, 
                 split: str = 'train',
                 split_files: dict = None,
                 image_height: int = 512,
                 image_width: int = 1024,
                 batch_size: int = 1, 
                 total_view: int = 10, 
                 num_workers: int = 4):
        self.root_dir = root_dir
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view
        
        self.split = split
        self.split_files = split_files if split_files is not None \
            else {'train': 'train.json', 'val': 'val.json'}
        self.split_filepath = split_files[split]

        image_transforms = [torchvision.transforms.Resize((image_height, image_width), antialias=True),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = KoolAIPanoData(root_dir=self.root_dir, total_view=self.total_view, validation=False,
                                 split_filepath=self.split_filepath,
                                 image_height=self.image_height, image_width=self.image_width,
                                image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
                             # sampler=sampler)

    def val_dataloader(self):
        dataset = KoolAIPanoData(root_dir=self.root_dir, total_view=self.total_view, validation=True,
                                 split_filepath=self.split_filepath,
                                 image_height=self.image_height, image_width=self.image_width,
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root_dir: str, split_filepath: str):
        super().__init__()
        self.root_dir = root_dir
        self.room_ids = self._load_room_ids(split_filepath)

    def __len__(self):
        return len(self.room_ids)

    @abstractmethod
    def inner_get_item(self, idx):
        pass

    def __getitem__(self, idx):
        try:
            return self.inner_get_item(idx)
        except Exception as e:
            print(f"[DEBUG-DATASET] Error when loading {self.room_ids[idx]}")
            # return self.__getitem__(idx+1)
            raise e

    @staticmethod
    def _load_room_ids(split_filepath: str) -> List[str]:
        # split_filepath is a json file
        with open(split_filepath, 'r') as f:
            uids = f.readlines()
        uids = [uid.strip() for uid in uids]
        return uids

    @staticmethod
    def _load_rgba_image(file_path, bg_color: float = 1.0):
        ''' Load and blend RGBA image to RGB with certain background, 0-1 scaled '''
        rgba = np.array(Image.open(file_path).convert('RGBA'))
        rgba = torch.from_numpy(rgba).float() / 255.0
        rgba = rgba.permute(2, 0, 1).unsqueeze(0)
        rgb = rgba[:, :3, :, :] * rgba[:, 3:4, :, :] + bg_color * (1 - rgba[:, 3:, :, :])
        return rgb

    @staticmethod
    def _load_rgb_image(file_path):
        ''' Load and blend RGBA image to RGB with certain background, 0-1 scaled '''
        rgb = np.array(Image.open(file_path).convert('RGB'))
        rgb = torch.from_numpy(rgb).float() / 255.0
        rgb = rgb.permute(2, 0, 1).unsqueeze(0)
        return rgb
    
    @staticmethod
    def _load_depth_image(file_path, scale: float = 4000.0):
        ''' Load depth image '''
        depth = np.array(Image.open(file_path)).astype(np.int32)
        depth = torch.from_numpy(depth).float() / scale
        depth = depth.unsqueeze(-1).permute(2, 0, 1).unsqueeze(0)
        return depth
    
    @staticmethod
    def _locate_datadir(root_dirs, room_id, locator: str):
        for root_dir in root_dirs:
            datadir = osp.join(root_dir, room_id, locator)
            if osp.exists(datadir):
                return root_dir
        raise FileNotFoundError(f"Cannot find valid data directory for room_id {room_id}")
    
class KoolAIPanoData(BaseDataset):
    def __init__(self,
                 root_dir: str = '.koolai/test_data/',
                 split_filepath: str = 'train.json',
                 image_height: int = 512, 
                 image_width: int = 1024,
                #  image_transforms: torchvision.transforms.transforms.Compose=None,
                 total_view: int= 10,
                 validation: bool=False,
                 T_in: int=1,
                 T_out: int=1,
                 fix_sample: bool=False,
                 ) -> None:
        """Create a dataset from KoolAI dataset.
        dataset structure:
        root_dir:
            --scene_id_1:
                --panorama:
                    --room_id:
                        --rgb:
                            --000.png
                            --001.png
                            ...
                        --depth:
                            --000.png
                            --001.png
                            ...
                        room_meta.json
                --perspective:
                    --room_id:
                        --mvp_rgb:
                            --000.png
                            --001.png
                            ...
                        --mvp_depth:
                            --000.png
                            --001.png
                            ...
                        room_meta.json
                        
            --scene_id_2:
                
        """
        
        super().__init__(root_dir, split_filepath)
        
        self.root_dir = Path(root_dir)
        self.total_view = total_view
        self.n_test_views = total_view - 1
        self.T_in = T_in
        self.T_out = T_out
        self.fix_sample = fix_sample
        self.image_height = image_height
        self.image_width = image_width

        # self.tform = image_transforms

        # total images
        num_rooms = len(self.room_ids)
     
        if validation:
            self.room_ids = self.room_ids[math.floor(num_rooms / 100. * 99.):]  # used last 1% as validation
        else:
            self.room_ids = self.room_ids[:math.floor(num_rooms / 100. * 99.)]  # used first 99% as training
        print('============= length of dataset %d =============' % len(self.room_ids))
        
        room_folders = [osp.join(root_dir, f) for f in self.room_ids if os.path.isdir(osp.join(root_dir, f))]
        num_cams = sum([len([f for f in os.listdir(os.path.join(room_folder, 'rgb')) if f.endswith('.png')]) for room_folder in room_folders])

        self.num_all_rooms = len(self.room_ids)
        self.num_all_views = num_cams
        ic(self.num_all_rooms, self.num_all_views)


    @staticmethod
    def _load_camera_meta(file_path) -> Dict:
        with open(file_path, 'r') as f:
            room_meta = json.load(f)
        
        cam_meta = room_meta['cameras']
        return cam_meta
    
    @staticmethod
    def _load_camera_pose(camera_meta_dict: Dict, idx: int) -> torch.Tensor:
        """ load c2w pose from camera meta

        Args:
            camera_meta_dict (Dict): camera meta json
            idx (int): camera idx

        Returns:
            torch.Tensor: pose
        """
        cam_meta = camera_meta_dict[str(idx)]
        
        # w2c pose
        pose = np.array(cam_meta['camera_transform']).reshape(4, 4)
        c2w_pose = np.linalg.inv(pose)
        c2w_pose = torch.from_numpy(c2w_pose).float()
        return c2w_pose
    
    @staticmethod
    def _load_scale_mat(file_path: str) -> Tuple[Float[Tensor, "4 4"], float]:
        with open(file_path, 'rb') as f:
            room_meta = json.load(f)
        
        scale_mat = np.array(room_meta['scale_mat']).reshape(4, 4)
        scale = float(room_meta['scale'])
        
        return torch.from_numpy(scale_mat).float(), scale
    
    def inner_get_item(self, idx):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
        """
        uid = self.room_ids[idx]
        room_index = idx
        
        rgb_dir = os.path.join(self.root_dir, uid, 'rgb')
        depth_dir = os.path.join(self.root_dir, uid, 'depth')
        normal_dir = os.path.join(self.root_dir, uid, 'normal')
        room_meta_filepath = os.path.join(self.root_dir, uid, 'room_meta.json')
        
        # load camera poses
        cameras_meta = self._load_camera_meta(room_meta_filepath)
        scale_mat, scale = self._load_scale_mat(room_meta_filepath)
        # sample views (incl. source view and side views)
        len_views = len([img for img in os.listdir(rgb_dir) if img.endswith('.png')])
        if len_views < self.total_view:
            ic(f"Warning: {uid} has less than {self.total_view} views, set to {len_views} views")
            self.total_view = len_views
        assert len_views >= self.T_in + self.T_out, f"Error: {uid} has less than {self.T_in + self.T_out} views"
        if self.fix_sample:
            if self.T_out > 1:
                indexes = range(self.total_view)
                index_targets = list(indexes[:2]) + list(indexes[-(self.T_out-2):])
                index_inputs = list(indexes[1:self.T_in+1])   # one overlap identity
                sample_views = index_inputs + index_targets
            else:
                indexes = range(self.total_view)
                index_targets = list(indexes[:self.T_out])
                index_inputs = list(indexes[self.T_out-1:self.T_in+self.T_out-1]) # one overlap identity
                sample_views = index_inputs + index_targets
        else:
            sample_views = np.random.choice(range(len_views), (self.T_in + self.T_out), replace=False)        
            index_inputs = sample_views[:self.T_in]
            index_targets = sample_views[self.T_in:]
        
        # ic(uid, sample_views)
        
        # c2w poses, rgbs, background colors
        poses, rgbs = [], []
        depths = []
        for view in sample_views:
            rgb_path = osp.join(rgb_dir, f'{view}.png')
            depth_path = osp.join(depth_dir, f'{view}.png')
            pose = self._load_camera_pose(cameras_meta, view)
            rgb = self._load_rgb_image(rgb_path)
            depth = self._load_depth_image(depth_path, scale=4000.0)
            # scale pose_c2w
            pose = scale_mat @ pose
            R_c2w = (pose[:3, :3]).numpy()
            q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
            q_c2w = trimesh.transformations.unit_vector(q_c2w)
            R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
            pose[:3, :3] = torch.from_numpy(R_c2w)
            
            # scale rgb to [-1, 1]
            normalized_rgb = rgb * 2.0 - 1.0
            poses.append(pose)
            rgbs.append(normalized_rgb)
            depths.append(depth)
            
        poses: Float[Tensor, "N 4 4"] = torch.stack(poses, dim=0)
        rgbs: Float[Tensor, "N 3 H W"] = torch.cat(rgbs, dim=0)
        depths: Float[Tensor, "N 1 H W"] = torch.cat(depths, dim=0)
        
        # scale depth according to scale
        depths = depths * scale
        
        # source views
        input_images: Float[Tensor, "N 3 H W"] = rgbs[:self.T_in]
        input_images = torch.clamp(input_images, 0, 1)

        # atarget views
        target_images: Float[Tensor, "N 3 H W"] = rgbs[self.T_in:]
        target_images = torch.clamp(target_images, 0, 1)
        
        all_depths: Float[Tensor, "N 1 H W"] = depths
        input_depths: Float[Tensor, "N 1 H W"] = all_depths[:self.T_in]
        target_depths: Float[Tensor, "N 1 H W"] = all_depths[self.T_in:]
        
        cond_Ts: Float[Tensor, "N 4 4"] = poses[:self.T_in]
        target_Ts: Float[Tensor, "N 4 4"] = poses[self.T_in:]
                
        data = {}
        data['room_uid'] = uid
        data['image_input'] = input_images
        data['image_target'] = target_images
        data['depth_input'] = input_depths
        data['depth_target'] = target_depths
        data['pose_out'] = target_Ts
        data['pose_out_inv'] = torch.inverse(target_Ts).permute(0, 2, 1)
        data['pose_in'] = cond_Ts
        data['pose_in_inv'] = torch.inverse(cond_Ts).permute(0, 2, 1)
        
        return data
        
    def compute_room_scale(self, idx:int) -> float:
        """ traverse all depth images and compute the scale of the room

        Args:
            room_id (int): room id

        Returns:
            float: largest depth value
        """
        uid = self.room_ids[idx]        
        depth_dir = os.path.join(self.root_dir, uid, 'depth')
        
        depth_image_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
        depth_image_files.sort(key=lambda x: int(x.split('.')[0]))
        
        max_depth = 0
        for f in depth_image_files:
            depth_path = osp.join(depth_dir, f)
            depth = self._load_depth_image(depth_path, scale=4000.0)
            max_depth = max(max_depth, torch.max(depth).item())
        
        return max_depth * 1.05
    
    def compute_all_room_scale(self) -> List[float]:
        scales = []
        for idx in range(len(self.room_ids)):
            room_uid = self.room_ids[idx]
            scale = self.compute_room_scale(idx)
            ic(room_uid, scale)
            scales.append(scale)
        
        # max_scale = max(scales)
        max_scale = np.quantile(np.array(scales), 0.95)
        return 1./max_scale
    
    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.image_height, "width": self.image_width})
        return batch
    