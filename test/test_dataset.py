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
import os.path as osp
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.insert(0, "./6DoF/")
import argparse

import numpy as np
import torch
import torchvision
from icecream import ic
import open3d as o3d

from koolai_dataset import KoolAIPanoData
from utils.typing import *
from utils.misc import get_device, todevice
from utils.pano_ops import cvt_to_spherical_pointcloud
from utils.cam_ops import project_points

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on KoolAID dataset')
    parser.add_argument('--root_data_dir', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/')
    parser.add_argument('--train_split_file', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/train.txt')
    parser.add_argument('--test_split_file', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/test.txt')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    
    return parser.parse_args()

def main(args):
    
    device = get_device()
    
    # Init Dataset
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((512, 1024)),  # 256, 256
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ]
    )
    T_in = 3
    T_out = 3
    T_in_val = 1
    T_out_val = 5
    train_dataset = KoolAIPanoData(root_dir=args.root_data_dir,
                                   split_filepath=args.train_split_file,
                 image_height=512, 
                 image_width=1024,
                #  image_transforms=image_transforms,
                 total_view= 10,
                 validation=False,
                 T_in=T_in,
                 T_out=T_out,
                 fix_sample=False,)
    # train_log_dataset = KoolAIPanoData(root_dir=args.root_data_dir, 
    #                                      split_filepath=args.train_split_file,
    #              image_height=512, 
    #              image_width=1024,
                #  image_transforms=image_transforms,
    #              total_view= 10,
    #                                      validation=False, 
    #                                      T_in=T_in_val, 
    #                                      T_out=T_out, 
    #                                      fix_sample=True)
    validation_dataset = KoolAIPanoData(root_dir=args.root_data_dir, 
                                          split_filepath=args.train_split_file,
                                         image_height=512, 
                                        image_width=1024,
                                        # image_transforms=image_transforms,
                                        total_view= 10,
                                          validation=True, 
                                          T_in=T_in_val, 
                                          T_out=T_out, 
                                          fix_sample=True)
    # for training
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=train_dataset.collate
    )
    # for validation set logs
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
        collate_fn=validation_dataset.collate
    )
    # for training set logs
    # train_log_dataloader = torch.utils.data.DataLoader(
    #     train_log_dataset,
    #     shuffle=False,
    #     batch_size=1,
    #     num_workers=1,
    #     collate_fn=train_log_dataset.collate
    # )

    for batch in train_dataloader:
        batch = todevice(batch, device)
        
        ic(batch['image_input'].shape)
        ic(batch['image_target'].shape)
        ic(batch['pose_in'].shape)
        ic(batch['pose_in_inv'].shape)
        ic(batch['pose_out'].shape)
        ic(batch['pose_out_inv'].shape)
        
    for batch in validation_dataloader:
        batch = todevice(batch, device)
        
        input_image = batch['image_input'][0]
        input_depth = batch['depth_input'][0]
        ic(input_image.shape)
        ic(input_depth.shape)
        
        target_images = batch['image_target'][0]
        target_depths = batch['depth_target'][0]
        ic(target_images.shape)
        ic(target_depths.shape)
        
        pose_in = batch['pose_in'][0]
        poses_out = batch['pose_out'][0]
        ic(pose_in.shape)
        ic(poses_out.shape)
        
        pcl_cam0 = cvt_to_spherical_pointcloud(depth_img=input_depth,
                                    rgb_img=input_image,
                                    depth_scale=1.0,
                                    num_sample_points=100000,
                                    saved_color_pcl_filepath='./pointcloud.ply')
        pcl_world0 = project_points(pcl_cam0[:, :, :3], pose_in[0:1, :, :])
        merged_ply = o3d.geometry.PointCloud()
        merged_ply.points = o3d.utility.Vector3dVector(pcl_world0.detach().cpu().numpy().reshape(-1, 3))
        
        num_target = target_images.shape[0]
        for i in range(num_target):
            sample_view_rgb = target_images[i:i+1, :, :, :]
            sample_view_depth = target_depths[i:i+1, :, :, :]
            pcl_cami = cvt_to_spherical_pointcloud(depth_img=sample_view_depth,
                                        rgb_img=sample_view_rgb,
                                        depth_scale=1.0,
                                        num_sample_points=100000,
                                        saved_color_pcl_filepath=f'./pointcloud_{i}.ply')
            pcl_worldi = project_points(pcl_cami[:, :, :3], poses_out[i:i+1, :, :])
            ply_worldi = o3d.geometry.PointCloud()
            ply_worldi.points = o3d.utility.Vector3dVector(pcl_worldi[:,:,:3].detach().cpu().numpy().reshape(-1, 3))
            merged_ply += ply_worldi
            
        o3d.io.write_point_cloud('./merged_pointcloud.ply', merged_ply)
        
        exit()
        

if __name__ == '__main__':
    args = parse_args()
    main(args)