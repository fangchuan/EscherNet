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

from koolai_dataset import KoolAIPersData
from utils.typing import *
from utils.misc import get_device, todevice
from utils.cam_ops import cvt_to_perspective_pointcloud, project_points

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on KoolAID dataset')
    parser.add_argument('--root_data_dir', type=str, default='/seaweedfs/training/dataset/qunhe/PanoRoom/processed_data_20240312')
    parser.add_argument('--train_split_file', type=str, default='/seaweedfs/training/dataset/qunhe/PanoRoom/processed_data_20240312/perspective_train.txt')
    parser.add_argument('--test_split_file', type=str, default='/seaweedfs/training/dataset/qunhe/PanoRoom/processed_data_20240312/perspective_test.txt')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=8)
    
    return parser.parse_args()

def main(args):
    
    device = get_device()
    
    # Init Dataset
    T_in = 8
    T_out = 8
    T_in_val = 8
    T_out_val = 8
    train_dataset = KoolAIPersData(root_dir=args.root_data_dir,
                                   split_filepath=args.train_split_file,
                 image_height=256, 
                 image_width=256,
                 total_view= 16,
                 validation=False,
                 T_in=T_in,
                 T_out=T_out,
                 fix_sample=False,)

    validation_dataset = KoolAIPersData(root_dir=args.root_data_dir, 
                                          split_filepath=args.train_split_file,
                                         image_height=256, 
                                        image_width=256,
                                        total_view= 16,
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


    for batch in train_dataloader:
        batch = todevice(batch, device)
        
        if '3FO4K5FWGG13/perspective/room_702' not in batch['room_uid']:
            continue
        
        # ic(batch['image_input'].shape)
        # ic(batch['image_target'].shape)
        # ic(batch['pose_in'].shape)
        # ic(batch['pose_in_inv'].shape)
        # ic(batch['pose_out'].shape)
        # ic(batch['pose_out_inv'].shape)
        input_images = batch['image_input'][0]
        input_depths = batch['depth_input'][0]
        ic(input_images.shape)
        ic(input_depths.shape)
        
        target_images = batch['image_target'][0]
        target_depths = batch['depth_target'][0]
        ic(target_images.shape)
        ic(target_depths.shape)
        
        pose_in = batch['pose_in'][0]
        poses_out = batch['pose_out'][0]
        ic(pose_in.shape)
        ic(poses_out.shape)
        
        input_ply = o3d.geometry.PointCloud()
        
        num_inputs = input_images.shape[0]
        for i in range(num_inputs):
            input_view_rgb = (input_images[i, :, :, :] + 1 ) / 2
            input_view_depth = input_depths[i, :, :, :]
            o3d_ply_cam0 = cvt_to_perspective_pointcloud(depth_image=input_view_depth,
                                        rgb_image=input_view_rgb,
                                        depth_scale=1.0,
                                        hfov_rad=0.5*np.pi,
                                        wfov_rad=0.5*np.pi)
            o3d.io.write_point_cloud(f'./input_cam{i}.ply', o3d_ply_cam0)
            input_ply += o3d_ply_cam0.transform(pose_in[i].cpu().numpy())
        o3d.io.write_point_cloud('./input_pointcloud.ply', input_ply)
        
        target_ply = o3d.geometry.PointCloud()
        num_target = target_images.shape[0]
        for i in range(num_target):
            sample_view_rgb = (target_images[i, :, :, :] + 1 )/ 2
            sample_view_depth = target_depths[i, :, :, :]
            o3d_ply_cam0 = cvt_to_perspective_pointcloud(depth_image=sample_view_depth,
                                        rgb_image=sample_view_rgb,
                                        depth_scale=1.0,
                                        hfov_rad=0.5*np.pi,
                                        wfov_rad=0.5*np.pi)
            o3d.io.write_point_cloud(f'./target_cam{i}.ply', o3d_ply_cam0)
            target_ply += o3d_ply_cam0.transform(poses_out[i].cpu().numpy())
            
        o3d.io.write_point_cloud('./target_pointcloud.ply', target_ply)
        
        
    for batch in validation_dataloader:
        batch = todevice(batch, device)
        
        ic(batch['room_uid'])
        input_images = batch['image_input'][0]
        input_depths = batch['depth_input'][0]
        ic(input_images.shape)
        ic(input_depths.shape)
        
        target_images = batch['image_target'][0]
        target_depths = batch['depth_target'][0]
        ic(target_images.shape)
        ic(target_depths.shape)
        
        pose_in = batch['pose_in'][0]
        poses_out = batch['pose_out'][0]
        ic(pose_in.shape)
        ic(poses_out.shape)
        
        input_ply = o3d.geometry.PointCloud()
        
        num_inputs = input_images.shape[0]
        for i in range(num_inputs):
            input_view_rgb = (input_images[i, :, :, :] + 1 ) / 2
            input_view_depth = input_depths[i, :, :, :]
            o3d_ply_cam0 = cvt_to_perspective_pointcloud(depth_image=input_view_depth,
                                        rgb_image=input_view_rgb,
                                        depth_scale=1.0,
                                        hfov_rad=0.5*np.pi,
                                        wfov_rad=0.5*np.pi)
            o3d.io.write_point_cloud(f'./input_cam{i}.ply', o3d_ply_cam0)
            input_ply += o3d_ply_cam0.transform(pose_in[i].cpu().numpy())
        o3d.io.write_point_cloud('./input_pointcloud.ply', input_ply)
        
        target_ply = o3d.geometry.PointCloud()
        num_target = target_images.shape[0]
        for i in range(num_target):
            sample_view_rgb = (target_images[i, :, :, :] + 1 )/ 2
            sample_view_depth = target_depths[i, :, :, :]
            o3d_ply_cam0 = cvt_to_perspective_pointcloud(depth_image=sample_view_depth,
                                        rgb_image=sample_view_rgb,
                                        depth_scale=1.0,
                                        hfov_rad=0.5*np.pi,
                                        wfov_rad=0.5*np.pi)
            o3d.io.write_point_cloud(f'./target_cam{i}.ply', o3d_ply_cam0)
            target_ply += o3d_ply_cam0.transform(poses_out[i].cpu().numpy())
            
        o3d.io.write_point_cloud('./target_pointcloud.ply', target_ply)
        
        exit()
        

if __name__ == '__main__':
    args = parse_args()
    main(args)