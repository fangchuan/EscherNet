
import os
import sys

import torch.utils
import torch.utils.data
sys.path.append('./')
sys.path.append('../')
sys.path.insert(0, './6DoF/')
import argparse
import json

import numpy as np
import torch
from PIL import Image
import open3d as o3d
import trimesh
# from accelerate.logging import get_logger
import logging
from logging import getLogger as get_logger


from koolai_dataset import KoolAIPanoData
from utils.typing import *
from utils.misc import get_device, todevice

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

R_gl_cv = np.asarray([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])
R_cv_gl = np.linalg.inv(R_gl_cv)
        
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on KoolAID dataset')
    parser.add_argument('--root_data_dir', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/')
    parser.add_argument('--train_split_file', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/train.txt')
    parser.add_argument('--test_split_file', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/test.txt')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--skip_calc_scene_scale', action='store_true')
    parser.add_argument('--skip_save_scene_scale', action='store_true')
    
    return parser.parse_args()

def main(args: argparse.Namespace):
    
    device = get_device()
    
    # prepare dataset
    T_in = 3
    T_out = 3
    train_dataset = KoolAIPanoData(root_dir=args.root_data_dir,
                                   split_filepath=args.train_split_file,
                 image_height=512, 
                 image_width=1024,
                 total_view= 10,
                 validation=False,
                 T_in=T_in,
                 T_out=T_out,
                 fix_sample=False,)
    
    # for training
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=train_dataset.collate
    )
    
    if not args.skip_calc_scene_scale:
    # calculate all rooms' scale
        room_scale_lst = []
        for idx in range(len(train_dataset)):
            room_uid = train_dataset.room_ids[idx]
            room_distance_scale = train_dataset.compute_room_scale(idx)
            room_scale_lst.append(torch.tensor([room_distance_scale], dtype=torch.float32))
            logger.info(f' room {room_uid} distance scale: {room_distance_scale}')

            
        room_scales = torch.cat(room_scale_lst, dim=0)    
        # draw room scale histogram
        import matplotlib.pyplot as plt
        plt.hist(room_scales.cpu().numpy(), bins=100)
        plt.show()
        
        # we take 95% quantile as the new scene scale, to remove some rooms with invaliid scale
        new_scene_scale = torch.quantile(room_scales, 0.95).item()    
        logger.info(f'training dataset room scale: {new_scene_scale}')
    else:
        # TODO: manually set the new_scene_scale
        new_scene_scale = 10.16777
    new_scene_scale = 1.0 / new_scene_scale
    
    if not args.skip_save_scene_scale:
        for idx in range(len(train_dataset)):
            room_uid = train_dataset.room_ids[idx]
            room_folderpath = os.path.join(args.root_data_dir, room_uid)
            room_meta_filepath = os.path.join(room_folderpath, 'room_meta.json')
            logger.info(f'correct scene scale for room: {room_meta_filepath}')
            
            with open(room_meta_filepath, 'r') as f:
                room_meta = json.load(f)
                
            scale_mat = np.array(room_meta['scale_mat']).reshape(4,4).astype(np.float32)
            original_scale = float(room_meta['scale'])
            
            # savee corrected scale_mat
            camera_center = scale_mat[:3, 3] / original_scale
            
            new_scale_mat = np.eye(4).astype(np.float32)
            new_scale_mat[:3, 3] = camera_center
            new_scale_mat[:3] *= new_scene_scale
            
            room_meta['new_scale_mat'] = new_scale_mat.flatten().tolist()
            room_meta['new_scale'] = new_scene_scale
            json.dump(room_meta, open(room_meta_filepath, 'w'), indent=4)
            
            # visualize normalized camera poses
            pose_mesh = o3d.geometry.TriangleMesh()
            
            camera_metas = room_meta['cameras']
            for cam_idx in range(len(camera_metas)):     
                cam_meta = camera_metas[str(cam_idx)]             
                # w2c pose
                pose = np.array(cam_meta['camera_transform']).reshape(4, 4)
                c2w_pose = np.linalg.inv(pose)
                # scale pose_c2w
                c2w_pose = new_scale_mat @ c2w_pose
                R_c2w = c2w_pose[:3, :3]
                q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
                q_c2w = trimesh.transformations.unit_vector(q_c2w)
                R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
                c2w_pose[:3, :3] = R_c2w
                
                T = c2w_pose
                T[:3, :3] = T[:3, :3] @ R_cv_gl
                pose_mesh += o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01).transform(T)
                o3d.io.write_triangle_mesh(os.path.join(room_folderpath, 'pose_mesh.ply'), pose_mesh)
            

    # # test on a specific room
    # room_folderpath = os.path.join(args.root_data_dir, '3FO4K5G0FKM8/panorama/room_471')
    # room_meta_filepath = os.path.join(room_folderpath, 'room_meta.json')
    # logger.info(f'correct scene scale for room: {room_meta_filepath}')
        
    # with open(room_meta_filepath, 'r') as f:
    #     room_meta = json.load(f)
        
    # scale_mat = np.array(room_meta['scale_mat']).reshape(4,4).astype(np.float32)
    # original_scale = float(room_meta['scale'])
    
    # # savee corrected scale_mat
    # camera_center = scale_mat[:3, 3] / original_scale
    # logger.info(f'original camera_center: {-camera_center}')
    
    # new_scale_mat = np.eye(4).astype(np.float32)
    # new_scale_mat[:3, 3] = camera_center
    # new_scale_mat[:3] *= new_scene_scale
    
    # room_meta['new_scale_mat'] = new_scale_mat.flatten().tolist()
    # room_meta['new_scale'] = new_scene_scale
    # json.dump(room_meta, open(room_meta_filepath, 'w'), indent=4)
    
    # # visualize normalized camera poses
    # pose_mesh = o3d.geometry.TriangleMesh()
    
    # camera_metas = room_meta['cameras']
    # for cam_idx in range(len(camera_metas)):     
    #     cam_meta = camera_metas[str(cam_idx)]       
    #     # w2c pose
    #     pose = np.array(cam_meta['camera_transform']).reshape(4, 4)
    #     c2w_pose = np.linalg.inv(pose)
    #     # scale pose_c2w
    #     c2w_pose = new_scale_mat @ c2w_pose
    #     R_c2w = c2w_pose[:3, :3]
    #     q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
    #     q_c2w = trimesh.transformations.unit_vector(q_c2w)
    #     R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
    #     c2w_pose[:3, :3] = R_c2w
        
    #     T = c2w_pose
    #     T[:3, :3] = T[:3, :3] @ R_cv_gl
    #     pose_mesh += o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01).transform(T)
        
    #     o3d.io.write_triangle_mesh(os.path.join(room_folderpath, 'pose_mesh.ply'), pose_mesh)

if __name__ == '__main__':
    args = parse_args()
    main(args)