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
from dataclasses import dataclass
from icecream import ic
from einops import rearrange
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

from koolai_dataset import KoolAIPanoData
from models.egformer import EGTransformer
from utils.typing import *
from utils.misc import get_device, todevice, load_module_weights
from utils.vis_ops import get_pca_map

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on KoolAID dataset')
    parser.add_argument('--root_data_dir', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/')
    parser.add_argument('--train_split_file', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/train.txt')
    parser.add_argument('--test_split_file', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/test.txt')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--ckpt_path', type=str, default='/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/egformer/ckpts/EGformer_pretrained.pkl')
    
    return parser.parse_args()

def pca_vis(features: Float[Tensor, "C H W"], 
            n_components: int = 3,
            background_threshold: float = 0.25,
            is_foreground_larger_than_threshold: bool = True,):
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)

    # Min-Max Scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(clip=True)
    
    ic(features.shape)
    feat_img_h, feat_img_w = features.shape[1], features.shape[2]
    features = rearrange(features, "C H W -> (H W) C")
    pca.fit(features)
    pca_features = pca.transform(features)
    scaler.fit(pca_features)
    pca_features = scaler.transform(pca_features)

    # visualize PCA components for finding a proper threshold
    # 3 histograms for 3 components
    ic('find a proper threshold for background')
    plt.subplot(2, 2, 1)
    plt.hist(pca_features[:, 0])
    plt.subplot(2, 2, 2)
    plt.hist(pca_features[:, 1])
    plt.subplot(2, 2, 3)
    plt.hist(pca_features[:, 2])
    plt.show()
    plt.close()
    
    # pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
    #                  (pca_features[:, 0].max() - pca_features[:, 0].min())
                     
    # Foreground/Background
    if is_foreground_larger_than_threshold:
        pca_features_bg = pca_features[:, 0] < background_threshold
    else:
        pca_features_bg = pca_features[:, 0] > background_threshold
    pca_features_fg = ~pca_features_bg

    # PCA with only foreground
    pca.fit(features[pca_features_fg])
    pca_features_rem = pca.transform(features[pca_features_fg])

    # Min Max Normalization
    scaler.fit(pca_features_rem)
    pca_features_rem = scaler.transform(pca_features_rem)
    ic('segment foreground and background')
    
    pca_features_rgb = np.zeros((feat_img_h * feat_img_w, 3))
    pca_features_rgb[pca_features_bg] = 0
    pca_features_rgb[pca_features_fg] = pca_features_rem
    pca_features_rgb = pca_features_rgb.reshape(feat_img_h, feat_img_w, 3)
    plt.imshow(pca_features_rgb)
    plt.show()
    return (pca_features_rgb).astype(np.float32)
    
def main(args):
    device = get_device()
    
    # prepare dataset
    T_in = 1
    T_out = 3
    valid_dataset = KoolAIPanoData(root_dir=args.root_data_dir,
                                   split_filepath=args.train_split_file,
                 image_height=512, 
                 image_width=1024,
                 total_view= 10,
                 validation=True,
                 T_in=T_in,
                 T_out=T_out,
                 fix_sample=False,)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=valid_dataset.collate
    )
    
    module = EGTransformer()
    # load pretrained model
    state_dicts = load_module_weights(args.ckpt_path, map_location='cpu')
    module.load_state_dict(state_dicts, strict=False)
    module.to(device)

    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()

    test_image_filepath = '/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/3FO4K5G405RH/panorama/room_638/rgb/20.png'
    to_tensor_fn = transforms.ToTensor()
    normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    rgb = np.array(Image.open(test_image_filepath).convert("RGB"))
    input_rgb = to_tensor_fn(rgb.copy()).float()
    inputs = {}
    inputs["rgb"] = input_rgb.unsqueeze(0).to(device)
    inputs["normalized_rgb"] = normalize_fn(input_rgb).unsqueeze(0).to(device)
    ic(inputs["rgb"].shape)
    
    for batch in valid_dataloader:
        batch = todevice(batch, device)
        
        ic(batch['image_input'].shape)
        # images: Float[Tensor, "B *N C H W"] = batch['image_input'].permute(0, 1, 4, 2, 3)
        images: Float[Tensor, "B N C H W"] = inputs["rgb"].unsqueeze(0)
        batch_size, n_input_views = images.shape[:2]
        raw_img_width, raw_img_height = images.shape[4], images.shape[3]
        out = module(
            rearrange(images, "B N C H W -> (B N) C H W"),
            modulation_cond=None,
        )
        depth_features, all_hidden_features = out.last_hidden_state, out.hidden_states
        semantic_features = all_hidden_features[0]
        depth_features = rearrange(
            depth_features, "(B N) Ct H W -> B N Ct H W", B=batch_size, H=raw_img_height//2, W=raw_img_width//2
        )
        semantic_features = rearrange(semantic_features, "(B N) Ct H W -> B N Ct H W", B=batch_size, H=raw_img_height//16, W=raw_img_width//16)
        ic(depth_features.shape)
        ic(semantic_features.shape)
        
        plt.imshow(images[0, 0].permute(1,2,0).cpu().numpy())
        plt.show()
        plt.close()
        
        # save the features
        vis_depth_featmap = get_pca_map(feature_map=depth_features[0].detach().cpu().permute(0, 2, 3, 1),
                                        img_size=(256, 512),)
        vis_depth_featmap = (vis_depth_featmap * 255).astype(np.uint8)
        Image.fromarray(vis_depth_featmap).save(f'./depth_features_{batch_size}_{n_input_views}.png')
        
        vis_semantic_featmap = get_pca_map(feature_map=semantic_features[0].detach().cpu().permute(0, 2, 3, 1),
                                           img_size=(256, 512),)
        vis_semantic_featmap = (vis_semantic_featmap * 255).astype(np.uint8)
        Image.fromarray(vis_semantic_featmap).save(f'./semantic_features_{batch_size}_{n_input_views}.png')
        break

if __name__ == '__main__':
    args = parse_args()
    main(args)