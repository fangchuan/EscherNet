import os
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.insert(0, "./6DoF/")
import argparse


import numpy as np
import torch
import torchvision
from icecream import ic
from matplotlib import pyplot as plt
from einops import rearrange
from PIL import Image
from torchvision import transforms

# from DenoisingViT.DenoisingViT.vit_wrapper import ViTWrapper
from koolai_dataset import KoolAIPanoData
from models.dinov2 import DinoV2
from utils.typing import *
from utils.misc import get_device, todevice
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


def main(args):
    
    device = get_device()
    # preparre DINO model
    ic("=> creating vit")
    # vit_model_type = 'vit_base_patch14_dinov2.lvd142m'
    dino_stride = 14
    # vit = ViTWrapper(model_type=vit_model_type, stride=dino_stride)
    # vit = vit.to(device).eval()
    # feature_dim = vit.n_output_dims
    # layer_index = int(vit.last_layer_index)
    # num_blocks = int(vit.num_blocks)
    # ic('last layer feature dim:', feature_dim)
    # ic('last layer index:', layer_index)
    # ic('num_blocks:', num_blocks)
    module = DinoV2()
    
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
    
    test_image_filepath = '/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/3FO4K5G405RH/panorama/room_638/rgb/20.png'
    to_tensor_fn = transforms.ToTensor()
    # new_h = int(np.ceil(512/dino_stride) * dino_stride)
    # new_w = int(np.ceil(1024/dino_stride) * dino_stride)
    # print(new_h, new_w)
    # resize_tensor_fn = transforms.Resize((new_h, new_w))
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
        # out = module(
        #     rearrange(images, "B N C H W -> (B N) C H W"),
        #     modulation_cond=None,
        # )
        # out = vit.get_intermediate_layers(x=rearrange(images, "B N C H W -> (B N) C H W"), 
        #                                   n=[layer_index],
        #                                   reshape=True,)[-1]
        out = module(rearrange(images, "B N C H W -> (B N) C H W"),
                     modulation_cond=None,)
        semantic_features = out
        semantic_features = rearrange(semantic_features, "(B N) Ct H W -> B N Ct H W", 
                                      B=batch_size,)
        ic(semantic_features.shape)
        
        plt.imshow(images[0, 0].permute(1,2,0).cpu().numpy())
        plt.show()
        plt.close()
        
        # save the features
        vis_semantic_featmap = get_pca_map(feature_map=semantic_features[0].detach().cpu().permute(0, 2, 3, 1),
                                           img_size=(256, 512),)
        vis_semantic_featmap = (vis_semantic_featmap * 255).astype(np.uint8)
        Image.fromarray(vis_semantic_featmap).save(f'./semantic_features_{batch_size}_{n_input_views}.png')
        break
    
    
if __name__=='__main__':
    args = parse_args()
    main(args)