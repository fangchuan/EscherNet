import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.insert(0, './6DoF/')
import argparse



import cv2
import numpy as np
import torch
import einops
from diffusers import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from icecream import ic
from packaging import version
from torchvision import transforms
from PIL import Image

from koolai_dataset import KoolAIPanoData
from utils.typing import *
from utils.misc import get_device, todevice


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on KoolAID dataset')
    parser.add_argument('--root_data_dir', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/')
    parser.add_argument('--train_split_file', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/train.txt')
    parser.add_argument('--test_split_file', type=str, default='/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/test.txt')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    
    return parser.parse_args()

def main(args):
    # prepare dataset
    device = get_device()
    
    # prepare VVAE model
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae = vae.to(device).eval()
    
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        ic(xformers_version)
        vae.enable_slicing()
        
    output_folder = './vae_results'
    os.makedirs(output_folder, exist_ok=True)

    T_in = 1
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
    
    # prepare VVAE model
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae = vae.to(device).eval()
    
    test_image_filepath = '/mnt/nas_3dv/hdd1/datasets/KoolAI/processed_data_20240413/3FO4K5G405RH/panorama/room_638/rgb/20.png'
    to_tensor_fn = transforms.ToTensor()
    normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    rgb = np.array(Image.open(test_image_filepath).convert("RGB")).astype(np.float32)
    # input_rgb = to_tensor_fn(rgb).float()
    inputs = {}
    inputs["rgb"] = torch.from_numpy(rgb/127.5-1.0).permute(2, 0, 1).unsqueeze(0).to(device)
    # inputs["normalized_rgb"] = normalize_fn(input_rgb).unsqueeze(0).to(device)
    
    # for batch in train_dataloader:
    for i in range(1):

        with torch.no_grad():
            # images: Float[Tensor, "B *N C H W"] = batch['image_input'].to(device)
            images: Float[Tensor, "B N C H W"] = inputs["rgb"].unsqueeze(0)
            ic(images.shape)
            
            images = einops.rearrange(images, 'b t c h w -> (b t) c h w')
            
            gt_latents = vae.encode(images).latent_dist.sample().detach()
            # follow zero123, only target image latent is scaled
            gt_latents = gt_latents * vae.config.scaling_factor
                
            raw_image = images.squeeze().cpu().permute(1, 2, 0).numpy()
            raw_image = ((raw_image+1.0)/2.0 * 255).astype(np.uint8)
            Image.fromarray(raw_image).save(os.path.join(output_folder, 'raw_image.png'))
            print(f'z.shape: {gt_latents.shape}')
            cv2.imwrite(os.path.join(output_folder, 'latent_z.png'), gt_latents.squeeze(0).permute(1, 2, 0).cpu().numpy())

            def decode_latents(latents):
                latents = 1.0/vae.config.scaling_factor * latents
                image = vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
                image = image.squeeze().cpu().permute(1, 2, 0).float().numpy()
                return image
            
            xrec = decode_latents(gt_latents)
            print(f'xrec.shape: {xrec.shape}')
            xrec = (xrec * 255).astype(np.uint8)
            Image.fromarray(xrec).save(os.path.join(output_folder, 'xrec.png'))
    
    
if __name__ == '__main__':
    
    args = parse_args()
    main(args)
    






