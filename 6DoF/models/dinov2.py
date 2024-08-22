import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torchvision import transforms
from omegaconf import OmegaConf
from icecream import ic

from utils.typing import *
from utils.misc import get_device
from models.dino.vit_wrapper import ViTWrapper

class DinoV2(nn.Module):
    @dataclass
    class Config:
        img_size: List[int] = field(default_factory=lambda : [512, 1024])
        model_type: str = "vit_base_patch14_dinov2.lvd142m"
        stride: int = 14
        patch_size: int = 14
        in_chans: int = 3
        
    
    cfg: Config
    """ Vision Transformer with support for patch or hybrid CNN input stage
    Args:
        depth       : Number of blocks in each stage
        split_size  : Width(Height) of stripe size in each stage
        num_heads   : Number of heads in each stage
        hybrid      : Whether to use hybrid patch embedding (ResNet50)/ Not used
    """
    def configure(self) -> None:
        self.cfg = OmegaConf.structured(self.Config)
        self.device = get_device()

        image_height, image_width = self.cfg.img_size
        
        self.model = ViTWrapper(model_type=self.cfg.model_type, stride=self.cfg.stride)
        self.model = self.model.to(self.device).eval()
        feature_dim = self.model.n_output_dims
        self.query_layer_index = int(self.model.last_layer_index)
        num_blocks = int(self.model.num_blocks)
        # ic('last layer feature dim:', feature_dim)
        # ic('last layer index:', self.query_layer_index)
        # ic('num_blocks:', num_blocks)
        
        new_height = int(math.ceil(image_height / self.cfg.stride) * self.cfg.stride)
        new_width = int(math.ceil(image_width / self.cfg.stride) * self.cfg.stride)
        self.image_preprocess = transforms.Compose([
                                transforms.Resize((new_height, new_width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def __init__(self) -> None:
        super().__init__()
        self.configure()
        
    def forward_features(self, pixel_values: torch.Tensor):
        # pixel_values = self.preprocess_image(pixel_values)
        out = self.model(pixel_values)
        out = self.model.get_intermediate_layers(x=pixel_values,
                                           n=self.query_layer_index,
                                           reshape=True)[-1]
        return out
    
    def forward(self, 
                pixel_values: Optional[torch.Tensor] = None,
                modulation_cond: Optional[torch.Tensor] = None,) -> Float[Tensor, "BN C H W"]:
        pixel_values = self.image_preprocess(pixel_values)
        out = self.forward_features(pixel_values)
        return out