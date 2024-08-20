# ------------------------------------------
# Equirectangular geometry-biased Transformer
# ------------------------------------------

# These codes are based on the following codes :
# CSwin-Transformer (https://github.com/microsoft/CSWin-Transformer), 
# Panoformer (https://github.com/zhijieshen-bjtu/PanoFormer),
# and others.

# We thank the authors providing the codes availble

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np

from dataclasses import dataclass
from utils.typing import *
from models.panoformer.equi_sampling import genSamplingPattern
from models.panoformer.model import BasicPanoformerLayer

from dataclasses import field

from utils.typing import DictConfig, Optional, Union

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x



VRPE_LUT=[]
HRPE_LUT=[]


def VRPE(num_heads, height,width,split_size): ## Used for vertical attention  / (theta,phi) -> (theta,phi') / This part is coded assuming that the split_size (stripe_with) is 1 to reduce the computational cost. To use larger split_size, ERPE must be calculated accoringly.

    H = height // split_size # Base height
    pi = torch.acos(torch.zeros(1)).item() * 2
    
    base_x = torch.linspace(0,H,H) * pi / H
    base_x = base_x.unsqueeze(0).repeat(H,1)

    base_y = torch.linspace(0,H,H) * pi / H
    base_y = base_y.unsqueeze(1).repeat(1,H)

    base = base_x - base_y
    pn = torch.where(base>0,1,-1)
    
    base =  torch.sqrt(2 * (1 - torch.cos(base))) # H x H 
    base = pn * base
    return (base.unsqueeze(0).unsqueeze(0)).repeat(width * split_size,num_heads,1,1) 

def HRPE(num_heads, height, width, split_size): ## Used for Horizontal attention  / (theta,phi) -> (theta',phi) / This part is coded assuming that the split_size (stripe_with) is 1 to reduce the computational cost. To use larger split_size, ERPE must be calculated accoringly.


    W = width // split_size # Base width
    pi = torch.acos(torch.zeros(1)).item() * 2

    base_x = torch.linspace(0,W,W) *2*pi / W
    base_x = base_x.unsqueeze(0).repeat(W,1)

    base_y = torch.linspace(0,W,W)*2*pi / W
    base_y = base_y.unsqueeze(1).repeat(1,W)
    base = base_x - base_y
    pn = torch.where(base>0,1,-1)
    base = base.unsqueeze(0).repeat(height,1,1)

    for k in range(0,height):
        base[k,:,:] = torch.sin(torch.as_tensor(k*pi/height)) * torch.sqrt(2 * (1 - torch.cos(base[k,:,:]))) # height x W x W  
    
    if True: # Unlike the vertical direction, EIs are cyclic along the horizontal direction. Set to 'False' to reflect this cyclic characteristic / Refer to discussions in repo for more details. 
        base = pn * base
    return base.unsqueeze(1).repeat(split_size,num_heads,1,1) 


# LUT should be updated if input resolution is not 512x1024. 

VRPE_LUT.append(VRPE(1,256,512,1))
HRPE_LUT.append(HRPE(1,256,512,1))

VRPE_LUT.append(VRPE(2,128,256,1))
HRPE_LUT.append(HRPE(2,128,256,1))

VRPE_LUT.append(VRPE(4,64,128,1))
HRPE_LUT.append(HRPE(4,64,128,1))

VRPE_LUT.append(VRPE(8,32,64,1))
HRPE_LUT.append(HRPE(8,32,64,1))

VRPE_LUT.append(VRPE(16,16,32,1))
HRPE_LUT.append(HRPE(16,16,32,1))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EGAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None,attention=0,depth_index=0):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        
        self.scale=1
        self.bias_level = 0.1

        self.sigmoid = nn.Sigmoid()
        self.d_idx = depth_index
        self.idx = idx 
        self.relu = nn.ReLU()
        if attention == 0:
            self.attention = 'L'
        
        if self.attention == 'L':
            # We assume split_size (stripe_with) is 1   
            assert self.split_size == 1, "split_size is not 1" 

            if idx == 0:  # Horizontal Self-Attention
                W_sp, H_sp = self.resolution[1], self.split_size
                self.RPE = HRPE_LUT[self.d_idx]
            elif idx == 1:  # Vertical Self-Attention
                H_sp, W_sp = self.resolution[0], self.split_size
                self.RPE = VRPE_LUT[self.d_idx]
            else:
                print ("ERROR MODE", idx)
                exit(0)


        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.attn_drop = nn.Dropout(attn_drop)

    def im2hvwin(self, x):
        B, N, C = x.shape
        # H = W = int(np.sqrt(N))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()  # B, H//H_sp, W//W_sp, H_sp * W_sp, C -> B, H//H_sp, W//W_sp, H_sp*W_sp, heads, C//heads
        return x

    def get_v(self, x): # LePE is not used for EGformer
        B, N, C = x.shape
        # H = W = int(np.sqrt(N))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, qkv,res_x):
        """
        x: B L C
        """
        q,k,v = qkv[0], qkv[1], qkv[2]
        
        pi = torch.acos(torch.zeros(1)).item() * 2

        ### Img2Window
        # H = W = self.resolution
        H, W = self.resolution[0], self.resolution[1]
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        
        q = self.im2hvwin(q)
        k = self.im2hvwin(k)
        v = self.get_v(v)

        if self.attention == 'L': 
            self.RPE = self.RPE.cuda(q.get_device())
            
            Re = int(q.size(0) / self.RPE.size(0))
    
            attn = q @ k.transpose(-2, -1)
            
            # ERPE
            attn = attn + self.bias_level * self.RPE.repeat(Re,1,1,1)
 
            M = torch.abs(attn) # Importance level of each local attention

            # DAS
            attn = F.normalize(attn,dim=-1) * pi/2
            attn = (1 - torch.cos(attn)) # Square of the distance from the baseline point. By setting the baseline point as (1/sqrt(2),0,pi/2), DAS can get equal score range (0,1) for both vertical & horizontal direction. 

            # EaAR
            M = torch.mean(M,dim=(1,2,3),keepdim=True)   # Check this part to utilize batch size > 1 per GPU.
           
            M = M / torch.max(M)
            M = torch.clamp(M, min=0.5)

            attn = attn * M  

            attn = self.attn_drop(attn)

            x = (attn @ v) 

        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C
        
        # EaAR
        res_x = res_x.reshape(-1,self.H_sp*self.W_sp,C).unsqueeze(1)
        res_x = res_x * (1 - M)
        res_x = res_x.view(B,-1,C)


        return x + res_x


class EGBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False,attention=0,idx=0,depth_index=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        
        self.branch_num = 1
       
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.idx = idx
        self.attention = attention
       
        self.attns = nn.ModuleList([
            EGAttention(
                dim, resolution=self.patches_resolution, idx = self.idx,
                split_size=split_size, num_heads=num_heads, dim_out=dim,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,attention=self.attention, depth_index = depth_index)
            for i in range(self.branch_num)])
        

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        # H = W = self.patches_resolution
        H, W = self.patches_resolution[0], self.patches_resolution[1]
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        
        attened_x = self.attns[0](qkv,x)

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)  # B, H//H_sp, W//W_sp, H_sp * W_sp, C
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class To_BCHW(nn.Module):
    def __init__(self, resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.resolution = resolution
        self.gelu = nn.GELU()
    def forward(self, x):
        B, new_HW, C = x.shape
        # H = W = int(np.sqrt(new_HW))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return x

class Tune_Block_Final(nn.Module):
    def __init__(self, dim, dim_out, resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.norm = norm_layer(dim_out)
        self.resolution = resolution
        self.gelu = nn.GELU()
    def forward(self, x):
        B, new_HW, C = x.shape
#        # H = W = int(np.sqrt(new_HW))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.gelu(x)
        x = self.norm(x)
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return x

class Tune_Block(nn.Module):
    def __init__(self, dim, dim_out, resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.norm = norm_layer(dim_out)
        self.resolution = resolution
        self.gelu = nn.GELU()
    def forward(self, x):
        H, W = self.resolution[0], self.resolution[1]
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.gelu(x)
        x = self.norm(x)
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return x

class Downdim(nn.Module):
    def __init__(self, in_channel, out_channel, reso=None):
        super().__init__()
        self.input_resolution = reso
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, in_resolution, norm_layer=nn.LayerNorm,scale_factor=0.5):
        super().__init__()

        if scale_factor < 1.:
            self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        elif scale_factor > 1.:
            self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        elif scale_factor == 1.:
            self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.scale_factor = scale_factor   
        self.norm = norm_layer(dim_out)
        self.resolution = in_resolution
        self.gelu = nn.GELU()
 
    def forward(self, x):
        B, new_HW, C = x.shape
        # H = W = int(np.sqrt(new_HW))
        H, W = self.resolution[0], self.resolution[1]
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if self.scale_factor >1.:
            x = F.interpolate(x,scale_factor=self.scale_factor)
        x = self.conv(x)
        x = self.gelu(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x

from omegaconf import OmegaConf
from utils.misc import get_device, load_module_weights
from transformers.modeling_outputs import BaseModelOutputWithPooling

@dataclass
class CustomBaseModelOutputWithPooling(BaseModelOutputWithPooling):
    patch_embeddings: Optional[torch.FloatTensor] = None
    
class EGTransformer(nn.Module):
    @dataclass
    class Config:
        img_size: List[int] = field(default_factory=lambda : [512, 1024])
        patch_size: int = 16
        in_chans: int = 3
        num_classes: int = 1000
        embed_dim: int = 32
        egformer_ehv_block_depths: List[int] = field(default_factory=lambda : [2, 2, 2, 2, 2, 2, 2, 2, 2])
        panoformer_pst_bolck_depths: List[int] = field(default_factory=lambda : [2, 2, 2, 2, 2, 2, 2, 2, 2])
        split_size: List[int] = field(default_factory=lambda : [1,1,1,1,1,1,1,1,1])
        panoformer_num_heads: List[int] = field(default_factory=lambda : [1, 2, 4, 8, 16, 16, 8, 4, 2])  # multiheads for each stage of panofromer
        egformer_num_heads: List[int] = field(default_factory=lambda : [1, 2, 4, 8, 8, 4, 2, 1, 16])  # multiheads for each stage of egformer
        mlp_ratio: float = 4.
        token_projection: str = 'linear'  # projection network for tokens in PST block
        token_mlp: str = 'leff'   # feed forward network of PST block
        win_size: int = 8      # local windoow size of PST block
        qkv_bias: bool = True
        qk_scale: Optional[float] = None
        drop_rate: float = 0.
        attn_drop_rate: float = 0.
        drop_path_rate: float = 0.
        hybrid_backbone: Optional[str] = None
        enable_gradient_checkpointing: bool = False
        se_layer: bool = False
    
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
        # embedding layer
        self.patch_embed= nn.Sequential(
            nn.Conv2d(self.cfg.in_chans, self.cfg.embed_dim, kernel_size=3, stride=2, padding=1),
            Rearrange('b c h w -> b (h w) c', h = image_height//2, w = image_width//2),
            nn.LayerNorm(self.cfg.embed_dim)
        )
        self.egformer_ehv_block_depths = self.cfg.egformer_ehv_block_depths
        self.panoformer_pst_block_depths = self.cfg.panoformer_pst_bolck_depths
        self.num_enc_layers = len(self.cfg.panoformer_pst_bolck_depths) // 2
        self.num_dec_layers = len(self.cfg.panoformer_pst_bolck_depths) // 2
        self.num_heads = self.cfg.panoformer_num_heads
        self.egformer_num_heads = self.cfg.egformer_num_heads
        self.embed_dim = self.cfg.embed_dim
        self.norm_layer = nn.LayerNorm

        self.mlp_ratio = self.cfg.mlp_ratio
        self.token_projection = self.cfg.token_projection
        self.token_feed_forward = self.cfg.token_mlp
        self.win_size = self.cfg.win_size
        self.ref_point256x512 = genSamplingPattern(h=256, w=512, kh=3, kw=3).cuda()
        self.ref_point128x256 = genSamplingPattern(h=128, w=256, kh=3, kw=3).cuda()
        self.ref_point64x128 = genSamplingPattern(h=64, w=128, kh=3, kw=3).cuda()
        self.ref_point32x64 = genSamplingPattern(h=32, w=64, kh=3, kw=3).cuda()
        self.ref_point16x32 = genSamplingPattern(h=16, w=32, kh=3, kw=3).cuda()
        
        # panoformer PST block depth decay
        panoformer_encoder_depth_decay_rates = [x.item() for x in torch.linspace(0, self.cfg.drop_path_rate, sum(self.panoformer_pst_block_depths[:self.num_enc_layers]))]
        panoformer_decoder_depth_decay_rates = panoformer_encoder_depth_decay_rates[::-1]
        # EGformer EHV block depth decay
        depth_decay_rates = [x.item() for x in torch.linspace(0, self.cfg.drop_path_rate, np.sum(self.egformer_ehv_block_depths))]  # stochastic depth decay rule
        curr_dim = self.embed_dim

        # encoder stages:
        # STAGE 1: PST block + EV block
        self.stage1 = nn.ModuleList([
            BasicPanoformerLayer(dim=curr_dim, output_dim=curr_dim,
                                                input_resolution=(image_height//2, image_width//2),
                                                depth=self.panoformer_pst_block_depths[0],
                                                num_heads=self.num_heads[0],
                                                win_size=self.win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=self.cfg.qkv_bias, 
                                                qk_scale=self.cfg.qk_scale,
                                                drop=self.cfg.drop_rate, 
                                                attn_drop=self.cfg.attn_drop_rate,
                                                drop_path=panoformer_encoder_depth_decay_rates[int(sum(self.panoformer_pst_block_depths[:0])):int(sum(self.panoformer_pst_block_depths[:1]))],
                                                norm_layer=self.norm_layer,
                                                use_checkpoint=self.cfg.enable_gradient_checkpointing,
                                                token_projection=self.token_projection, 
                                                token_mlp=self.token_feed_forward,
                                                se_layer=self.cfg.se_layer, 
                                                ref_point=self.ref_point256x512, flag = 0) if i%2==0 else 
            EGBlock(
                dim=curr_dim, 
                num_heads=self.egformer_num_heads[0], 
                reso=[image_height//2, image_width//2], 
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.cfg.qkv_bias, 
                qk_scale=self.cfg.qk_scale, 
                split_size=self.cfg.split_size[0],
                drop=self.cfg.drop_rate, 
                attn_drop=self.cfg.attn_drop_rate,
                drop_path=depth_decay_rates[i], 
                norm_layer=self.norm_layer,
                attention=0,
                idx=i%2, 
                depth_index=0)
            for i in range(self.egformer_ehv_block_depths[0])])
        self.downsample1 = Merge_Block(dim=curr_dim, dim_out=curr_dim *2 , in_resolution=[image_height//2, image_width//2])
       
        # STAGE 2: PST block + EV block
        curr_dim = curr_dim*2
        self.stage2 = nn.ModuleList([
            BasicPanoformerLayer(dim=curr_dim, output_dim=curr_dim,
                                                input_resolution=(image_height//4, image_width//4),
                                                depth=self.panoformer_pst_block_depths[1],
                                                num_heads=self.num_heads[1],
                                                win_size=self.win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=self.cfg.qkv_bias,
                                                qk_scale=self.cfg.qk_scale,
                                                drop=self.cfg.drop_rate, 
                                                attn_drop=self.cfg.attn_drop_rate,
                                                drop_path=panoformer_encoder_depth_decay_rates[sum(self.panoformer_pst_block_depths[:1]):sum(self.panoformer_pst_block_depths[:2])],
                                                norm_layer=self.norm_layer,
                                                use_checkpoint=self.cfg.enable_gradient_checkpointing,
                                                token_projection=self.token_projection, 
                                                token_mlp=self.token_feed_forward,
                                                se_layer=self.cfg.se_layer, ref_point=self.ref_point128x256, flag = 0) if i%2==0 else 
            EGBlock(
                dim=curr_dim, 
                num_heads=self.egformer_num_heads[1], 
                reso=[image_height//4, image_width//4], 
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.cfg.qkv_bias, 
                qk_scale=self.cfg.qk_scale, 
                split_size=self.cfg.split_size[1],
                drop=self.cfg.drop_rate, 
                attn_drop=self.cfg.attn_drop_rate,
                drop_path=depth_decay_rates[np.sum(self.egformer_ehv_block_depths[:1])+i], 
                norm_layer=self.norm_layer,
                attention=0, 
                idx=i%2, 
                depth_index=1)
            for i in range(self.egformer_ehv_block_depths[1])])
        self.downsample2 = Merge_Block(dim=curr_dim, dim_out=curr_dim*2, in_resolution=[image_height//4, image_width//4])
       
        # STAGE 3: EH block + EV block
        curr_dim = curr_dim*2
        self.stage3 = nn.ModuleList([
            EGBlock(
                dim=curr_dim, 
                num_heads=self.egformer_num_heads[2], 
                reso=[image_height//8, image_width//8], 
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.cfg.qkv_bias, 
                qk_scale=self.cfg.qk_scale, 
                split_size=self.cfg.split_size[2],
                drop=self.cfg.drop_rate, 
                attn_drop=self.cfg.attn_drop_rate,
                drop_path=depth_decay_rates[np.sum(self.egformer_ehv_block_depths[:2])+i], 
                norm_layer=self.norm_layer,
                attention=0, 
                idx=i%2, 
                depth_index=2)
            for i in range(self.egformer_ehv_block_depths[2])])

        self.downsample3 = Merge_Block(dim=curr_dim, dim_out=curr_dim*2, in_resolution=[image_height//8, image_width//8])

        # STAGE 4: EH block + EV block
        curr_dim = curr_dim*2
        self.stage4 = nn.ModuleList([
            EGBlock(
                dim=curr_dim, 
                num_heads=self.egformer_num_heads[3], 
                reso=[image_height//16, image_width//16], 
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.cfg.qkv_bias, 
                qk_scale=self.cfg.qk_scale, 
                split_size=self.cfg.split_size[3],
                drop=self.cfg.drop_rate, 
                attn_drop=self.cfg.attn_drop_rate,
                drop_path=depth_decay_rates[np.sum(self.egformer_ehv_block_depths[:3])+i], 
                norm_layer=self.norm_layer, 
                last_stage=False,
                attention=0, 
                idx=i%2, 
                depth_index=3)
            for i in range(self.egformer_ehv_block_depths[3])])
 
        self.downsample4 = Merge_Block(dim=curr_dim, dim_out=curr_dim * 2, in_resolution=[image_height//16, image_width//16])
        
        # STAGE 5: bottle_neck  EH block + EV block
        curr_dim = curr_dim*2
        self.bottleneck = nn.ModuleList([
            EGBlock(
                dim=curr_dim, 
                num_heads=self.egformer_num_heads[-1], 
                reso=[image_height//32, image_width//32], 
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.cfg.qkv_bias, 
                qk_scale=self.cfg.qk_scale, 
                split_size=self.cfg.split_size[-1],
                drop=self.cfg.drop_rate, 
                attn_drop=self.cfg.attn_drop_rate,
                drop_path=depth_decay_rates[np.sum(self.egformer_ehv_block_depths[:-1])+i], 
                norm_layer=self.norm_layer, 
                last_stage=False,
                attention = 0, 
                idx= i%2, 
                depth_index = 4)
            for i in range(self.egformer_ehv_block_depths[-1])])
 
        # decoder stages
        self.red_ch = []
        self.set_dim = []
        self.rearrange = []
        # STAGE 5: EH block + EV block
        self.upsample5 = Merge_Block(curr_dim, curr_dim // 2, in_resolution=[image_height//32, image_width//32],scale_factor=2.)
        curr_dim = curr_dim // 2
        self.dec_stage5 = nn.ModuleList(
            [EGBlock(
                dim=curr_dim, 
                num_heads=self.egformer_num_heads[4], 
                reso=[image_height//16, image_width//16], 
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.cfg.qkv_bias, 
                qk_scale=self.cfg.qk_scale, 
                split_size=self.cfg.split_size[4],
                drop=self.cfg.drop_rate, 
                attn_drop=self.cfg.attn_drop_rate,
                drop_path=depth_decay_rates[np.sum(self.egformer_ehv_block_depths[:4])+i], 
                norm_layer=self.norm_layer, 
                last_stage=False, 
                attention = 0, 
                idx= i%2, 
                depth_index = 3)
            for i in range(self.egformer_ehv_block_depths[4])])
        self.tune5 = Tune_Block(dim=curr_dim * 2, dim_out=curr_dim, resolution = [image_height//16, image_width//16]) # Tune_5
        self.set_dim.append(To_BCHW(resolution = [image_height//16, image_width//16])) # BCHW_5
        self.rearrange.append(Rearrange('b c h w -> b (h w) c', h = image_height//16, w = image_width//16))

        # STAGE 6: EH block + EV block
        self.upsample6 = Merge_Block(dim=curr_dim, dim_out=curr_dim // 2, in_resolution=[image_height//16, image_width//16],scale_factor=2.)
        curr_dim = curr_dim // 2
        self.dec_stage6 = nn.ModuleList(
            [EGBlock(
                dim=curr_dim, 
                num_heads=self.egformer_num_heads[5], 
                reso=[image_height//8, image_width//8], 
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.cfg.qkv_bias, 
                qk_scale=self.cfg.qk_scale, 
                split_size=self.cfg.split_size[5],
                drop=self.cfg.drop_rate, 
                attn_drop=self.cfg.attn_drop_rate,
                drop_path=depth_decay_rates[np.sum(self.egformer_ehv_block_depths[:5])+i], 
                norm_layer=self.norm_layer,
                attention = 0, 
                idx= i%2, 
                depth_index = 2)
            for i in range(self.egformer_ehv_block_depths[5])]) 
        self.tune6 = Tune_Block(curr_dim * 2  ,curr_dim, resolution = [image_height//8, image_width//8]) # Tune_6
        self.set_dim.append(To_BCHW(resolution = [image_height//8, image_width//8])) # BCHW_6
        self.rearrange.append(Rearrange('b c h w -> b (h w) c', h = image_height//8, w = image_width//8))
        
        # STAGE 7: PST block + EV block
        self.upsample7 = Merge_Block(curr_dim , curr_dim //2, in_resolution=[image_height//8, image_width//8],scale_factor=2.)
        curr_dim = curr_dim // 2
        self.dec_stage7 = nn.ModuleList([
            BasicPanoformerLayer(dim=curr_dim, output_dim=curr_dim,
                                                input_resolution=(image_height//4, image_width//4),
                                                depth=self.panoformer_pst_block_depths[7],
                                                num_heads=self.num_heads[7],
                                                win_size=self.win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=self.cfg.qkv_bias, 
                                                qk_scale=self.cfg.qk_scale,
                                                drop=self.cfg.drop_rate, 
                                                attn_drop=self.cfg.attn_drop_rate,
                                                drop_path=panoformer_decoder_depth_decay_rates[sum(self.panoformer_pst_block_depths[5:7]):sum(self.panoformer_pst_block_depths[5:8])],
                                                norm_layer=self.norm_layer,
                                                use_checkpoint=self.cfg.enable_gradient_checkpointing,
                                                token_projection=self.token_projection, 
                                                token_mlp=self.token_feed_forward,
                                                se_layer=self.cfg.se_layer,
                                                ref_point=self.ref_point128x256, flag = 1) if i%2==0 else
            EGBlock(
                dim=curr_dim, 
                num_heads=self.egformer_num_heads[6], 
                reso=[image_height//4, image_width//4], 
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.cfg.qkv_bias, 
                qk_scale=self.cfg.qk_scale, 
                split_size=self.cfg.split_size[6],
                drop=self.cfg.drop_rate, 
                attn_drop=self.cfg.attn_drop_rate,
                drop_path=depth_decay_rates[np.sum(self.egformer_ehv_block_depths[:6])+i], 
                norm_layer=self.norm_layer,
                attention=0, 
                idx= i%2, 
                depth_index = 1)
            for i in range(self.egformer_ehv_block_depths[6])])        
        self.tune7 = Tune_Block(curr_dim * 2  ,curr_dim, resolution = [image_height//4, image_width//4]) # Tune_7
        self.set_dim.append(To_BCHW(resolution = [image_height//4, image_width//4])) # BCHW_7
        self.rearrange.append(Rearrange('b c h w -> b (h w) c', h = image_height//4, w = image_width//4))

        # STAGE 8: PST block + EV block
        self.upsample8 = Merge_Block(curr_dim , curr_dim//2, in_resolution=[image_height//4, image_width//4],scale_factor=2.)
        curr_dim = curr_dim // 2
        self.dec_stage8 = nn.ModuleList([
            BasicPanoformerLayer(dim=curr_dim, output_dim=curr_dim,
                                                input_resolution=(image_height//2, image_width//2),
                                                depth=self.panoformer_pst_block_depths[8],
                                                num_heads=self.num_heads[8],
                                                win_size=self.win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=self.cfg.qkv_bias, 
                                                qk_scale=self.cfg.qk_scale,
                                                drop=self.cfg.drop_rate, 
                                                attn_drop=self.cfg.attn_drop_rate,
                                                drop_path=panoformer_decoder_depth_decay_rates[sum(self.panoformer_pst_block_depths[5:8]):sum(self.panoformer_pst_block_depths[5:9])],
                                                norm_layer=self.norm_layer,
                                                use_checkpoint=self.cfg.enable_gradient_checkpointing,
                                                token_projection=self.token_projection, 
                                                token_mlp=self.token_feed_forward,
                                                se_layer=self.cfg.se_layer, 
                                                ref_point=self.ref_point256x512, flag = 1) if i%2==0 else
            EGBlock(
                dim=curr_dim, 
                num_heads=self.egformer_num_heads[7], 
                reso=[image_height//2, image_width//2], 
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.cfg.qkv_bias, 
                qk_scale=self.cfg.qk_scale, 
                split_size=self.cfg.split_size[7],
                drop=self.cfg.drop_rate, 
                attn_drop=self.cfg.attn_drop_rate,
                drop_path=depth_decay_rates[np.sum(self.egformer_ehv_block_depths[:7])+i], 
                norm_layer=self.norm_layer,
                attention=0, 
                idx= i%2, 
                depth_index = 0)
            for i in range(self.egformer_ehv_block_depths[7])])

        
        self.tune8 = Tune_Block(curr_dim * 2  ,curr_dim, resolution=[image_height//2, image_width//2]) # Tune_8
        self.set_dim.append(To_BCHW(resolution = [image_height//2, image_width//2])) # BCHW_8
        self.rearrange.append(Rearrange('b c h w -> b (h w) c', h = image_height//2, w = image_width//2))
    
        self.tune_final = Tune_Block_Final(curr_dim,curr_dim, resolution = [image_height//2, image_width//2])

        # # final normalization layer
        self.norm = self.norm_layer(curr_dim)
        # final
        # self.depth_pred_head = nn.Sequential(
        #     nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        #     nn.Conv2d(in_channels=curr_dim, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0),
        #     # nn.Sigmoid() if non_negative else nn.Identity(),
        #     nn.Sigmoid(),
        # )

        self.apply(self._init_weights)
        
    def __init__(self) -> None:
        super().__init__()
        self.configure()
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_input_embeddings(self):
        """
        This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
        `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
        Transformer.
        """
        return self.embeddings
    
    def forward_features(self, x):
        features= []
        output_latent_features = []
    ########## Encoder       
        self.embeddings = self.patch_embed(x)
        # print(f'patch_embed x.shape: {self.embeddings.shape}')
        
        for blk in self.stage1:
            x = blk(self.embeddings)
        features.append(x)
        # print(f'stage1 x.shape: {x.shape}')
        x = self.downsample1(x)
        # print(f'downsample1 x.shape: {x.shape}')

        for blk in self.stage2:
            x = blk(x)
        features.append(x)
        # print(f'stage2 x.shape: {x.shape}')
        x = self.downsample2(x)
        # print(f'downsample2 x.shape: {x.shape}')

        for blk in self.stage3:
            x = blk(x)
        features.append(x)
        # print(f'stage3 x.shape: {x.shape}')
        x = self.downsample3(x)
        # print(f'downsample3 x.shape: {x.shape}')
        
        for blk in self.stage4:
            x = blk(x)
        features.append(x)
        # print(f'stage4 x.shape: {x.shape}')
        x = self.downsample4(x)
        # print(f'downsample4 x.shape: {x.shape}')
        
        for blk in self.bottleneck:
            x = blk(x)
        # print(f'bottleneck x.shape: {x.shape}')

    ######## Decoder
        x = self.upsample5(x)
        # print(f'upsample5 x.shape: {x.shape}')
        for blk in self.dec_stage5:
            x = blk(x)
        # print(f'stage5 x.shape: {x.shape}')
        x = torch.cat((self.set_dim[0](features[3]), self.set_dim[0](x)), dim=1)
        x = self.tune5(x)
        output_latent_features.append(x)
        x = self.rearrange[0](x)
        # print(f'rearrange[0] x.shape: {x.shape}')

        x = self.upsample6(x)
        # print(f'upsample6 x.shape: {x.shape}')
        for blk in self.dec_stage6:
            x = blk(x)
        # print(f'stage6 x.shape: {x.shape}')
        x = torch.cat((self.set_dim[1](features[2]), self.set_dim[1](x)), dim=1)
        x = self.tune6(x)
        output_latent_features.append(x)
        x = self.rearrange[1](x)
        # print(f'rearrange[1] x.shape: {x.shape}')

        x = self.upsample7(x)
        # print(f'upsample7 x.shape: {x.shape}')
        for blk in self.dec_stage7:
            x = blk(x)
        # print(f'stage7 x.shape: {x.shape}')
        x = torch.cat((self.set_dim[2](features[1]), self.set_dim[2](x)), dim=1)
        x = self.tune7(x)
        output_latent_features.append(x)
        x = self.rearrange[2](x)
        # print(f'rearrange[2] x.shape: {x.shape}')

        x = self.upsample8(x)
        # print(f'upsample8 x.shape: {x.shape}')
        for blk in self.dec_stage8:
            x = blk(x)
        # print(f'stage8 x.shape: {x.shape}')
        x = torch.cat((self.set_dim[3](features[0]), self.set_dim[3](x)), dim=1)
        x = self.tune8(x)
        output_latent_features.append(x)
        x = self.rearrange[3](x)
        # print(f'rearrange[3] x.shape: {x.shape}')
        
        # EGformer Output Projection
        x = self.tune_final(x)
        # return x

        return CustomBaseModelOutputWithPooling(
                    last_hidden_state=x,
                    pooler_output=None,
                    hidden_states=tuple(output_latent_features),
                    attentions=None,
                    patch_embeddings=self.embeddings,
                )

    def forward(self, 
                pixel_values: Optional[torch.Tensor] = None,
                modulation_cond: Optional[torch.Tensor] = None,):
        out = self.forward_features(pixel_values)
        return out

    # def get_encoder(self):
    #     return nn.ModuleList([self.patch_embed, self.stage1, self.stage2, self.stage3, self.stage4, self.bottleneck])
    
    # def get_decoder(self):
    #     return nn.ModuleList([self.dec_stage5, self.dec_stage6, self.dec_stage7, self.dec_stage8])
