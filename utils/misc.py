import os
import re

import torch
from packaging import version
import cv2

from utils.typing import *

def parse_version(ver: str):
    return version.parse(ver)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(
    path, module_name=None, ignore_modules=None, map_location=None
) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                # print(f'ignore k: {k}')
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            # print(f'load k: {k}, m: {m.group(1)}')
            # state_dict_to_load[m.group(1)] = v
            state_dict_to_load[k] = v

    return state_dict_to_load

# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars, *args, **kwargs):
        if isinstance(vars, list):
            return [wrapper(x, *args, **kwargs) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, *args, **kwargs) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, *args, **kwargs) for k, v in vars.items()}
        else:
            return func(vars, *args, **kwargs)

    return wrapper

@make_recursive_func
def todevice(vars, device="cuda"):
    if isinstance(vars, torch.Tensor):
        return vars.to(device)
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, bool):
        return vars
    elif isinstance(vars, float):
        return vars
    elif isinstance(vars, int):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

def colorize_single_channel_image(image, color_map=cv2.COLORMAP_JET):
    '''
    return numpy data
    '''
    image:Float[Tensor, "1 Ht Wt"]
    image = image.squeeze()
    assert len(image.shape) == 2

    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    if torch.is_tensor(image):
        image = image.cpu().numpy()

    image = image.astype(np.uint8)

    image = cv2.applyColorMap(image, color_map)

    return image