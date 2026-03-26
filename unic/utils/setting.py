import torch
from torch.distributed import init_process_group
import numpy as np
import random
import os

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12351"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def seed_everything(seed: int=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # set CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # set TORCH_CUDA_ARCH
    torch_arch = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{torch_arch[0]}.{torch_arch[1]}"