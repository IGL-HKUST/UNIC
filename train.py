import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

from unic.config import parse_args
from unic.data.build_data import build_data
from unic.models.build_model import build_model
from unic.losses.build_monitor import build_monitor
from unic.utils.logger import create_logger, log_info
from unic.utils.setting import *
from unic.trainer import Trainer

USE_DDP = True

def main(rank, world_size):
    cfg = parse_args(phase="train")
    cfg.TRAIN.USE_DDP = USE_DDP

    if USE_DDP:
        ddp_setup(rank, world_size)

    # create logger, get time_stamp
    logger, time_stamp = create_logger(cfg, rank, phase="train")
    log_info(rank, logger, OmegaConf.to_yaml(cfg))

    # set random seed
    seed_everything(cfg.SEED_VALUE)
    log_info(rank, logger, f"Random seed set as {cfg.SEED_VALUE}")

    # gpu device
    device = torch.device("cuda")

    # initialize W&B monitor
    build_monitor(cfg, rank, time_stamp)

    # dataset
    dataloader = build_data(cfg, device)
    log_info(rank, logger, f"datasets module {cfg.DATASET.target} initialized")

    # model
    model = build_model(cfg)
    if USE_DDP:
        model = DDP(model.cuda(), device_ids=[rank], output_device=rank)
    else:
        model = model.cuda()
    log_info(rank, logger, f"model {cfg.MODEL.target} loaded")

    # trainer
    trainer = Trainer(cfg, model, dataloader, logger, device, time_stamp=time_stamp)
    log_info(rank, logger, "Trainer initialized")

    # train the model
    trainer.train(rank)

    # Training ends
    log_info(rank, logger, f"The outputs of this experiment are stored in {cfg.EXP_FOLDER}")
    log_info(rank, logger, "Training ends!")

if __name__ == "__main__":
    if USE_DDP:
        world_size = torch.cuda.device_count()
        assert world_size > 1, "ERROR: only use DDP when the world size is larger than 1."
        mp.spawn(
            main,
            args=(world_size,),
            nprocs=world_size
        )
    else:
        main(0, 1)