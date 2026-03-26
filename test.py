import torch
from omegaconf import OmegaConf

from unic.config import parse_args
from unic.data.build_data import build_data
from unic.models.build_model import build_model
from unic.utils.logger import create_logger, log_info
from unic.utils.setting import *
from unic.trainer import Trainer

def main(rank):
    cfg = parse_args(phase="train")

    # create logger, get time_stamp
    logger, time_stamp = create_logger(cfg, rank, phase="test")
    log_info(rank, logger, OmegaConf.to_yaml(cfg))

    # set random seed
    seed_everything(cfg.SEED_VALUE)
    log_info(rank, logger, f"Random seed set as {cfg.SEED_VALUE}")

    # gpu device
    device = torch.device("cuda")

    # dataset
    dataloader = build_data(cfg, device, phase="test")
    log_info(rank, logger, f"datasets module {cfg.DATASET.target} initialized")

    # model
    model = build_model(cfg, phase="test")
    model = model.cuda()
    model.eval()
    log_info(rank, logger, f"model {cfg.MODEL.target} loaded")

    # trainer
    trainer = Trainer(cfg, model, dataloader, logger, device, phase="test", time_stamp=time_stamp)
    log_info(rank, logger, "Trainer initialized")

    # test the model
    if cfg.TEST.CONVERT_ONNX:
        trainer.convert_to_onnx()
    else:
        trainer.test()

    # Training ends
    log_info(rank, logger, f"The outputs of this experiment are stored in {cfg.LOGGER_DIR}")
    log_info(rank, logger, "Testing ends!")

if __name__ == "__main__":
    main(0)