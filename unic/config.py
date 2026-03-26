import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
from os.path import join as pjoin
import os
import glob


def get_module_config(cfg, filepath="./configs"):
    """
    Load yaml config files from subfolders
    """

    yamls = glob.glob(pjoin(filepath, '*', '*.yaml'))
    yamls = [y.replace(filepath, '') for y in yamls]
    for yaml in yamls:
        nodes = yaml.replace('.yaml', '').replace('/', '.')
        nodes = nodes[1:] if nodes[0] == '.' else nodes
        OmegaConf.update(cfg, nodes, OmegaConf.load('./configs' + yaml))

    return cfg


def get_obj_from_str(string, reload=False):
    """
    Get object from string
    """

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Instantiate object from config
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def resume_config(cfg: OmegaConf):
    """
    Resume model and wandb
    """
    
    if cfg.TRAIN.RESUME:
        resume = cfg.TRAIN.RESUME
        if os.path.exists(resume):
            # Checkpoints
            cfg.TRAIN.PRETRAINED = pjoin(resume, "checkpoints", "last.ckpt")
            # Wandb
            wandb_files = os.listdir(pjoin(resume, "wandb", "latest-run"))
            wandb_run = [item for item in wandb_files if "run-" in item][0]
            cfg.LOGGER.WANDB.params.id = wandb_run.replace("run-","").replace(".wandb", "")
        else:
            raise ValueError("Resume path is not right.")

    return cfg

def parse_args(phase="train"):
    """
    Parse arguments and load config files
    """

    # add arguments
    parser = ArgumentParser()
    group = parser.add_argument_group("Training options")
    group.add_argument(
        "--cfg_assets",
        type=str,
        required=False,
        default="./configs/assets.yaml",
        help="config file for asset paths",
    )
    group.add_argument(
        "--cfg",
        type=str,
        required=False,
        default="./configs/default.yaml",
        help="config file",
    )
    group.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="training batch size"
    )
    group.add_argument(
        "--num_nodes",
        type=int,
        required=False,
        help="number of nodes"
    )
    group.add_argument(
        "--device",
        type=int,
        nargs="+",
        required=False,
        help="training device"
    )
    group.add_argument(
        "--nodebug",
        action="store_true",
        required=False,
        help="debug or not"
    )
    params = parser.parse_args()
    
    # load yaml config files
    OmegaConf.register_new_resolver("eval", eval)
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg_base = OmegaConf.load(pjoin(cfg_assets.CONFIG_FOLDER, 'default.yaml'))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    # specify debug mode
    if phase == "train":
        cfg.DEBUG = not params.nodebug
    else:
        cfg.DEBUG = False

    if cfg.DEBUG:
        cfg.EXP_NAME = "debug--" + cfg.EXP_NAME
        cfg.LOGGER.WANDB.params.offline = True
        cfg.LOGGER.VAL_EVERY_STEPS = 1

    # specify render configs
    if phase == "render":
        if params.npy:
            cfg.RENDER.NPY = params.npy
            cfg.RENDER.INPUT_MODE = "npy"
        if params.dir:
            cfg.RENDER.DIR = params.dir
            cfg.RENDER.INPUT_MODE = "dir"
        if params.fps:
            cfg.RENDER.FPS = float(params.fps)
        cfg.RENDER.MODE = params.mode

    return cfg