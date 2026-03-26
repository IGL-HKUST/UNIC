import wandb

def build_monitor(cfg, rank, time_stamp: str):
    if rank != 0 or cfg.DEBUG:
        return None
    
    config={
        # meta information
        "seed": cfg.SEED_VALUE,

        # model information
        "epochs": cfg.TRAIN.END_EPOCH,
        "optimizer": cfg.TRAIN.OPTIM.target,
        "learning_rate": cfg.TRAIN.OPTIM.params.lr
    }
    
    if cfg.TRAIN.RESUME:
        api = wandb.Api()
        entity = cfg.LOGGER.WANDB.params.entity
        project_name = cfg.LOGGER.WANDB.params.project + '.' + cfg.TRAIN.RESUME_MILESTONE.split('/')[1]
        runs = api.runs(path=entity + '/' + project_name)
        assert len(runs) == 1
        run_id = runs[0].id

        run = wandb.init(
            project=project_name,
            id=run_id,
            resume="must"
        )
    else:
        run = wandb.init(
            project=(cfg.LOGGER.WANDB.params.project + '.' + cfg.EXP_NAME + '-' + time_stamp),
            name=cfg.EXP_NAME,
            config=config
        )