import numpy as np
import torch
import os
import wandb
from tqdm import tqdm
from copy import deepcopy

from unic.config import get_obj_from_str
from unic.utils.io import save_deformation

class Trainer():
    def __init__(
        self,
        cfg,
        model,
        dataloader,
        logger,
        device,
        phase="train",
        time_stamp=""
    ):
        self.cfg = cfg
        self.phase = phase
        self.rt_config = cfg.TRAIN if self.phase == "train" else cfg.TEST
        self.device = device
        self.dataloader = dataloader
        self.logger = logger
        self.time_stamp = time_stamp

        if phase == "train":
            self.model = model.module if self.rt_config.USE_DDP else model
            self.optimizer, self.scheduler = self.configure_optimizers()
            if self.rt_config.RESUME:
                self.resume_trainer_state()
            self.train_loss_verbose = {}
        elif phase == "test":
            self.model = model

    def configure_optimizers(self):
        # optimizer
        optim_target = 'torch.optim.' + self.rt_config.OPTIM.target
        optimizer = get_obj_from_str(optim_target)(
            params=self.model.parameters(),
            **self.rt_config.OPTIM.params
        )

        # scheduler
        scheduler_target = 'torch.optim.lr_scheduler.' + self.rt_config.LR_SCHEDULER.target
        lr_scheduler = get_obj_from_str(scheduler_target)(
            optimizer=optimizer,
            **self.rt_config.LR_SCHEDULER.params
        )

        return optimizer, lr_scheduler
    
    def load_pretrained(self):
        ckpt_path = self.rt_config.RESUME_MILESTONE if self.phase == "train" else self.rt_config.CHECKPOINTS
        if self.logger != None:
            self.logger.info(f"Load pretrain model from {ckpt_path}.")

        milestone = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        self.model.load_state_dict(milestone["model"], strict=True)
        
        return milestone
    
    def resume_trainer_state(self):
        milestone = self.load_pretrained()
        if self.rt_config.RESUME_OPTIMIZER:
            self.optimizer.load_state_dict(milestone["optimizer"])
        if self.rt_config.RESUME_SCHEDULER:
            self.scheduler.load_state_dict(milestone["scheduler"])
        self.milestone = milestone
    
    def update_loss_verbose(self, loss_verbose, batch_len):
        for key in loss_verbose:
            if key in self.train_loss_verbose:
                self.train_loss_verbose[key] += loss_verbose[key] / batch_len
            else:
                self.train_loss_verbose[key] = loss_verbose[key] / batch_len

    def loss_reduce(self):
        loss_reduced = {}
        for loss_name, loss_item in self.train_loss_verbose.items():
            if self.rt_config.USE_DDP:
                loss_tensor = torch.tensor(loss_item).float().to(self.device)
                torch.distributed.all_reduce(loss_tensor)
                mean_loss = loss_tensor.item() / torch.distributed.get_world_size() / len(self.dataloader)
            else:
                mean_loss = loss_item / len(self.dataloader)
            loss_reduced[loss_name] = mean_loss
            
        # reset verbose training loss info
        self.train_loss_verbose = {}

        return loss_reduced
    
    def log_train_loss(self, rank: int, loss_reduced: dict):
        if rank == 0 and (not self.cfg.DEBUG):
            wandb.log(loss_reduced)

    def save_onnx_model(self, path):
        c_d = self.cfg.MODEL.params.encoder_params.character_dim
        n_v = self.cfg.DATASET.NUM_VERTEX
        c_v = self.cfg.DATASET.NUM_CHARACTER_VERTEX
        torch.onnx.export(
            self.model.cpu(),
            (
                {
                    "Ct-1": torch.zeros(1, c_d),
                    "Ct": torch.zeros(1, c_d),
                    "Gt-1": torch.zeros(1, n_v, 3),
                    "Gt": torch.zeros(1, n_v*3),
                    "Dt": torch.zeros(1, n_v, 7)
                    # "CGt": {
                    #     "geometry": torch.zeros(1, c_v, 3),
                    #     "normal": torch.zeros(1, c_v, 3),
                    # }
                },
                {}
            ),
            path,
            training=torch.onnx.TrainingMode.EVAL,
            export_params=True,
            opset_version=12,
            do_constant_folding=False,
            input_names = ["batch_data"],
            output_names = ["garm_geometry_pred"]
        )
        print(f"ONNX model saved at {path}...")

    def save_states(self, rank: int, epoch: int):
        if rank != 0:
            return
        
        if epoch % self.rt_config.SAVE_PER == 0:
            # save pytorch model
            milestone = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
            }
            save_dir = os.path.join(self.rt_config.CHECKPOINT_SAVE, self.cfg.EXP_NAME+ '-' + self.time_stamp) 
            os.makedirs(save_dir, exist_ok=True)
            torch.save(milestone, os.path.join(save_dir, f"epoch{epoch}.pth"))

    def data_to_device(self, data):
        batch = deepcopy(data)
        batch["character"]["motion_state"] = batch["character"]["motion_state"].to(self.device)
        for k in batch["character"].keys():
            if k == "topology_v" or k == "topology_f" or k == "neighbors":
                batch["character"][k] = batch["character"][k].to(self.device)
        for k in batch["garment"].keys():
            if k == "deformation":
                for kd in batch["garment"]["deformation"].keys():
                    batch["garment"]["deformation"][kd] = batch["garment"]["deformation"][kd].to(self.device)
            else:
                batch["garment"][k] = batch["garment"][k].to(self.device)

        return batch
    
    def train_steps(self, batches, epoch):
        for i in range(len(batches)):
            batch = batches[i]

            self.optimizer.zero_grad()
            loss, verbose = self.model.train_forward(self.data_to_device(batch), epoch)
            loss.backward()
            self.optimizer.step()
            self.update_loss_verbose(verbose, len(batches))

    def train(self, rank: int):
        start_epoch = 0 if not self.rt_config.RESUME else self.milestone["epoch"]
        train_loop = tqdm(range(start_epoch, self.rt_config.END_EPOCH), disable=rank!=0)
        for epoch in train_loop:
            train_loop.set_description(f'Epoch: {epoch}/{self.rt_config.END_EPOCH}')
            
            if self.rt_config.USE_DDP:
                self.dataloader.sampler.set_epoch(epoch)
            data_loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False, disable=rank!=0)
            for i, batches in data_loop:
                self.train_steps(batches, epoch)
            
            # reduce training loss and log in wandb
            loss_reduced = self.loss_reduce()
            self.log_train_loss(rank, loss_reduced)
            
            # save model states
            self.scheduler.step()
            self.save_states(rank, epoch)
    
    def test(self):
        # load checkpoint
        self.load_pretrained()

        first_clip, _ = self.dataloader[0]
        batch = {
            "character": {
                "motion_state": [],
                "mesh": {
                    "geometry":[],
                    "normal": []
                },
                "joints_of_feet": first_clip["character"]["joints_of_feet"],
                "mask_for_normalization": first_clip["character"]["mask_for_normalization"],
                "topology_v": first_clip["character"]["topology_v"],
                "topology_f": first_clip["character"]["topology_f"],
                "uv_v": first_clip["character"]["uv_v"],
                "uv_f": first_clip["character"]["uv_f"],
                "neighbors": first_clip["character"]["neighbors"],
            },
            "garment": {
                "deformation": {
                    "geometry": [],
                    "velocity": [],
                    "normal": [],
                    "normal_change": []
                },
                "topology_v": first_clip["garment"]["topology_v"],
                "topology_f": first_clip["garment"]["topology_f"],
                "uv_v": first_clip["garment"]["uv_v"],
                "uv_f": first_clip["garment"]["uv_f"]
            }
        }
        seq_transl = []
        start_clip = self.rt_config.START_CLIP; end_clip = self.rt_config.END_CLIP if self.rt_config.END_CLIP != -1 else len(self.dataloader)
        for i in tqdm(range(start_clip, end_clip), desc="loading test sequence clips..."):
            clip_data, clip_transl = self.dataloader[i]
            batch["character"]["motion_state"].append(clip_data["character"]["motion_state"])
            for k in batch["character"]["mesh"].keys():
                batch["character"]["mesh"][k].append(clip_data["character"]["mesh"][k])
            for k in batch["garment"]["deformation"].keys():
                batch["garment"]["deformation"][k].append(clip_data["garment"]["deformation"][k])
            seq_transl.append(clip_transl)
        batch["character"]["motion_state"] = torch.cat(batch["character"]["motion_state"], dim=0)
        for k in batch["character"]["mesh"].keys():
            batch["character"]["mesh"][k] = np.concatenate(batch["character"]["mesh"][k], axis=0)
        for k in batch["garment"]["deformation"].keys():
            batch["garment"]["deformation"][k] = torch.cat(batch["garment"]["deformation"][k], dim=0)
        seq_transl = np.concatenate(seq_transl, axis=0)

        # load test data, batch size is 1
        batch["character"]["motion_state"] = batch["character"]["motion_state"].unsqueeze(0)
        for k in batch["character"]["mesh"].keys():
            batch["character"]["mesh"][k] = np.expand_dims(batch["character"]["mesh"][k], axis=0)
        for k in batch["garment"]["deformation"].keys():
            batch["garment"]["deformation"][k] = batch["garment"]["deformation"][k].unsqueeze(0)
        batch["transl"] = seq_transl
        
        # test forward
        chunk_size = self.rt_config.CHUNK_SIZE; seq_len = batch["garment"]["deformation"]["geometry"].shape[1]
        num_chunks = seq_len // chunk_size
        rst = []; total_time = 0
        for i in tqdm(range(num_chunks), desc="inference test sequence..."):
            chunk_data = {
                "character": {
                    "motion_state": deepcopy(batch["character"]["motion_state"][:, i*chunk_size:(i+1)*chunk_size, :]),
                    "mesh": {
                        "geometry": deepcopy(batch["character"]["mesh"]["geometry"][:, i*chunk_size:(i+1)*chunk_size, :, :]),
                        "normal": deepcopy(batch["character"]["mesh"]["normal"][:, i*chunk_size:(i+1)*chunk_size, :, :]),
                    },
                    "joints_of_feet": batch["character"]["joints_of_feet"],
                    "mask_for_normalization": batch["character"]["mask_for_normalization"],
                    "topology_v": batch["character"]["topology_v"],
                    "topology_f": batch["character"]["topology_f"],
                    "uv_v": batch["character"]["uv_v"],
                    "uv_f": batch["character"]["uv_f"],
                    "neighbors": batch["character"]["neighbors"],
                },
                "garment": {
                    "deformation":{
                        "geometry": deepcopy(batch["garment"]["deformation"]["geometry"][:, i*chunk_size:(i+1)*chunk_size, :, :]),
                        "velocity": deepcopy(batch["garment"]["deformation"]["velocity"][:, i*chunk_size:(i+1)*chunk_size, :, :]),
                        "normal": deepcopy(batch["garment"]["deformation"]["normal"][:, i*chunk_size:(i+1)*chunk_size, :, :]),
                        "normal_change": deepcopy(batch["garment"]["deformation"]["normal_change"][:, i*chunk_size:(i+1)*chunk_size, :, :]),
                    },
                    "topology_v": batch["garment"]["topology_v"],
                    "topology_f": batch["garment"]["topology_f"],
                    "uv_v": batch["garment"]["uv_v"],
                    "uv_f": batch["garment"]["uv_f"],
                }
            }
            pred, step_time = self.model.test_forward(self.data_to_device(chunk_data), record_time=self.rt_config.RECORD_TIME)
            rst.append(pred)
            
            if self.rt_config.RECORD_TIME and i > 0: total_time += step_time
        
        rst = torch.cat(rst, dim=0)

        if self.rt_config.RECORD_TIME:
            print(f"Average time consuming: {total_time/(num_chunks-1):.4f}s")
            exit()
        
        # save predicted deformation sequence
        save_deformation(
            batch,
            rst,
            save_dir=os.path.join(r'/nas/nas_10/share_folder/zhaochf/gmt/final/saved_objs', self.cfg.EXP_NAME, self.cfg.DATASET.testing_set[0]),
            style=self.cfg.DATASET.STYLE,
            save_obj_files=self.rt_config.SAVE_OBJS,
            save_with_transl=self.rt_config.SAVE_WITH_TRANSL
        )

    def convert_to_onnx(self):
        self.load_pretrained()
        self.save_onnx_model(self.rt_config.CHECKPOINTS.replace('.pth', '.onnx'))