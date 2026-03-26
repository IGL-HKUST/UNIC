import torch
import torch.nn as nn
from knn_cuda import KNN
import time

from unic.models.modules import LinearEncoder, LinearDecoder
from unic.losses.losses import UnicLosses
from unic.utils.intersection_handler import *

class UNIC(nn.Module):
    def __init__(
        self,
        encoder_params,
        decoder_params,
        **kwargs
    ):
        super(UNIC, self).__init__()
        
        # configs
        self.cfg = kwargs["cfg"]
        self.phase = kwargs["phase"]

        # encoder network
        self.codebook_c = encoder_params["codebook_channel"]
        self.codebook_d = encoder_params["codebook_dim"]
        self.categorical = encoder_params["categorical"]
        if not self.categorical:
            encoder_params["codebook_dim"] = 1
        
        self.motion_encoder = LinearEncoder(
            2 * encoder_params["character_dim"],
            encoder_params["hidden_dim"],
            encoder_params["hidden_dim"],
            encoder_params["codebook_channel"],
            encoder_params["codebook_dim"],
            encoder_params["dropout"],
            encoder_params["categorical"]
        )

        # decoder network
        self.deform_decoder = LinearDecoder(
            decoder_params["n_layers"],
            decoder_params["input_dim"],
            decoder_params["hidden_dim"],
            decoder_params["output_dim"],
            decoder_params["skips"],
        )

        # losses
        self.losses = UnicLosses(self.cfg)

        # knn for intersection handling
        self.knn = KNN(k=self.cfg.MODEL.intersection.top_k, transpose_mode=True)

        # register character vertex neighbors
        self.is_registered = False
    
    @torch.no_grad()
    def preprocess(self, batch):
        B, seq_len, c_d = batch["character"]["motion_state"].shape
        _, _, V, _ = batch["garment"]["deformation"]["geometry"].shape
        _, _, CV, _ = batch["character"]["mesh"]["geometry"].shape
        
        G = batch["garment"]["deformation"]["geometry"]
        Gt_1 = G[:, :-1, ...].contiguous().view(B*(seq_len-1), V, 3)
        Gt = G[:, 1:, ...].contiguous().view(B*(seq_len-1), V*3)
        Dt = torch.cat(
            [
                batch["garment"]["deformation"]["velocity"],
                batch["garment"]["deformation"]["normal_change"]
            ],
            dim=-1
        )[:, :-1, ...].contiguous().view(B*(seq_len-1), V, 7)

        batch_data = {
            "Ct-1": batch["character"]["motion_state"][:, :-1, :].contiguous().view(B*(seq_len-1), c_d),
            "Ct": batch["character"]["motion_state"][:, 1:, :].contiguous().view(B*(seq_len-1), c_d),
            "Gt-1": Gt_1,
            "Gt": Gt,
            "Dt": Dt,
            "CGt": {
                "geometry": batch["character"]["mesh"]["geometry"][:, 1:, :, :].reshape(B*(seq_len-1), CV, 3),
                "normal": batch["character"]["mesh"]["normal"][:, 1:, :, :].reshape(B*(seq_len-1), CV, 3),
            }
        }

        if not self.is_registered:
            self.register_buffer(
                "neighbors",
                batch["character"]["neighbors"].unsqueeze(0).repeat(seq_len-1, 1, 1),
                persistent=False
            )
            self.is_registered = True

        return batch_data
    
    def encode(self, batch_data, knn: torch.Tensor = torch.ones(1)):
        char_motion = torch.cat([batch_data["Ct"], batch_data["Ct-1"]], dim=1)
        knn = knn.to(char_motion.device)

        # encode character motion
        char_motion_codes, char_motion_probs, char_motion_onehot = self.motion_encoder(char_motion, knn)
        if not self.categorical:
            return char_motion_codes
        else:
            char_motion_sampled = torch.nonzero(char_motion_onehot == 1)
            char_motion_vq = char_motion_codes[char_motion_sampled[:, 0], char_motion_sampled[:, 1]].view(-1, self.codebook_c)
            return char_motion_vq
    
    def decode(self, batch_data, motion_feature):
        garm_geometry_1 = batch_data["Gt-1"]

        # concatenate character motion features with garment geometry
        garm_1 = torch.cat([garm_geometry_1, motion_feature.unsqueeze(1).repeat(1, self.cfg.DATASET.NUM_VERTEX, 1)], dim=-1)

        # predict garment deformation
        deform_pred = self.deform_decoder(garm_1)

        return deform_pred

    def handle_intersection(self, garment_pred: torch.Tensor, **kwargs):
        seeds = k_nearest_vertex(
            self.knn,
            garment_pred,
            kwargs["geometry"]
        )

        dragged = drag_to_body_surface(
            garment_pred,
            seeds.cpu().numpy(),
            buffer=self.cfg.MODEL.intersection.buffer,
            **kwargs
        )

        return dragged
    
    def forward(self, batch_data, epoch):            
        # encoder-decoder forward
        motion_feature = self.encode(batch_data)
        deform_pred = self.decode(batch_data, motion_feature)

        # add incremental deformation to previous garment geometry
        garm_geometry_1 = batch_data["Gt-1"]
        garm_geometry_pred = garm_geometry_1.view(-1, 3) + deform_pred["deform_p"]

        # intersection handling
        if self.phase == "train":
            if epoch >= self.cfg.MODEL.intersection.handle_in_train_after:
                garm_geometry_pred = self.handle_intersection(garm_geometry_pred, **batch_data["CGt"])
        elif self.phase == "test":
            if self.cfg.MODEL.intersection.handle_in_test:
                garm_geometry_pred = self.handle_intersection(garm_geometry_pred, **batch_data["CGt"])
        
        return garm_geometry_pred.view(-1, 3)

    def train_forward(self, batch, epoch):
        # preprocess data
        batch_data = self.preprocess(batch)

        # forward pass
        garm_geometry_pred = self.forward(batch_data, epoch)
        rst = {
            "geometry_pred": garm_geometry_pred,
            "geometry_label": batch_data["Gt"].view(-1, 3),
            "geometry_character": batch_data["CGt"]
        }
        
        # calculate loss
        total_loss, verbose = self.losses.update(rst)

        return total_loss, verbose
    
    @torch.no_grad()
    def test_forward(self, batch, record_time=True):
        batch_data = self.preprocess(batch)
        
        if record_time:
            torch.cuda.synchronize()
            start_time = time.time()
            garm_geometry_pred = self.forward(batch_data, None)
            torch.cuda.synchronize()
            duration = time.time() - start_time
        else:
            garm_geometry_pred = self.forward(batch_data, None)
            duration = None
        garm_geometry_pred = garm_geometry_pred.view(-1, self.cfg.DATASET.NUM_VERTEX, 3)
        garm_geometry_pred = torch.cat([batch_data["Gt-1"][0:1], garm_geometry_pred], dim=0)

        return garm_geometry_pred, duration