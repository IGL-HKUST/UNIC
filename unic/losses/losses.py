import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

from .base import BaseLosses
    
class UnicLosses(BaseLosses):
    def __init__(self, cfg):
        self.cfg = cfg

        # define loss weights
        losses = ["geometry"]
        weights = {"geometry": cfg.LOSS.LAMBDA_GEOMETRY}

        # define loss functions
        losses_func = {}
        if cfg.LOSS.TYPE == "l2":
            for loss in losses:
                losses_func[loss] = nn.MSELoss

        super().__init__(cfg, losses, weights, losses_func)

    def update(self, rs_set):
        total = 0.0

        # vertex position deformation loss
        geometry_loss = self._update_loss("geometry", rs_set["geometry_pred"], rs_set["geometry_label"])
        total += geometry_loss

        # record verbose loss information
        verbose = {
            "geometry_loss": geometry_loss.item(),
            "total": total.item()
        }

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total, verbose