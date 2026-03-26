import numpy as np
import torch
import torch.nn.functional as F
import time

@torch.no_grad()
def k_nearest_vertex(
    knn,
    garment_v: torch.Tensor,
    character_v: np.ndarray
):
    """
    find knn on character body surface for each garment vertex
    """
    T, CV, _ = character_v.shape
    V = garment_v.shape[0] // T
    garment_v = garment_v.view(T, V, 3)
    device = garment_v.device

    _, inds = knn(torch.from_numpy(character_v).float().to(device), garment_v)

    return inds

def drag_to_body_surface(
    garment_v: torch.Tensor,
    seeds: list,
    buffer: float,
    **kwargs
):
    """
    drag garment vertex inside character body to surface
    """
    character_v = kwargs["geometry"]; character_vn = kwargs["normal"]
    T, CV, _ = character_vn.shape
    V = garment_v.shape[0] // T
    garment_v = garment_v.view(T, V, 1, 3)
    device = garment_v.device

    # determine garment vertices inside body
    frames = torch.arange(T).view(T, 1, 1)
    character_vertices = torch.from_numpy(character_v[frames, seeds, :]).float().to(device)
    character_normals = torch.from_numpy(character_vn[frames, seeds, :]).float().to(device)

    inside_body_ids = (F.relu(torch.einsum("tvnl,tvnl->tvn", garment_v-character_vertices, character_normals).neg_()) > 0).nonzero()
    t_ids = inside_body_ids[:, 0]; v_ids = inside_body_ids[:, 1]

    # drag garment vertices inside body to body surface
    garment_v[t_ids, v_ids, :] = garment_v[t_ids, v_ids, :] + (character_vertices[t_ids, v_ids, :] - garment_v[t_ids, v_ids, :]).detach()
    garment_v[t_ids, v_ids, :] = buffer * character_normals[t_ids, v_ids, :] + garment_v[t_ids, v_ids, :]

    return garment_v.view(T*V, -1)