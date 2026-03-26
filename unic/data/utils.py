import numpy as np
import torch

def collate_fn(batch_data, cfg):
    # crop or extend to cfg.DATASET.SEQ_LEN frames, ignore sequences less than cfg.DATASET.MIN_LEN frames
    T, min_t, V = cfg.DATASET.SEQ_LEN, cfg.DATASET.MIN_LEN, cfg.DATASET.NUM_VERTEX
    collated_motion = []
    collated_mesh = {
        "geometry": [],
        "normal": []
    }
    collated_deform = {
        "geometry": [],
        "velocity": [],
        "normal": [],
        "normal_change": []
    }
    for b in batch_data:
        character = b["character"]; garment = b["garment"]
        device = character["motion_state"].device
        seq_len, motion_dim = character["motion_state"].shape

        if seq_len < min_t:
            continue
        
        if seq_len < T:
            zeros = torch.zeros(T-seq_len, motion_dim).to(device)
            collated_motion.append(torch.cat([character["motion_state"], zeros], dim=0))
            for k in collated_mesh.keys(): 
                zeros = np.zeros((T-seq_len, V, 3))
                collated_mesh[k].append(np.concatenate([character["mesh"][k], zeros], axis=0))
            for k in collated_deform.keys():
                zeros = torch.zeros(T-seq_len, V, 4).to(device) if k == "normal_change" else torch.zeros(T-seq_len, V, 3).to(device)
                collated_deform[k].append(torch.cat([garment["deformation"][k], zeros], dim=0))
        else:
            # seq_len >= T
            chunk_num = seq_len // T
            for i in range(chunk_num):
                collated_motion.append(character["motion_state"][i*T:(i+1)*T])
                for k in collated_mesh.keys():
                    collated_mesh[k].append(character["mesh"][k][i*T:(i+1)*T])
                for k in collated_deform.keys():
                    collated_deform[k].append(garment["deformation"][k][i*T:(i+1)*T])
            
    collated_motion = torch.stack(collated_motion, dim=0)
    for k in collated_mesh.keys():
        collated_mesh[k] = np.stack(collated_mesh[k], axis=0)
    for k in collated_deform.keys():
        collated_deform[k] = torch.stack(collated_deform[k], dim=0)

    data_len = collated_motion.shape[0]
    batch_size = cfg.TRAIN.BATCH_SIZE; batch_num = data_len // batch_size
    batches = []
    for i in range(batch_num):
        batches.append({
            "character": {
                "motion_state": collated_motion[i*batch_size:(i+1)*batch_size],
                "mesh": {
                    "geometry": collated_mesh["geometry"][i*batch_size:(i+1)*batch_size],
                    "normal": collated_mesh["normal"][i*batch_size:(i+1)*batch_size]
                },
                "joints_of_feet": character["joints_of_feet"],
                "mask_for_normalization": character["mask_for_normalization"],
                "topology_v": character["topology_v"],
                "topology_f": character["topology_f"],
                "uv_v": character["uv_v"],
                "uv_f": character["uv_f"],
                "neighbors": character["neighbors"]
            },
            "garment":{
                "deformation": {
                    "geometry": collated_deform["geometry"][i*batch_size:(i+1)*batch_size],
                    "normal": collated_deform["normal"][i*batch_size:(i+1)*batch_size],
                    "velocity": collated_deform["velocity"][i*batch_size:(i+1)*batch_size],
                    "normal_change": collated_deform["normal_change"][i*batch_size:(i+1)*batch_size]
                },
                "topology_v": garment["topology_v"],
                "topology_f": garment["topology_f"],
                "uv_v": garment["uv_v"],
                "uv_f": garment["uv_f"]
            }
        })

    return batches

def normalize(
    to_norm,
    mean_std,
    key: str = None
):
    if key == None:
        mean_std["std"][(mean_std["std"] == 0).nonzero()[0]] = 1.0
        return (to_norm - mean_std["mean"]) / mean_std["std"]
    else:
        std = 1.0 if mean_std["std"][key] == 0 else mean_std["std"][key]
        return (to_norm - mean_std["mean"][key]) / std