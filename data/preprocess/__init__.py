import numpy as np
import torch
from pytorch3d.transforms import *
import os
import pickle as pkl
from tqdm import tqdm
from glob import glob
from copy import deepcopy
import multiprocessing

from ..utils.io import *

def calc_normal_change(pid, normal, return_dict):
    a = normal[:-1]; b = normal[1:]
    T, V, _ = a.shape
    v = np.cross(a, b)
    c = np.einsum("tvm,tvm->tv", a, b)

    I = np.eye(3)
    k = np.zeros((T, V, 3, 3))
    r = np.zeros((T, V, 3, 3))
    for t in tqdm(range(T), leave=False, disable=pid!=0, desc="calculating normal change..."):
        for x in range(V):
            k[t, x] = np.array([
                [0, -v[t, x, 2], v[t, x, 1]],
                [v[t, x, 2], 0, -v[t, x, 0]],
                [-v[t, x, 1], v[t, x, 0], 0]
            ])
            r[t, x] = I + k[t, x] + (k[t, x]@k[t, x])*(1/(1+c[t, x]))

    return_dict[pid] = r

def calc_normal_change_parallel(normal, process_number):
    chunk_size = normal.shape[0] // process_number
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict(); jobs = []
    for i in range(process_number):
        if i == process_number - 1:
            chunk = normal[i*chunk_size:]
        else:
            chunk = normal[i*chunk_size:(i+1)*chunk_size+1]
        p = multiprocessing.Process(
            target=calc_normal_change,
            args=(i, chunk, return_dict)
        )
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    return np.concatenate([return_dict[i] for i in range(process_number)], axis=0)

class BaseProcessor():
    def __init__(self, root_path: str):
        self.dataset_name = None
        self.data_root = root_path

    def normalize(
        self,
        garm_deform: dict,
        char_motion: dict,
        char_mesh: dict
    ):
        char_rotation = torch.from_numpy(char_motion["joint_rotation"]).float()
        char_position = torch.from_numpy(char_motion["joint_position"]).float()
        garm_geometry = torch.from_numpy(garm_deform["geometry"]).float()
        garm_normal = torch.from_numpy(garm_deform["normal"]).float()
        char_mesh_geometry = torch.from_numpy(char_mesh["geometry"]).float()
        char_mesh_normal = torch.from_numpy(char_mesh["normal"]).float()
        
        first_global_t = char_position[0, 0, :].view(1, 1, 3)
        first_global_o = char_rotation[0, 0, :].view(1, 4)
        first_global_mat_inv = quaternion_to_matrix(first_global_o).transpose(1, 2)

        # normalize character motion
        normed_rotation = char_rotation.clone()
        normed_position = char_position.clone()
        normed_position -= first_global_t
        normed_position = torch.einsum("mn,tjn->tjm", first_global_mat_inv[0], normed_position)
        root_rotmat = quaternion_to_matrix(char_rotation[:, 0, :])
        normed_rotation[:, 0, :] = matrix_to_quaternion(torch.einsum("mn,tnl->tml", first_global_mat_inv[0], root_rotmat))

        # normalize character mesh
        normed_char_geometry = char_mesh_geometry.clone()
        normed_char_normal = char_mesh_normal.clone()
        normed_char_geometry -= char_position[:, :1, :]
        normed_char_geometry = torch.einsum("mn,tvn->tvm", first_global_mat_inv[0], normed_char_geometry)
        normed_char_normal = torch.einsum("mn,tvn->tvm", first_global_mat_inv[0], normed_char_normal)

        # normalize garment deformation
        # NOTE: normalize garment geometry at each frame to 0-translation
        normed_garm_geometry = garm_geometry.clone()
        normed_garm_normal = garm_normal.clone()
        normed_garm_geometry -= char_position[:, :1, :]
        normed_garm_geometry = torch.einsum("mn,tvn->tvm", first_global_mat_inv[0], normed_garm_geometry)
        normed_garm_normal = torch.einsum("mn,tvn->tvm", first_global_mat_inv[0], normed_garm_normal)
        
        # collection
        normed_deformation = {
            "geometry": normed_garm_geometry.numpy(),
            "normal": normed_garm_normal.numpy()
        }
        normed_motion = {
            "joint_rotation": normed_rotation.numpy(),
            "joint_position": normed_position.numpy()
        }
        normed_mesh = {
            "geometry": normed_char_geometry.numpy(),
            "normal": normed_char_normal.numpy()
        }

        return normed_deformation, normed_motion, normed_mesh

    def foot_detect(self, positions: np.ndarray, thres: float):
        l1, l2, r1, r2 = self.feet_mask
        fid_l, fid_r,  = [l1-1, l2-1], [r1-1, r2-1],  # leftfoot, rightfoot
        velfactor = np.array([thres, thres])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)

        return feet_l, feet_r
    
    def to_char_representation(self, char_motion: dict):
        joint_rotation = char_motion["joint_rotation"]
        joint_position = char_motion["joint_position"]
        T, J, _ = joint_rotation.shape
        eu_mask = []
        
        # root translation, euclidean
        root_trans = joint_position[:, 0, :]
        eu_mask.extend([i for i in range(root_trans.shape[-1])])

        # root linear velocity, euclidean
        root_linear_velo = joint_position[1:, 0, :] - joint_position[:-1, 0, :]
        eu_mask.extend([eu_mask[-1]+1+i for i in range(root_linear_velo.shape[-1])])

        # root orientation
        root_orient_mat = quaternion_to_matrix(torch.from_numpy(joint_rotation[:, 0, :]).float())
        root_orient_6d = matrix_to_rotation_6d(root_orient_mat).numpy()

        # root angular velocity
        root_angular_velo = torch.einsum("tmn,tnl->tml", root_orient_mat[1:], root_orient_mat[:-1].transpose(1, 2))
        root_angular_velo = matrix_to_axis_angle(root_angular_velo).numpy()
        eu_mask.extend([eu_mask[-1]+root_orient_6d.shape[-1]+1+i for i in range(root_angular_velo.shape[-1])])

        # local joint rotation
        local_joint_rot = matrix_to_rotation_6d(
            quaternion_to_matrix(
                torch.from_numpy(joint_rotation[:, 1:, :]).float()
            )
        ).view(-1, (J-1)*6).numpy()

        # local joint position
        local_joint_pos = joint_position[:, 1:, :].reshape(-1, (J-1)*3)
        eu_mask.extend([eu_mask[-1]+local_joint_rot.shape[-1]+1+i for i in range(local_joint_pos.shape[-1])])

        # local joint linear velocity
        local_joint_linear_velo = local_joint_pos[1:] - local_joint_pos[:-1]
        eu_mask.extend([eu_mask[-1]+1+i for i in range(local_joint_linear_velo.shape[-1])])

        # foot-ground contacts
        feet_l, feet_r = self.foot_detect(local_joint_pos.reshape(-1, J-1, 3), 0.002)

        # collection
        representation = np.concatenate(
            [
                root_trans[:-1],
                root_linear_velo,
                root_orient_6d[:-1],
                root_angular_velo,
                local_joint_rot[:-1],
                local_joint_pos[:-1],
                local_joint_linear_velo,
                feet_l,
                feet_r
            ],
            axis=1
        )
        self.char_motion_eu_mask = eu_mask
        
        return representation
    
    def to_garm_representation(self, garm_deform: dict):
        repr = deepcopy(garm_deform)
        geometry = garm_deform["geometry"]; normal = garm_deform["normal"]
        repr.update({
            "velocity": geometry[1:, ...] - geometry[:-1, ...],
            "normal_change": calc_normal_change_parallel(normal, self.args.process_number)
        })

        return repr
        
    def save_mean_std(self):
        seq_dir = [os.path.join(self.data_root, self.dataset_name, 'pre_processed', self.args.cfg.DATASET.STYLE, 'train', x) for x in self.args.cfg.DATASET.training_set]

        char_motion = []
        garment_deform_geo = []; garment_deform_vel = []
        for seq in tqdm(sorted(seq_dir), desc="calculating statistcs..."):
            for clip in tqdm(sorted(os.listdir(seq)), leave=False, desc=f"loading clips of {seq.split('/')[-1]}..."):
                with open(os.path.join(seq, clip), 'rb') as f:
                    clip_data = pkl.load(f)
                char_motion.append(clip_data["character"]["motion_state"])
                garment_deform_geo.append(clip_data["garment"]["deformation"]["geometry"])
                garment_deform_vel.append(clip_data["garment"]["deformation"]["velocity"])
                self.char_motion_eu_mask = clip_data["character"]["mask_for_normalization"]
        char_motion = np.vstack(char_motion)[:, self.char_motion_eu_mask]
        garment_deform_geo = np.vstack(garment_deform_geo)
        garment_deform_vel = np.vstack(garment_deform_vel)

        char_motion_mean= np.mean(char_motion, axis=0); char_motion_std = np.std(char_motion, axis=0)
        garment_deform_geo_mean = np.mean(garment_deform_geo, axis=0); garment_deform_geo_std = np.std(garment_deform_geo, axis=0)
        garment_deform_vel_mean = np.mean(garment_deform_vel, axis=0); garment_deform_vel_std = np.std(garment_deform_vel, axis=0)
        statistics = {
            "character": {
                "mask_for_normalization": self.char_motion_eu_mask,
                "mean": char_motion_mean,
                "std": char_motion_std
            },
            "garment": {
                "mean": {
                    "geometry": garment_deform_geo_mean,
                    "velocity": garment_deform_vel_mean,
                },
                "std": {
                    "geometry": garment_deform_geo_std,
                    "velocity": garment_deform_vel_std,
                }
            },
        }
        with open(os.path.join(self.data_root, self.dataset_name, 'pre_processed', self.args.cfg.DATASET.STYLE, 'mean_std.pkl'), 'wb') as f:
            pkl.dump(statistics, f)

    def read_character_motion(self):
        raise NotImplementedError
    
    def read_garment_template(self):
        raise NotImplementedError

    def read_garment_deform(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError
    
    def prepare_data(self):
        raise NotImplementedError