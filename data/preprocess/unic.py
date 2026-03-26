from . import BaseProcessor

import numpy as np
import os
import argparse
from tqdm import tqdm
from glob import glob
import pickle as pkl
from omegaconf import OmegaConf
import multiprocessing

from data.utils.io import *
from data.utils.skeletons import *
from data.utils.FbxReadWriter import FbxReadWrite

class UnicProcessor(BaseProcessor):
    def __init__(self, data_root, args):
        super(UnicProcessor, self).__init__(data_root)

        self.dataset_name = "final"

        # skeleton definition
        if args.character == "unity_smpl":
            self.skeleton = UNITY_SMPL_SKELETON
            self.feet_mask = UNITY_SMPL_FEET_JOINT_MASK
            self.ignore_mask = UNITY_SMPL_JOINT_TO_IGNORE
        elif args.character == "mixamo":
            self.skeleton = MIXAMO_CHARACTER_SKELETON
            self.feet_mask = MIXAMO_CHARACTER_FEET_JOINT_MASK
            self.ignore_mask = MIXAMO_CHARACTER_JOINT_TO_IGNORE
        elif args.character == "ue":
            self.skeleton = UE_CHARACTER_SKELETON
            self.feet_mask = UE_CHARACTER_FEET_JOINT_MASK
            self.ignore_mask = UE_CHARACTER_JOINT_TO_IGNORE
        
        # scale skeleton to meters
        self.scale = 1.
        if args.scale == "mm":
            self.scale /= 1000
        elif args.scale == "cm":
            self.scale /= 100

        self.args = args
    
    def read_character_motion(self, filename, seq_len):
        fbx_object = FbxReadWrite(filename, self.skeleton, self.args)
        character_motion = fbx_object.load_animation(seq_len, fps=self.args.fps, skip=self.args.skip_morph)
        character_motion["joint_position"] *= self.scale
        
        return character_motion

    def read_mesh_template(self, filename):
        template = read_topology(filename, self.args.encoding)

        return {
            "topology_v": template[0],
            "topology_f": template[1],
            "uv_v": template[2],
            "uv_f": template[3]
        }
    
    def read_vertex_neighbors(self, filename):
        neighbors = read_topology(filename, self.args.encoding)[-1]
        max_num = -1
        for k,v in neighbors.items():
            neighbors[k] = list(set(v)) + [k]   # add itself
            if len(neighbors[k]) > max_num:
                max_num = len(neighbors[k])
            
        mapping = np.zeros((len(neighbors), max_num))
        for k,v in neighbors.items():
            mapping[k] = np.array(neighbors[k] + [neighbors[k][-1] for _ in range(max_num-len(v))])

        return mapping

    def read_mesh_deform(self, pid, chunk, return_dict):
        deformation = {
            "geometry": [],
            "normal": []
        }
        
        for i in tqdm(range(len(chunk)), leave=False, disable=pid!=0, desc=f"pre-processing {chunk[0].split('/')[-3]} mesh..."):
            # read single frame geometry
            frame = chunk[i]
            V, F, Vt, Ft, Vn, Vni, _ = read_topology(frame, self.args.encoding)
            deformation["geometry"].append(V)

            # localize vertex normals through faces
            deform_normal = np.zeros_like(V)
            for n in range(F.shape[0]):
                deform_normal[F[n]] = Vn[Vni[n]]
            deformation["normal"].append(deform_normal)
        deformation["geometry"] = np.stack(deformation["geometry"], axis=0)
        deformation["normal"] = np.stack(deformation["normal"], axis=0)

        return_dict[pid] = deformation
    
    def read_mesh_deform_parallel(self, dir):
        frames = sorted(glob(os.path.join(dir, '*.obj')), key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        # seq_len = 32
        seq_len = len(frames) - self.args.skip_morph
        chunk_size = seq_len // self.args.process_number

        manager = multiprocessing.Manager()
        return_dict = manager.dict(); jobs = []
        for i in range(self.args.process_number):
            if i == self.args.process_number-1:
                # chunk = frames[self.args.skip_morph+i*chunk_size: self.args.skip_morph+(i+1)*chunk_size]
                chunk = frames[self.args.skip_morph+i*chunk_size:]
            else:
                chunk = frames[self.args.skip_morph+i*chunk_size:self.args.skip_morph+(i+1)*chunk_size]
            p = multiprocessing.Process(
                target=self.read_mesh_deform,
                args=(i, chunk, return_dict)
            )
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        
        return {
            "geometry": np.concatenate([return_dict[i]["geometry"] for i in range(self.args.process_number)], axis=0),
            "normal": np.concatenate([return_dict[i]["normal"] for i in range(self.args.process_number)], axis=0)
        }
    
    def process(self):
        seq_dir = os.path.join(self.data_root, self.dataset_name, self.args.cfg.DATASET.STYLE)
        save_dir = os.path.join(self.data_root, self.dataset_name, 'pre_processed', self.args.cfg.DATASET.STYLE)

        for seq in tqdm(sorted(os.listdir(seq_dir)), desc="pre-processing UNIC dataset..."):
            # read garment deformation
            garm_topology = self.read_mesh_template(os.path.join(seq_dir, seq, 'deformation', 's_0000001.obj'))
            garment_deform = {
                "deformation": self.read_mesh_deform_parallel(os.path.join(seq_dir, seq, 'deformation'))
            }
            garment_deform.update(garm_topology)
            seq_len = garment_deform["deformation"]["geometry"].shape[0]

            # read character motion
            char_motion = self.read_character_motion(
                os.path.join(seq_dir, seq, 'animation.fbx'),
                seq_len
            )
            char_topology = self.read_mesh_template(os.path.join(seq_dir, seq, 'motion', 's_0000001.obj'))
            char_mesh = self.read_mesh_deform_parallel(os.path.join(seq_dir, seq, 'motion'))
            
            # normalize garment deformation, character motion and character mesh
            garment_deform["deformation"], char_motion, char_mesh = self.normalize(garment_deform["deformation"], char_motion, char_mesh)
            assert seq_len == char_motion["joint_rotation"].shape[0] == char_mesh["geometry"].shape[0], "ERROR: inconsistent motion length of human and garment."

            # save sequence dynamics
            char_motion_state = self.to_char_representation(char_motion)
            char_vertex_neighbors = self.read_vertex_neighbors(os.path.join(seq_dir, seq, 'motion', 's_0000001.obj'))
            garment_deform_representation = self.to_garm_representation(garment_deform["deformation"])
            for i in range(seq_len // self.args.clip_len):
                start = i * self.args.clip_len
                end = (i+1) * self.args.clip_len if i != seq_len//self.args.clip_len-1 else seq_len
                delta_end = end if i != seq_len//self.args.clip_len-1 else end-1
                seq_dynamics = {
                    "character": {
                        "motion_state": char_motion_state[start:end],   # character motion representation
                        "mesh": {                                                                           # character mesh and normal
                            "geometry": char_mesh["geometry"][start:end],
                            "normal": char_mesh["normal"][start:end]
                        },                                          
                        "joints_of_feet": self.feet_mask,                                                   # index of feet joints
                        "mask_for_normalization": self.char_motion_eu_mask,                                 # index of euclidean part in motion representation, which will be normalized
                        "topology_v": char_topology["topology_v"],
                        "topology_f": char_topology["topology_f"],
                        "uv_v": char_topology["uv_v"],
                        "uv_f": char_topology["uv_f"],
                        "neighbors": char_vertex_neighbors
                    },
                    "garment": {
                        "deformation": {                                                            # garment geometry, velocity, normal and normal change in each frame
                            "geometry": garment_deform_representation["geometry"][start:end],
                            "normal": garment_deform_representation["normal"][start:end],
                            "velocity": garment_deform_representation["velocity"][start:delta_end],
                            "normal_change": garment_deform_representation["normal_change"][start:delta_end]
                        },                                                                          
                        "topology_v": garment_deform["topology_v"],                                 # garment vertices in rest pose
                        "topology_f": garment_deform["topology_f"],                                 # garment faces in rest pose
                        "uv_v": garment_deform["uv_v"],                                             # garment vertex uv map
                        "uv_f": garment_deform["uv_f"]                                              # garment face uv map
                    }
                }

                split = "train" if seq in self.args.cfg.DATASET.training_set else "test"
                os.makedirs(os.path.join(save_dir, split, seq), exist_ok=True)
                with open(os.path.join(save_dir, split, seq, f'{seq}_clip_{str(i).zfill(5)}.pkl'), 'wb') as f:
                    pkl.dump(seq_dynamics, f)

    def add_neighbors(self):
        dataset_dir = os.path.join(self.data_root, self.dataset_name, self.args.cfg.DATASET.STYLE)
        neighbors = self.read_vertex_neighbors(os.path.join(dataset_dir, 'sequence_01', 'motion', 's_0000001.obj'))
        train_seqs = [os.path.join(self.data_root, self.dataset_name, 'pre_processed', self.args.cfg.DATASET.STYLE, 'train', x+'.pkl') for x in self.args.cfg.DATASET.training_set]
        test_seqs = [os.path.join(self.data_root, self.dataset_name, 'pre_processed', self.args.cfg.DATASET.STYLE, 'test', x+'.pkl') for x in self.args.cfg.DATASET.testing_set]
        for seq in train_seqs+test_seqs:
            with open(seq, 'rb') as f:
                seq_data = pkl.load(f)
            seq_data["character"]["neighbors"] = neighbors
            with open(seq, 'wb') as f:
                pkl.dump(seq_data, f)

    def prepare_data(self):
        self.process()
        self.save_mean_std()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--character", type=str, default="unity_smpl")  # ["unity_smpl", "ue"]
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip_morph", type=int, default=0)
    parser.add_argument("--scale", type=str, default="cm")
    parser.add_argument("--clip_len", type=int, default=100)
    parser.add_argument("--process_number", type=int, default=16)
    parser.add_argument("--encoding", type=str, default="utf-8") # ["utf-8", "gb2312"]
    args = parser.parse_args()
    args.cfg = OmegaConf.load(args.cfg)

    processor = UnicProcessor(r'/nas/nas_10/share_folder/zhaochf/gmt', args)
    processor.prepare_data()