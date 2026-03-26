import torch
from pytorch3d.transforms import *
import os
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy

from unic.data.utils import normalize

class UnicDataset(torch.utils.data.Dataset):
    def __init__(self, phase, **kwargs):
        super(UnicDataset, self).__init__()

        self.data_config = kwargs["cfg"]["DATASET"]
        self.data_root = self.data_config.ROOT
        self.phase = phase
        self.curr_iter = 0
        self.dataset_len = 0
        self.idx_map = {}

        self.all_paths = self.load_paths()
        self.statistics = self.load_statistics()
            
    def __len__(self):
        return self.dataset_len

    def __next__(self):
        if self.curr_iter >= len(self):
            self.curr_iter = 0
            raise StopIteration()
        else:
            single_data = self.__getitem__(self.curr_iter)
            self.curr_iter += 1
        
        return single_data

    def __getitem__(self, idx):
        s_i, c_i = self.idx_map[idx]
        return self.load_data(s_i, c_i)
    
    def get_sequence_clip_idx(self, idx):
        return self.idx_map[idx]

    def set_device(self, device):
        self.device = device
        
    def load_paths(self):
        dataset_path = os.path.join(self.data_root, self.data_config.STYLE, self.phase)
        seqs_available = self.data_config.training_set if self.phase == "train" else self.data_config.testing_set
        seqs = [os.path.join(dataset_path, x) for x in seqs_available]
        all_paths = []
        for s in range(len(seqs)):
            seq = seqs[s]; clips = sorted(os.listdir(os.path.join(dataset_path, seq)))
            seq_path = []
            for c in range(len(clips)):
                clip = clips[c]
                if not clip.endswith('.pkl'):
                    continue
                seq_path.append(os.path.join(dataset_path, seq, clip))
                self.idx_map[self.dataset_len] = (s, c)
                self.dataset_len += 1
            all_paths.append(seq_path)

        return all_paths
    
    def load_data(self, seq_idx, clip_idx):
        with open(self.all_paths[seq_idx][clip_idx], "rb") as f:
            data = pkl.load(f)

        # record translation in world coordinate
        data_transl = deepcopy(data["character"]["motion_state"][:, :3])

        # convert data from numpy array to torch tensor
        char_motion_norm_mask = data["character"]["mask_for_normalization"]
        data["character"]["motion_state"][:, char_motion_norm_mask] = normalize(
            data["character"]["motion_state"][:, char_motion_norm_mask],
            self.statistics["character"]
        )
        data["character"]["motion_state"] = torch.from_numpy(data["character"]["motion_state"]).float()
        for k in data["character"].keys():
            if k == "topology_v" or k == "topology_f":
                data["character"][k] = torch.from_numpy(data["character"][k]).float()
            elif k == "neighbors":
                data["character"][k] = torch.from_numpy(data["character"][k]).int()
        for k in data["garment"].keys():
            if k == "deformation":
                for kd in data["garment"]["deformation"].keys():
                    data["garment"]["deformation"][kd] = torch.from_numpy(data["garment"]["deformation"][kd]).float()
                    # convert normal change from rotation matrix to quaternion
                    if kd == "normal_change":
                        data["garment"]["deformation"][kd] = matrix_to_quaternion(data["garment"]["deformation"][kd])
            else:
                data["garment"][k] = torch.from_numpy(data["garment"][k]).float()

        # align garment deformation to character motion sequence length
        if clip_idx == len(self.all_paths[seq_idx]) - 1:
            data["character"]["mesh"]["geometry"] = data["character"]["mesh"]["geometry"][:-1]
            data["character"]["mesh"]["normal"] = data["character"]["mesh"]["normal"][:-1]
            data["garment"]["deformation"]["geometry"] = data["garment"]["deformation"]["geometry"][:-1]
            data["garment"]["deformation"]["normal"] = data["garment"]["deformation"]["normal"][:-1]

        if self.phase == "test":
            return data, data_transl
        else:
            return data
    
    def load_all_data(self):
        T = self.data_config.SEQ_LEN
        all_data = []
        for s_i in range(len(self.all_paths)):
            seq = self.all_paths[s_i]
            for c_i in tqdm(range(len(seq)), desc=f"loading data of {seq[0].split('/')[-2]}"):
                single_data, _ = self.load_data(s_i, c_i)
                char_motion = single_data["character"]
                garment_deform = single_data["garment"]
                self.assert_data_validity(char_motion, garment_deform)

                # split each animation sequence into chunks
                chunks = char_motion["motion_state"].shape[0] // T
                if chunks == 0:
                    all_data.append(single_data)
                else:
                    for x in range(chunks):
                        all_data.append({
                            "character": {
                                "motion_state": char_motion["motion_state"][x*T:(x+1)*T],
                                "mesh": {
                                    "geometry": char_motion["mesh"]["geometry"][x*T:(x+1)*T],
                                    "normal": char_motion["mesh"]["normal"][x*T:(x+1)*T]
                                },
                                "joints_of_feet": deepcopy(char_motion["joints_of_feet"]),
                                "mask_for_normalization": deepcopy(char_motion["mask_for_normalization"]),
                                "topology_v": deepcopy(char_motion["topology_v"]),
                                "topology_f": deepcopy(char_motion["topology_f"]),
                                "uv_v": deepcopy(char_motion["uv_v"]),
                                "uv_f": deepcopy(char_motion["uv_f"]),
                                "neighbors": deepcopy(char_motion["neighbors"]),
                            },
                            "garment": {
                                "deformation": {
                                    "geometry": garment_deform["deformation"]["geometry"][x*T:(x+1)*T],
                                    "velocity": garment_deform["deformation"]["velocity"][x*T:(x+1)*T],
                                    "normal": garment_deform["deformation"]["normal"][x*T:(x+1)*T],
                                    "normal_change": garment_deform["deformation"]["normal_change"][x*T:(x+1)*T]
                                },
                                "topology_v": deepcopy(garment_deform["topology_v"]),
                                "topology_f": deepcopy(garment_deform["topology_f"]),
                                "uv_v": deepcopy(garment_deform["uv_v"]),
                                "uv_f": deepcopy(garment_deform["uv_f"])
                            }
                        })

        return all_data
    
    def load_statistics(self):
        with open(os.path.join(self.data_root, self.data_config.STYLE, 'mean_std.pkl'), 'rb') as f:
            statistics = pkl.load(f)
        
        return statistics
    
    def assert_data_validity(
        self,
        char_data: dict,
        garm_data: dict
    ):
        assert char_data["motion_state"].shape[0] == char_data["mesh"]["geometry"].shape[0]
        assert char_data["mesh"]["geometry"].shape[0] == char_data["mesh"]["normal"].shape[0]
        assert char_data["motion_state"].shape[0] == garm_data["deformation"]["geometry"].shape[0] 
        assert garm_data["deformation"]["velocity"].shape[0] == garm_data["deformation"]["geometry"].shape[0]
        assert garm_data["deformation"]["normal_change"].shape[0] == garm_data["deformation"]["geometry"].shape[0]
        assert garm_data["deformation"]["normal"].shape[0] == garm_data["deformation"]["geometry"].shape[0]