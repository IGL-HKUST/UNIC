import numpy as np
from copy import deepcopy
import os
from tqdm import tqdm
from argparse import ArgumentParser

def merge_parts_by_mtl(parts):
    mtl_dict = {}
    for i in range(len(parts)):
        part = parts[i]
        mtl = part[1]
        if mtl not in mtl_dict.keys():
            mtl_dict[mtl] = [i]
        else:
            mtl_dict[mtl].append(i)

    mtl_parts = []
    for mtl in mtl_dict.keys():
        part_idx = mtl_dict[mtl]
        mtl_part = []
        for i in range(len(part_idx)):
            pid = part_idx[i]
            if i == 0:
                mtl_part += parts[pid]
            else:
                mtl_part += parts[pid][2:]
        mtl_parts.append(deepcopy(mtl_part))
    
    return mtl_parts

def rearrange_part_mesh(topology, frame, args):
    save_dir = f'/nas/nas_10/share_folder/zhaochf/gmt/final/saved_objs/{args.exp_name}/{args.pose}/{args.garment_name}/garment_{args.mode}_parts'
    os.makedirs(save_dir, exist_ok=True)

    V = topology["vertices"]
    Vt = topology["texture"]
    Vn = topology["normal"]
    parts = topology["parts"]

    parts = merge_parts_by_mtl(parts)
    print(len(parts))
    exit()

    for part_idx in range(len(parts)):
        part = parts[part_idx]
        vs = []
        for line in part:
            if line.startswith('f '):
                idx = [n.split('/') for n in line.replace('f ', '').replace('\n', '').split(' ')]
                vs += [int(n[0]) - 1 for n in idx]
        old_vs = list(set(sorted(vs)))
        new_vs = [x for x in range(len(old_vs))]
        old_to_new = {}
        for i in range(len(old_vs)):
            old_to_new.update({old_vs[i]: new_vs[i]})

        # rewrite part mesh .obj file
        # 1. header 
        part_obj_lines = [
            '# OBJ Exporter v1.2 by Jaden Seungwoo Oh at CLO Virtual Fashion Inc.\n'
            'mtllib s.mtl\n'
        ]

        # 2. vertices, texture and normal
        part_v = V[old_vs]; part_vt = Vt[old_vs]; part_vn = Vn[old_vs]
        for i in range(len(old_vs)):
            part_obj_lines.append(f'v {part_v[i][0]} {part_v[i][1]} {part_v[i][2]}\n')
        for i in range(len(old_vs)):
            part_obj_lines.append(f'vt {part_vt[i][0]} {part_vt[i][1]}\n')
        for i in range(len(old_vs)):
            part_obj_lines.append(f'vn {part_vn[i][0]} {part_vn[i][1]} {part_vn[i][2]}\n')
        
        # 3. faces
        part_obj_lines.append(part[0]); part_obj_lines.append(part[1])
        for line in part:
            if line.startswith('f '):
                idx = [n.split('/') for n in line.replace('f ', '').replace('\n', '').split(' ')]
                old_face = [int(n[0]) - 1 for n in idx]
                new_face = [old_to_new[x] for x in old_face]
                part_obj_lines.append(f'f {new_face[0]+1}/{new_face[0]+1}/{new_face[0]+1} {new_face[1]+1}/{new_face[1]+1}/{new_face[1]+1} {new_face[2]+1}/{new_face[2]+1}/{new_face[2]+1}\n')

        # 4. write obj file
        save_fn = os.path.join(save_dir, f'{args.garment_name}_{part_idx}_{frame}.obj')
        with open(save_fn, "w") as f:
            for line in part_obj_lines:
                f.write(line)

def read_topology(file, encoding):
    V, Vt, Vn, F, Ft, Vni = [], [], [], [], [], []
    with open(file, 'r', encoding=encoding) as f:
        T = f.readlines()

    part_start = []
    for i in range(len(T)):
        t = T[i]

		# 3D vertex
        if t.startswith('v '):
            v = [float(n) for n in t.replace('v ', '').split(' ')]
            V += [v]
        # UV vertex
        elif t.startswith('vt '):
            v = [float(n) for n in t.replace('vt ', '').split(' ')]
            Vt += [v]
        # Normal vertex
        elif t.startswith('vn '):
            v = [float(n) for n in t.replace('vn ', '').split(' ')]
            Vn += [v]
        # parts
        if t.startswith('s 1'):
            part_start.append(i)

    parts = []
    for ps in range(len(part_start)-1):
        part = []
        for l in range(part_start[ps], part_start[ps+1]):
            part.append(T[l])
        parts.append(deepcopy(part))
    last_part = []
    for l in range(part_start[-1], len(T)):
        last_part.append(T[l])
    parts.append(deepcopy(last_part))

    V = np.array(V, np.float32)
    Vt = np.array(Vt, np.float32)
    Vn = np.array(Vn, np.float32)
    Vn = Vn / np.linalg.norm(Vn, axis=-1, keepdims=True)
    
    return {
        "vertices": V,
        "texture": Vt,
        "normal": Vn,
        "parts": parts
    }

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--style", type=str)
    parser.add_argument("--pose", type=str)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    file = r'/nas/nas_10/share_folder/zhaochf/gmt/final/jk_dress_2/sequence_01/deformation/s_0000001.obj'
    encoding = 'utf-8' # gb2312 utf-8

    if args.style == "jk":
        exp_name = "debug--unic_jk_smpl"
        garment_name = "jk_dress_2"
        args.exp_name = exp_name
        args.garment_name = garment_name
    elif args.style == "princess":
        exp_name = "debug--unic_princess_smpl"
        garment_name = "princess_dress"
        args.exp_name = exp_name
        args.garment_name = garment_name
    elif args.style == "hanfu":
        exp_name = "debug--unic_hanfu_smpl"
        garment_name = "hanfu_dress"
        args.exp_name = exp_name
        args.garment_name = garment_name
    
    seq_dir = f'/nas/nas_10/share_folder/zhaochf/gmt/final/saved_objs/{exp_name}/{args.pose}/{garment_name}/garment_{args.mode}'
    for frame in tqdm(sorted(os.listdir(seq_dir))):
        topology = read_topology(os.path.join(seq_dir, frame), encoding)
        rearrange_part_mesh(topology, frame.split('.')[0], args)