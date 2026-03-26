import torch
import os
import pickle as pkl
from pytorch3d.transforms.rotation_conversions import *

def smpl2obj(vertices, path, scale='mm'):
    if scale == 'm':
        pass
    elif scale == 'mm':
        vertices = vertices * 1000
    else:
        raise ValueError("scale should be 'm' or 'mm'")

    fs = list()
    with open("./body_model/smpl.obj") as f:
        lines = f.readlines()
        for line in lines:
            l = line.split(" ")
            if "f" in l:
                fs.append((int(l[1]), int(l[2]), int(l[3])))

    with open(path, "w") as f:
        for v in vertices:
            f.write(( 'v %f %f %f\n' % ( v[0], v[1], v[2]) ))
        for face in fs:
            f.write(( 'f %d %d %d\n' % ( face[0], face[1], face[2]) ))
    return torch.tensor(fs)

def save_geometry_as_obj(
    save_dir: str,
    vs: torch.Tensor,
    vns: torch.Tensor,
    fs: torch.Tensor,
    uv_v: torch.Tensor = None,
    uv_f: torch.Tensor = None
):
    os.makedirs(save_dir, exist_ok=True)

    garment_style = save_dir.split('/')[-2]
    with open(f'./tmp/{garment_style}_mtl.txt', 'r') as f:
        mtl_lines = f.readlines()

    for i in range(vs.shape[0]):
        vertices = vs[i]
        normals = vns[i]
        with open(os.path.join(save_dir, str(i).zfill(6)+'.obj'), "w") as f:
            f.write(mtl_lines[1])

            for v in vertices:
                f.write(( 'v %f %f %f\n' % ( v[0], v[1], v[2]) ))
            for vn in normals:
                f.write(( 'vn %f %f %f\n' % ( vn[0], vn[1], vn[2]) ))
            if uv_v is not None:
                # garment
                for mtl_line in mtl_lines[2:]:
                    f.write(mtl_line)
            else:
                # smpl
                for face in fs:
                    f.write(( 'f %d %d %d\n' % ( face[0]+1, face[1]+1, face[2]+1) ))

def save_deformation(
    data,
    output,
    save_dir=r'./results',
    style="tshirt",
    save_obj_files=False,
    save_with_transl=False
):
    # ground truth data
    T = output.shape[0]
    garment_v = data["garment"]["deformation"]["geometry"][0, :T].cpu().numpy()
    garment_vn = data["garment"]["deformation"]["normal"][0, :T].cpu().numpy()
    garment_f = data["garment"]["topology_f"].cpu().numpy()
    garment_uv_v = data["garment"]["uv_v"].cpu().numpy()
    garment_uv_f = data["garment"]["uv_f"].cpu().numpy()
    character_v = data["character"]["mesh"]["geometry"][0, :T]
    character_vn = data["character"]["mesh"]["normal"][0, :T]
    character_f = data["character"]["topology_f"].cpu().numpy()
    seq_transl = data["transl"][:T].reshape(T, 1, 3)
    
    # network prediction
    geometry_pred = output.cpu().numpy()

    # saving results
    if save_obj_files:
        if save_with_transl:
            save_geometry_as_obj(
                os.path.join(save_dir, style, 'garment_pred_with_transl'),
                geometry_pred + seq_transl,
                garment_vn,
                garment_f,
                uv_v=garment_uv_v,
                uv_f=garment_uv_f
            )
        else:
            save_geometry_as_obj(
                os.path.join(save_dir, style, 'garment_pred'),
                geometry_pred,
                garment_vn,
                garment_f,
                uv_v=garment_uv_v,
                uv_f=garment_uv_f
            )
        
        if save_with_transl:
            save_geometry_as_obj(
                os.path.join(save_dir, style, 'garment_gt_with_transl'),
                garment_v + seq_transl,
                garment_vn,
                garment_f,
                uv_v=garment_uv_v,
                uv_f=garment_uv_f
            )
        else:
            save_geometry_as_obj(
                os.path.join(save_dir, style, 'garment_gt'),
                garment_v,
                garment_vn,
                garment_f,
                uv_v=garment_uv_v,
                uv_f=garment_uv_f
            )
        
        if save_with_transl:
            save_geometry_as_obj(
                os.path.join(save_dir, style,'character_with_transl'),
                character_v + seq_transl,
                character_vn,
                character_f
            )
        else:
            save_geometry_as_obj(
                os.path.join(save_dir, style,'character'),
                character_v,
                character_vn,
                character_f
            )
    else:
        deformation_pred = {
            "pred": geometry_pred,
            "cloth_faces": garment_f,
            "obstacle": character_v,
            "obstacle_faces": character_f
        }
        deformation_gt = {
            "pred": garment_v,
            "cloth_faces": garment_f,
            "obstacle": character_v,
            "obstacle_faces": character_f
        }
        os.makedirs(os.path.join(save_dir, style), exist_ok=True)
        with open(os.path.join(save_dir, style, 'garment_pred.pkl'), 'wb') as f:
            pkl.dump(deformation_pred, f)
        with open(os.path.join(save_dir, style, 'garment_gt.pkl'), 'wb') as f:
            pkl.dump(deformation_gt, f)