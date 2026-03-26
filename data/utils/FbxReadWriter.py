# script adapted from https://github.com/softcat477/SMPL-to-FBX/blob/main/FbxReadWriter.py

import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from fbx import *

from data.utils.FbxCommon import *

class FbxReadWrite(object):
    def __init__(
        self,
        fbx_source_path: str,
        skeleton: list,
        args
    ):
        # prepare the FBX SDK
        lSdkManager, lScene = InitializeSdkObjects()
        self.lSdkManager = lSdkManager
        self.lScene = lScene
        self.skeleton = skeleton
        self.character = args.character
        self.root_node = self.lScene.GetRootNode()

        lResult = LoadScene(self.lSdkManager, self.lScene, fbx_source_path)
        if not lResult:
            raise Exception("An error occured while loading the scene :(")

    def get_armature_name(self):
        for i in range(self.root_node.GetChildCount()):
            if "armature" in self.root_node.GetChild(i).GetName().lower():
                return self.root_node.GetChild(i).GetName()

    def get_smpl_name(self):
        for i in range(self.root_node.GetChildCount()):
            if "smpl" in self.root_node.GetChild(i).GetName().lower():
                return self.root_node.GetChild(i).GetName()

    def load_animation(
        self,
        seq_len: int,
        fps: int = 30,
        skip: int = 60
    ):
        # set animation fps, default is 30
        fbx_time = FbxTime()
        fbx_fps = FbxTime.EMode.eFrames30
        if fps == 24:
            fbx_fps = FbxTime.EMode.eFrames24
        elif fps == 60:
            fbx_fps = FbxTime.EMode.eFrames60
        elif fps == 120:
            fbx_fps = FbxTime.EMode.eFrames120
        fbx_time.SetGlobalTimeMode(fbx_fps)

        # load character animation
        root_node_name = self.get_smpl_name() if self.character == "unity_smpl" else self.get_armature_name()
        character_node = self.root_node.FindChild(root_node_name)
        animation = {"joint_rotation": [], "joint_position": []}
        for t in tqdm(range(skip, skip+seq_len), leave=False, desc="pre-processing animation data..."):
            fbx_time.SetFrame(t, fbx_fps)
            joint_rotations = np.zeros((len(self.skeleton), 4)) # in quaternion, scalar first
            joint_positions = np.zeros((len(self.skeleton), 3))
            for i in range(len(self.skeleton)):
                joint_name = self.skeleton[i]
                joint_node = character_node.FindChild(joint_name)

                # get local, global and geometric transformation
                joint_local_transformation = list(joint_node.EvaluateLocalTransform(fbx_time))
                joint_local_transformation = np.array([list(joint_local_transformation[i]) for i in range(4)]).T
                joint_global_transformation = list(joint_node.EvaluateGlobalTransform(fbx_time))
                joint_global_transformation = np.array([list(joint_global_transformation[i]) for i in range(4)]).T
                joint_geometric_transformation = list(FbxAMatrix(
                    joint_node.GetGeometricTranslation(joint_node.EPivotSet.eSourcePivot),
                    joint_node.GetGeometricRotation(joint_node.EPivotSet.eSourcePivot),
                    joint_node.GetGeometricScaling(joint_node.EPivotSet.eSourcePivot)
                ))
                joint_geometric_transformation = np.array([list(joint_geometric_transformation[i]) for i in range(4)]).T
                
                # apply geometric transformation and then local/global transformation
                joint_local_transformation = joint_local_transformation @ joint_geometric_transformation
                joint_global_transformation = joint_global_transformation @ joint_geometric_transformation

                joint_rotations[i] = R.from_matrix(joint_local_transformation[:3,:3]).as_quat(scalar_first=True)
                joint_positions[i] = joint_global_transformation[:3, 3]
            animation["joint_rotation"].append(joint_rotations)
            animation["joint_position"].append(joint_positions)
        animation["joint_rotation"] = np.stack(animation["joint_rotation"], axis=0)
        animation["joint_position"] = np.stack(animation["joint_position"], axis=0)

        return animation