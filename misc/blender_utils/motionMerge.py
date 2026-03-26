import bpy
import os
import csv
from bpy import context

# ---------------------------- User Configuration ----------------------------

# Path to the base FBX model
BASE_MODEL_FBX_PATH = r"E:/ly/narph_dataset_generation/base_model.fbx"  # Modify this path as needed

# Path to the folder containing action FBX files
ACTIONS_FOLDER_PATH = r"E:/ly/narph_dataset_generation/actions_folder"  # Modify this path as needed

# Path to save the CSV file
CSV_OUTPUT_PATH = r"E:/ly/narph_dataset_generation/actions_log.csv"  # Modify this path as needed

# Frame interval between actions (from end of previous action to start of new action)
FRAME_INTERVAL = 360

# Offset for copying the first frame at the midpoint (180 frames before start of new action)
FRAME_OFFSET = 180

# --------------------------------------------------------------------------------

class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.writer = None

    def open(self):
        """Open the CSV file and write the header."""
        try:
            self.file = open(self.filepath, mode='w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['Action Name', 'Start Frame', 'End Frame'])
            print(f"CSV file created and header written: {self.filepath}")
        except IOError as e:
            print(f"Failed to open CSV file for writing: {e}")
            self.file = None
            self.writer = None

    def write_row(self, action_name, start_frame, end_frame):
        """Write a row to the CSV file."""
        if self.writer:
            self.writer.writerow([action_name, start_frame, end_frame])
            print(f"Log entry: Action Name={action_name}, Start Frame={start_frame}, End Frame={end_frame}")
        else:
            print("CSV writer not initialized; cannot write log.")

    def close(self):
        """Close the CSV file."""
        if self.file:
            self.file.close()
            print(f"CSV file closed: {self.filepath}")

def clear_scene():
    """Clear the current Blender scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.actions):
        bpy.data.actions.remove(block)

def import_fbx(path, import_mesh=True):
    """Import an FBX file. Optionally only imports the armature."""
    bpy.ops.import_scene.fbx(filepath=path)
    imported_objects = list(bpy.context.selected_objects)
    armatures = [obj for obj in imported_objects if obj.type == 'ARMATURE']

    if not import_mesh:
        for obj in imported_objects:
            if obj.type != 'ARMATURE':
                bpy.data.objects.remove(obj, do_unlink=True)

    return armatures

def register_rest_pose(armature_obj):
    """Register the Rest Pose at frame 0."""
    bpy.context.scene.frame_set(0)
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.transforms_clear()
    bpy.ops.anim.keyframe_insert_menu(type='LocRotScale')
    bpy.ops.object.mode_set(mode='OBJECT')

def copy_action_to_base(action, base_action, frame_offset):
    """Copy keyframes from one action to the base action with an offset."""
    for fcurve in action.fcurves:
        data_path = fcurve.data_path
        array_index = fcurve.array_index
        base_fcurve = base_action.fcurves.find(data_path, index=array_index)
        if base_fcurve is None:
            base_fcurve = base_action.fcurves.new(data_path=data_path, index=array_index)
        for keyframe in fcurve.keyframe_points:
            new_frame = keyframe.co.x + frame_offset
            new_value = keyframe.co.y
            base_fcurve.keyframe_points.insert(frame=new_frame, value=new_value).interpolation = keyframe.interpolation
    # Update the frame range of the base action
    base_action.frame_range = (
        min(base_action.frame_range[0], frame_offset),
        max(base_action.frame_range[1], frame_offset + action.frame_range[1] - action.frame_range[0])
    )

def copy_pose_to_frame(armature_obj, from_frame, to_frame):
    """Copy the pose from from_frame to to_frame and insert keyframes."""
    # Set to from_frame to capture the pose
    bpy.context.scene.frame_set(from_frame)
    bpy.context.view_layer.update()

    # Store the current pose
    pose_transforms = {}
    for bone in armature_obj.pose.bones:
        pose_transforms[bone.name] = {
            'location': bone.location.copy(),
            'rotation_quaternion': bone.rotation_quaternion.copy()
        }

    # Apply the stored pose at to_frame
    bpy.context.scene.frame_set(to_frame)
    bpy.ops.object.mode_set(mode='POSE')
    for bone in armature_obj.pose.bones:
        if bone.name in pose_transforms:
            bone.location = pose_transforms[bone.name]['location']
            bone.rotation_quaternion = pose_transforms[bone.name]['rotation_quaternion']
    bpy.ops.anim.keyframe_insert_menu(type='LocRotScale')
    bpy.ops.object.mode_set(mode='OBJECT')

def main():
    clear_scene()
    base_armatures = import_fbx(BASE_MODEL_FBX_PATH, import_mesh=True)
    if not base_armatures:
        print("No armature found in the base model.")
        return
    base_armature = base_armatures[0]
    register_rest_pose(base_armature)

    if not base_armature.animation_data:
        base_armature.animation_data_create()
    if base_armature.animation_data.action is None:
        base_action = bpy.data.actions.new(name="BaseAction")
        base_armature.animation_data.action = base_action
    else:
        base_action = base_armature.animation_data.action

    logger = CSVLogger(CSV_OUTPUT_PATH)
    logger.open()

    try:
        action_files = [f for f in os.listdir(ACTIONS_FOLDER_PATH) if f.lower().endswith('.fbx')]
        print(f"Found {len(action_files)} action files: {action_files}")
    except FileNotFoundError as e:
        print(f"Actions folder not found: {e}")
        logger.close()
        return

    if not action_files:
        print("No FBX files found in the actions folder.")
        logger.close()
        return

    # Initialize to 0 so the first action starts at 0 + FRAME_INTERVAL
    previous_end_frame = 0

    for action_file in action_files:
        print(f"Processing action file: {action_file}")
        action_path = os.path.join(ACTIONS_FOLDER_PATH, action_file)
        action_name = os.path.splitext(action_file)[0]

        imported_armatures = import_fbx(action_path, import_mesh=False)
        if not imported_armatures:
            print(f"No armature found in {action_file}. Skipping.")
            continue
        action_armature = imported_armatures[0]

        if not action_armature.animation_data or not action_armature.animation_data.action:
            print(f"No action found in {action_file}. Skipping.")
            bpy.data.objects.remove(action_armature, do_unlink=True)
            continue
        action = action_armature.animation_data.action

        unique_action_name = action_name
        if unique_action_name in bpy.data.actions:
            suffix = 1
            while f"{unique_action_name}_{suffix}" in bpy.data.actions:
                suffix += 1
            unique_action_name = f"{unique_action_name}_{suffix}"
        action.name = unique_action_name

        # Calculate the start frame for the new action
        start_frame = previous_end_frame + FRAME_INTERVAL
        copy_action_to_base(action, base_action, start_frame)

        # Insert keyframes at the start_frame
        bpy.context.scene.frame_set(start_frame)
        bpy.ops.object.select_all(action='DESELECT')
        base_armature.select_set(True)
        bpy.context.view_layer.objects.active = base_armature
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.anim.keyframe_insert_menu(type='LocRotScale')

        # Calculate the midpoint_frame where the first pose will be copied
        midpoint_frame = start_frame - FRAME_OFFSET

        # Copy the first frame pose of the action to the midpoint_frame
        copy_pose_to_frame(base_armature, start_frame, midpoint_frame)

        # Ensure the scene is back to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Calculate action length
        action_length = int(action.frame_range[1] - action.frame_range[0])
        end_frame = start_frame + action_length

        # Log the action details
        logger.write_row(unique_action_name, start_frame, end_frame)

        # Update previous_end_frame to the current action's end frame
        previous_end_frame = end_frame

        # Remove the imported action armature
        bpy.data.objects.remove(action_armature, do_unlink=True)

    logger.close()
    print("Script execution completed.")

if __name__ == "__main__":
    main()
