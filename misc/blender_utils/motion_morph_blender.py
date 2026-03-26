import bpy


def motion_morph(fbx_file_path, export_fbx_path):
    # Import the FBX file
    bpy.ops.import_scene.fbx(
        filepath=fbx_file_path,
        axis_forward='-Z',  # Set Z axis forward
        axis_up='Y',  # Set Y axis up (compatible with Marvelous Designer)
        use_manual_orientation=False,  # Disable manual orientation
        global_scale=1.0,  # Keep the original scale
        use_custom_normals=True,  # Preserve custom normals
        use_anim=True,  # Import animation
        anim_offset=1.0  # Animation offset
    )

    # Get the imported object
    obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None

    # Frame offset
    frame_ = 60

    if obj and obj.type == 'ARMATURE':
        # Switch to Pose mode
        bpy.ops.object.mode_set(mode='POSE')
        bpy.context.scene.frame_set(0)

        # Offset keyframe timeline
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.co.x += frame_  # Keyframe offset

        # Insert keyframes for each bone
        for bone in obj.pose.bones:
            bone.bone.select = True  # Select the bone

            # Clear location and rotation
            bpy.ops.pose.rot_clear()
            bpy.ops.pose.loc_clear()

            # Insert keyframes for location, rotation, and scale
            bone.keyframe_insert(data_path="location", frame=0)
            bone.keyframe_insert(data_path="rotation_quaternion", frame=0)
            bone.keyframe_insert(data_path="scale", frame=0)

            bone.bone.select = False  # Deselect the bone

        # Extend the scene's end frame
        bpy.context.scene.frame_end += frame_

    # Switch back to Object mode before exporting
    bpy.ops.object.mode_set(mode='OBJECT')

    # Export the modified FBX file
    bpy.ops.export_scene.fbx(
        filepath=export_fbx_path,
        axis_forward='-Z',  # Set Z axis forward
        axis_up='Y',  # Set Y axis up
        use_selection=True,  # Export only selected objects
        global_scale=1.0,  # Keep the original scale
        apply_unit_scale=True,  # Apply scale transformation
        bake_anim=True,  # Export animations
        bake_anim_use_all_bones=True  # Ensure all bones are included in the animation export
    )


# ./blender/blender.exe --background --factory-startup --python ./utils/motion_morph_blender.py
def main():
    # input_path = "./motion_raw/Catwalk Walk.fbx"
    input_path = "../motion_raw/Catwalk Walk.fbx"

    output_path = "../motion/1.fbx"
    motion_morph(input_path, output_path)

if __name__ == "__main__":
    # Run the script from command line
    main()
