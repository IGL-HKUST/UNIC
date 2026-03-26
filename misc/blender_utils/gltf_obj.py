import bpy
import os


# 配置路径和文件名
glb_file_path = "E:/ly/narph_dataset_generation/cloth_motion/jump.gltf"  # 修改为你的 GLB 文件路径
output_directory = "E:/ly/narph_dataset_generation/cloth_motion/Jump_obj/"  # 修改为你的导出路径
motion_name = "Jump"

if not os.path.exists(glb_file_path):
    raise FileNotFoundError(f"GLB file not found at {glb_file_path}")

# import GLB 
try:
    bpy.ops.import_scene.gltf(filepath=glb_file_path)
    print("GLB file imported successfully.")
except Exception as e:
    print(f"Failed to import GLB file: {e}")
    raise

# 查找带有形态键的网格对象
mesh_obj = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and obj.data.shape_keys:
        mesh_obj = obj
        break

if mesh_obj is None:
    raise ValueError("No Mesh with Shape Keys found in the imported GLB file.")

# 确保网格对象的形态键存在动画数据
if mesh_obj.data.shape_keys.animation_data is None or mesh_obj.data.shape_keys.animation_data.action is None:
    raise ValueError("No animation data found on the Shape Keys.")

# 获取形态键动画的帧范围
action = mesh_obj.data.shape_keys.animation_data.action
start_frame = int(action.frame_range[0])
end_frame = int(action.frame_range[1])
total_frames = end_frame - start_frame + 1

# 输出起始帧和结束帧
print(f"Start frame: {start_frame}, End frame: {end_frame}")
print(f"Total frames: {total_frames}")
# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 确保选中所有物体
bpy.ops.object.select_all(action='SELECT')

# 逐帧导出 OBJ 文件
for frame in range(start_frame, end_frame + 1):
    # 设置当前帧
    bpy.context.scene.frame_set(frame)
    
    # 配置 OBJ 文件的输出路径
    obj_file_path = os.path.join(output_directory, f"{motion_name}_{frame}.obj")
    
    # 导出当前帧为 OBJ 文件
    bpy.ops.export_scene.obj(filepath=obj_file_path, use_selection=True, use_animation=False)
    
    print(f"Exported frame {frame} to {obj_file_path}")
