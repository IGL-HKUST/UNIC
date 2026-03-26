import subprocess
import os

def run_blender_script():

    # windows 
    blender_path = r"..\Blender\blender.exe"
    #
    # for linux/macos
    blender_path = "blender"
    blender_command = [
        blender_path,
        # linux
        # “blender”

        "--background",
        "--factory-startup",
        "--python",
        os.path.abspath("./motion_morph.py")  # Get absolute path to the Python script
    ]

    try:
        subprocess.run(blender_command, check=True)
        print("Blender script executed successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the Blender path or Python script path.")
    except subprocess.CalledProcessError as e:
        print(f"Blender script failed: {e}")

if __name__ == "__main__":
    run_blender_script()
