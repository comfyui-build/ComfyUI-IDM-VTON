import sys
import os
import subprocess
from huggingface_hub import snapshot_download

CUSTOM_NODES_PATH = os.path.dirname(os.path.abspath(__file__))
HF_REPO_ID = "yisol/IDM-VTON"

if 'COMFYUI_PATH' in os.environ:
    models_dir = os.path.join(os.environ['COMFYUI_PATH'], "models")
else:
    parent_dir = os.path.dirname(CUSTOM_NODES_PATH)
    models_dir = os.path.join(parent_dir, "models")

WEIGHTS_PATH = os.path.join(models_dir, "IDM-VTON")

def build_pip_install_cmds(args):
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        return [sys.executable, '-s', '-m', 'pip', 'install'] + args
    else:
        return [sys.executable, '-m', 'pip', 'install'] + args

def ensure_package():
    cmds = build_pip_install_cmds(['-r', 'requirements.txt'])
    subprocess.run(cmds, cwd=CUSTOM_NODES_PATH)

if __name__ == "__main__":
    ensure_package()
    if not os.path.exists(WEIGHTS_PATH) or not os.listdir(WEIGHTS_PATH):
        snapshot_download(repo_id=HF_REPO_ID, local_dir=WEIGHTS_PATH, local_dir_use_symlinks=False)
    else:
        print(f"{WEIGHTS_PATH} already exists and is not empty. No need to download.")
        print(f"If a model loading error occurs, please delete the directory: {WEIGHTS_PATH}.")
