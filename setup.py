"""
Setup script: installs dependencies, clones dinov3, and downloads VOC2012.
Run once after cloning this repo:

    python setup.py
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DINOV3_URL = "https://github.com/facebookresearch/dinov3"
DINOV3_DIR = os.path.join(ROOT, "dinov3")


def run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def install_deps():
    print("[deps] Running uv sync...")
    run(["uv", "sync"])
    print("[deps] Done.")


def clone_dinov3():
    if os.path.isdir(DINOV3_DIR):
        print(f"[dinov3] Already cloned at {DINOV3_DIR}, skipping.")
        return
    print("[dinov3] Cloning dinov3...")
    run(["git", "clone", DINOV3_URL, DINOV3_DIR])
    print("[dinov3] Done.")


def download_voc():
    print("[VOC2012] Downloading dataset via download_voc.py...")
    run([sys.executable, os.path.join(ROOT, "download_voc.py")])
    print("[VOC2012] Done.")


if __name__ == "__main__":
    install_deps()
    clone_dinov3()
    download_voc()
    print("\nSetup complete.")
