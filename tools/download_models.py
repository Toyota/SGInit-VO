# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 

import os
import subprocess
import gdown

ROOT_DIR = "/data/models/papers"
MODELS = ["SGInit_MR_selfsup_DDAD.ckpt", "ZeroDepth_unified.ckpt", "droid.pth"]
URLS = [
    "https://tri-ml-public.s3.amazonaws.com/github/vidar/models/SGInit_MR_selfsup_DDAD.ckpt",
    "https://tri-ml-public.s3.amazonaws.com/github/vidar/models/ZeroDepth_unified.ckpt", 
    "https://drive.google.com/u/0/uc?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh",
]

# Ensure ROOT_DIR exists
os.makedirs(ROOT_DIR, exist_ok=True)

for model, url in zip(MODELS, URLS):
    full_path = os.path.join(ROOT_DIR, model)
    
    if os.path.isfile(full_path):
        print(f"File {full_path} already exists. No action taken.")
    else:
        print(f"File {full_path} does not exist. Downloading...")

        if model.endswith('droid.pth'):
            # Use gdown for .pth files
            gdown.download(url, full_path, quiet=False)
        else:
            # Use wget for other files
            subprocess.run(["wget", url, "-O", full_path])

        print(f"Downloaded {model} to {full_path}.")
