# Copyright 2024 Toyota Motor Corporation. All rights reserved.
# The implementation is derived from TRI-VIDAR -- 2023 Toyota Research Institute.

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTORCH_VERSION=1.12.1+cu113
ENV TORCHVISION_VERSION=0.13.1+cu113
ENV TORCHAUDIO_VERSION=0.12.1

ENV CUDNN_VERSION=8.2.1.32-1+cuda11.3
ENV NCCL_VERSION=2.9.9-1+cuda11.3
ENV DEBIAN_FRONTEND=noninteractive

# Install basic libraries
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl docker.io vim wget ca-certificates

# Install python and pip
RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# Install pytorch
RUN pip install  \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install CUDNN and NCCL
RUN apt-get install -y \
    libcudnn8=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION}

# Install extra packages (apt-get)
RUN apt-get install -y libgl1-mesa-dev libglib2.0-0 unzip
RUN apt install -y ninja-build libsm6 libxrender1 python3-tk

# Install extra ML/CV packages (pip) for vidar
RUN pip install opencv-python==4.8.1.78 wandb==0.10.31 einops==0.3.2 lpips==0.1.4 scikit-image==0.19.3 timm==0.6.11 flow_vis==0.1 seaborn==0.13.2
# Install LieTorch
RUN pip install git+https://github.com/princeton-vl/lietorch.git@91887c3be4fe72aefe4af1deb67f2ea9ce17a049
# Install more transformers stuff with restricting pytorch update to 2.0
RUN pip install mmcv==2.1.0
RUN pip install taming-transformers==0.0.1 --no-deps 
RUN pip install taming-transformers-rom1504==0.0.6  --no-deps
# Resolve `--no-deps` trouble
RUN pip install \
    omegaconf \
    pytorch-lightning==2.1.2
# Install torch-scatter
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
# Install open3d
RUN pip install open3d==0.17.0
# Install gdown
RUN pip install gdown==5.2.0

# ######################################
# https://github.com/TRI-ML/vidar/blob/main/docker/Dockerfile
# Install packages for depth estimations via vidar
RUN pip install \
    fire==0.4.0 
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# Install CamViz dependencies
RUN pip install                 \
    pygame==2.0.1               \
    PyOpenGL==3.1.5             \
    PyOpenGL-accelerate==3.1.5
RUN apt-get update && apt-get install -y          \
    mesa-utils                  \
    freeglut3-dev               \
    libsdl2-2.0-0               \
    python-pygame
# ######################################


# Downgrade protobuf
RUN pip install --upgrade protobuf==3.20.0

# Define project name
ENV PROJECT=sginit-vo

# Create workspace folder
RUN mkdir -p /workspace/${PROJECT}

# Copy project to workspace folder
WORKDIR /workspace/${PROJECT}
COPY . /workspace/${PROJECT}

# Postprocess after copy files
RUN cd /workspace/${PROJECT}/ && python setup.py install

# Set environment variables
ENV PYTHONPATH="/workspace/${PROJECT}/droid_slam:$PYTHONPATH"
ENV PYTHONPATH="/workspace/${PROJECT}/thirdparty/vidar:$PYTHONPATH"
ENV PYTHONPATH="/workspace/${PROJECT}/thirdparty/vidar/externals/camviz:$PYTHONPATH"
ENV PYTHONPATH="/workspace/${PROJECT}/thirdparty/vidar/externals/efm_datasets:$PYTHONPATH"