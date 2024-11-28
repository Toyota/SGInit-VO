# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 
# The implementation is derived from PackNet-SfM (https://github.com/TRI-ML/packnet-sfm)

PROJECT ?= sginit-vo
VERSION ?= latest

WORKSPACE ?= /workspace/$(PROJECT)
DOCKER_IMAGE ?= ${PROJECT}:${VERSION}

# Please fit the following for your environment
CKPT_MNT ?= /mnt/fsx2TB/data/checkpoints
DATA_MNT ?= /mnt/fsx2TB/data/datasets
MDLS_MNT ?= /mnt/fsx2TB/data/models
PRPD_MNT ?= /mnt/fsx2TB/data/preprocessed
TMUX_MNT  ?= ~/.tmux


SHMSIZE ?= 444G
WANDB_MODE ?= run
DOCKER_OPTS := \
			--name ${PROJECT} \
			--rm -it \
			--shm-size=${SHMSIZE} \
			-e AWS_DEFAULT_REGION \
			-e AWS_ACCESS_KEY_ID \
			-e AWS_SECRET_ACCESS_KEY \
			-e WANDB_API_KEY \
			-e WANDB_ENTITY \
			-e WANDB_MODE \
			-e HOST_HOSTNAME= \
			-e OMP_NUM_THREADS=1 -e KMP_AFFINITY="granularity=fine,compact,1,0" \
			-e OMPI_ALLOW_RUN_AS_ROOT=1 \
			-e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
			-e NCCL_DEBUG=VERSION \
			-e DISPLAY=${DISPLAY} \
			-e XAUTHORITY \
			-e NVIDIA_DRIVER_CAPABILITIES=all \
			-v ~/.aws:/home/${USER}/.aws \
			-v ~/.cache:/home/${USER}/.cache \
			-v ${CKPT_MNT}:/data/checkpoints \
			-v ${DATA_MNT}:/data/datasets \
			-v ${MDLS_MNT}:/data/models \
			-v ${PRPD_MNT}:/data/preprocessed \
			-v ${TMUX_MNT}:/home/${USER}/.tmux \
			-v ${PWD}/thirdparty/vidar/configs:${WORKSPACE}/configs \
			-v /dev/null:/dev/raw1394 \
			-v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v ${PWD}:${WORKSPACE} \
			-w ${WORKSPACE} \
			--privileged \
			--ipc=host \
			--network=host

NGPUS=$(shell nvidia-smi -L | wc -l)

docker-build:
	docker build \
		-f docker/Dockerfile \
		-t ${DOCKER_IMAGE} .

docker-interactive: docker-build
	docker run --gpus all ${DOCKER_OPTS} ${DOCKER_IMAGE} bash

docker-run: docker-build
	docker run --gpus all ${DOCKER_OPTS} ${DOCKER_IMAGE} bash -c "${COMMAND}"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ---------- FRC NO-ROOT-TOOLS COMMAND PART ----------
PATH2NRT ?= thirdparty/no_root_docker_run

initialize-scripts:
	python3 ${PATH2NRT}/run_setup.py ${DOCKER_IMAGE}

mkdir-writable:
	python3 ${PATH2NRT}/mk_writable.py --dirs \
		${CKPT_MNT} ${TMUX_MNT}

noroot-build: docker-build initialize-scripts
	docker build \
		-f ${PATH2NRT}/dist/Dockerfile.no_root \
		-t ${DOCKER_IMAGE}-${USER} .

noroot-interactive: mkdir-writable
	docker run --gpus all ${DOCKER_OPTS} \
    --name ${PROJECT}-${USER} ${DOCKER_IMAGE}-${USER} bash

noroot-exec:
	docker exec -it --user ${USER} ${PROJECT}-${USER} bash
# ----------------------------------------------------
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
