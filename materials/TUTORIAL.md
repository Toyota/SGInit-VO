<!-- Copyright 2024 Toyota Motor Corporation.  All rights reserved.  -->
# Tutorial

Download tiny dataset at first;

```bash
curl -s https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/datasets/KITTI_tiny.tar | tar -xv -C /data/datasets/
```

## Visualization (GIF's Left)
Adopting [vidar](https://github.com/TRI-ML/vidar) and [CamViz](https://github.com/TRI-ML/camviz) to visualization demo. 

```bash
python thirdparty/vidar/scripts/launch.py demo_configs/camviz_demo.yaml
```

## Self-supervised Prior Learning
Using [vidar](https://github.com/TRI-ML/vidar), pre-train the depth, ego-motion, and intrinsics as:

```bash
python thirdparty/vidar/scripts/launch.py demo_configs/selfsup_resnet18_vo_calib.yaml
```
Then, the checkpoint file will be store at `/data/checkpoints/vo_demo/<DATE>/models/###.ckpt`


## Visual Odometry

Set the above checkpoint path to `SELFSUP_CKPT_OVERRIDE=` of [demo_selfsup_vo_integration.sh](../shells/demo_selfsup_vo_integration.sh), then run

```shell
./shells/demo_selfsup_vo_integration.sh
```