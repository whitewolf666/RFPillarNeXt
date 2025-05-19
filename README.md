# RFPillarNeXt: Enhancing Geometric Information Extraction in 3D Object Detection

Our implementation is built upon the **OpenPCDet** framework.

To run our model, simply select the provided configuration file:

```
pillarnext5111.yaml
pillarnext4211.yaml
pillarnext3221.yaml
pillarnext2222.yaml
```

Example command:

```
python train.py --cfg_file cfgs/pillarnext5111.yaml
```

Make sure you have followed the standard installation steps from the [OpenPCDet repository](https://github.com/open-mmlab/OpenPCDet).



# OpenPCDet

`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection. 

It is also the official code release of [`[PointRCNN]`](https://arxiv.org/abs/1812.04244), [`[Part-A2-Net]`](https://arxiv.org/abs/1907.03670), [`[PV-RCNN]`](https://arxiv.org/abs/1912.13192), [`[Voxel R-CNN]`](https://arxiv.org/abs/2012.15712), [`[PV-RCNN++]`](https://arxiv.org/abs/2102.00463) and [`[MPPNet]`](https://arxiv.org/abs/2205.05979). 

**Highlights**: 
* `OpenPCDet` has been updated to `v0.6.0` (Sep. 2022).
* The codes of PV-RCNN++ has been supported.
* The codes of MPPNet has been supported. 
* The multi-modal 3D detection approaches on Nuscenes have been supported. 

## Overview
- [Changelog](#changelog)
- [Design Pattern](#openpcdet-design-pattern)
- [Model Zoo](#model-zoo)
- [Installation](docs/INSTALL.md)
- [Quick Demo](docs/DEMO.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Citation](#citation)


## Changelog
[2023-06-30] **NEW:** Added support for [`DSVT`](https://arxiv.org/abs/2301.06051), which achieves state-of-the-art performance on large-scale Waymo Open Dataset with real-time inference speed (27HZ with TensorRT).

[2023-05-13] **NEW:** Added support for the multi-modal 3D object detection models on Nuscenes dataset.  
* Support multi-modal Nuscenes detection (See the [GETTING_STARTED.md](docs/GETTING_STARTED.md) to process data).
* Support [TransFusion-Lidar](https://arxiv.org/abs/2203.11496) head, which ahcieves 69.43% NDS on Nuscenes validation dataset.
* Support [`BEVFusion`](https://arxiv.org/abs/2205.13542), which fuses multi-modal information on BEV space and reaches 70.98% NDS on Nuscenes validation dataset. (see the [guideline](docs/guidelines_of_approaches/bevfusion.md) on how to train/test with BEVFusion).

[2023-04-02] Added support for [`VoxelNeXt`](https://arxiv.org/abs/2303.11301) on Nuscenes, Waymo, and Argoverse2 datasets. It is a fully sparse 3D object detection network, which is a clean sparse CNNs network and predicts 3D objects directly upon voxels.

[2022-09-02] **NEW:** Update `OpenPCDet` to v0.6.0:
* Official code release of [`MPPNet`](https://arxiv.org/abs/2205.05979) for temporal 3D object detection, which supports long-term multi-frame 3D object detection and ranks 1st place on [3D detection learderboard](https://waymo.com/open/challenges/2020/3d-detection) of Waymo Open Dataset on Sept. 2th, 2022. For validation dataset, MPPNet achieves 74.96%, 75.06% and 74.52% for vehicle, pedestrian and cyclist classes in terms of mAPH@Level_2. (see the [guideline](docs/guidelines_of_approaches/mppnet.md) on how to train/test with MPPNet).
* Support multi-frame training/testing on Waymo Open Dataset (see the [change log](docs/changelog.md) for more details on how to process data).
* Support to save changing training details (e.g., loss, iter, epoch) to file (previous tqdm progress bar is still supported by using `--use_tqdm_to_record`). Please use `pip install gpustat` if you also want to log the GPU related information.
* Support to save latest model every 5 mintues, so you can restore the model training from latest status instead of previous epoch.   

[2022-08-22] Added support for [custom dataset tutorial and template](docs/CUSTOM_DATASET_TUTORIAL.md) 

[2022-07-05] Added support for the 3D object detection backbone network [`Focals Conv`](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Focal_Sparse_Convolutional_Networks_for_3D_Object_Detection_CVPR_2022_paper.pdf).

[2022-02-12] Added support for using docker. Please refer to the guidance in [./docker](./docker).

[2022-02-07] Added support for Centerpoint models on Nuscenes Dataset.

[2022-01-14] Added support for dynamic pillar voxelization, following the implementation proposed in [`H^23D R-CNN`](https://arxiv.org/abs/2107.14391) with unique operation and [`torch_scatter`](https://github.com/rusty1s/pytorch_scatter) package.

[2022-01-05] **NEW:** Update `OpenPCDet` to v0.5.2:
* The code of [`PV-RCNN++`](https://arxiv.org/abs/2102.00463) has been released to this repo, with higher performance, faster training/inference speed and less memory consumption than PV-RCNN.
* Add performance of several models trained with full training set of [Waymo Open Dataset](#waymo-open-dataset-baselines).
* Support Lyft dataset, see the pull request [here](https://github.com/open-mmlab/OpenPCDet/pull/720).


[2021-12-09] **NEW:**  Update `OpenPCDet` to v0.5.1:
* Add PointPillar related baseline configs/results on [Waymo Open Dataset](#waymo-open-dataset-baselines).
* Support Pandaset dataloader, see the pull request [here](https://github.com/open-mmlab/OpenPCDet/pull/396).
* Support a set of new augmentations, see the pull request [here](https://github.com/open-mmlab/OpenPCDet/pull/653).

[2021-12-01] **NEW:** `OpenPCDet` v0.5.0 is released with the following features:
* Improve the performance of all models on [Waymo Open Dataset](#waymo-open-dataset-baselines). Note that you need to re-prepare the training/validation data and ground-truth database of Waymo Open Dataset (see [GETTING_STARTED.md](docs/GETTING_STARTED.md)). 
* Support anchor-free [CenterHead](pcdet/models/dense_heads/center_head.py), add configs of `CenterPoint` and `PV-RCNN with CenterHead`.
* Support lastest **PyTorch 1.1~1.10** and **spconv 1.0~2.x**, where **spconv 2.x** should be easy to install with pip and faster than previous version (see the official update of spconv [here](https://github.com/traveller59/spconv)).  
* Support config [`USE_SHARED_MEMORY`](tools/cfgs/dataset_configs/waymo_dataset.yaml) to use shared memory to potentially speed up the training process in case you suffer from an IO problem.  
* Support better and faster [visualization script](tools/visual_utils/open3d_vis_utils.py), and you need to install [Open3D](https://github.com/isl-org/Open3D) firstly. 

[2021-06-08] Added support for the voxel-based 3D object detection model [`Voxel R-CNN`](#KITTI-3D-Object-Detection-Baselines).

[2021-05-14] Added support for the monocular 3D object detection model [`CaDDN`](#KITTI-3D-Object-Detection-Baselines).

[2020-11-27] Bugfixed: Please re-prepare the validation infos of Waymo dataset (version 1.2) if you would like to 
use our provided Waymo evaluation tool (see [PR](https://github.com/open-mmlab/OpenPCDet/pull/383)). 
Note that you do not need to re-prepare the training data and ground-truth database. 

[2020-11-10] The [Waymo Open Dataset](#waymo-open-dataset-baselines) has been supported with state-of-the-art results. Currently we provide the 
configs and results of `SECOND`, `PartA2` and `PV-RCNN` on the Waymo Open Dataset, and more models could be easily supported by modifying their dataset configs. 

[2020-08-10] Bugfixed: The provided NuScenes models have been updated to fix the loading bugs. Please redownload it if you need to use the pretrained NuScenes models.

[2020-07-30] `OpenPCDet` v0.3.0 is released with the following features:
   * The Point-based and Anchor-Free models ([`PointRCNN`](#KITTI-3D-Object-Detection-Baselines), [`PartA2-Free`](#KITTI-3D-Object-Detection-Baselines)) are supported now.
   * The NuScenes dataset is supported with strong baseline results ([`SECOND-MultiHead (CBGS)`](#NuScenes-3D-Object-Detection-Baselines) and [`PointPillar-MultiHead`](#NuScenes-3D-Object-Detection-Baselines)).
   * High efficiency than last version, support **PyTorch 1.1~1.7** and **spconv 1.0~1.2** simultaneously.

[2020-07-17]  Add simple visualization codes and a quick demo to test with custom data. 

[2020-06-24] `OpenPCDet` v0.2.0 is released with pretty new structures to support more models and datasets. 

[2020-03-16] `OpenPCDet` v0.1.0 is released. 


## Introduction


### What does `OpenPCDet` toolbox do?

Note that we have upgrated `PCDet` from `v0.1` to `v0.2` with pretty new structures to support various datasets and models.

`OpenPCDet` is a general PyTorch-based codebase for 3D object detection from point cloud. 
It currently supports multiple state-of-the-art 3D object detection methods with highly refactored codes for both one-stage and two-stage 3D detection frameworks.

Based on `OpenPCDet` toolbox, we win the Waymo Open Dataset challenge in [3D Detection](https://waymo.com/open/challenges/3d-detection/), 
[3D Tracking](https://waymo.com/open/challenges/3d-tracking/), [Domain Adaptation](https://waymo.com/open/challenges/domain-adaptation/) 
three tracks among all LiDAR-only methods, and the Waymo related models will be released to `OpenPCDet` soon.    

We are actively updating this repo currently, and more datasets and models will be supported soon. 
Contributions are also welcomed. 

### `OpenPCDet` design pattern

* Data-Model separation with unified point cloud coordinate for easily extending to custom datasets:
<p align="center">
  <img src="docs/dataset_vs_model.png" width="95%" height="320">
</p>

* Unified 3D box definition: (x, y, z, dx, dy, dz, heading).

* Flexible and clear model structure to easily support various 3D detection models: 
<p align="center">
  <img src="docs/model_framework.png" width="95%">
</p>

* Support various models within one framework as: 
<p align="center">
  <img src="docs/multiple_models_demo.png" width="95%">
</p>


### Currently Supported Features

- [x] Support both one-stage and two-stage 3D object detection frameworks

- [x] Support distributed training & testing with multiple GPUs and multiple machines

- [x] Support multiple heads on different scales to detect different classes

- [x] Support stacked version set abstraction to encode various number of points in different scenes

- [x] Support Adaptive Training Sample Selection (ATSS) for target assignment

- [x] Support RoI-aware point cloud pooling & RoI-grid point cloud pooling

- [x] Support GPU version 3D IoU calculation and rotated NMS 

  

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.


## Quick Demo
Please refer to [DEMO.md](docs/DEMO.md) for a quick demo to test with a pretrained model and 
visualize the predicted results on your custom data or the original KITTI data.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.


## License

`OpenPCDet` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
`OpenPCDet` is an open source project for LiDAR-based 3D scene perception that supports multiple
LiDAR-based perception models as shown above. Some parts of `PCDet` are learned from the official released codes of the above supported methods. 
We would like to thank for their proposed methods and the official implementation.   

We hope that this repo could serve as a strong and flexible codebase to benefit the research community by speeding up the process of reimplementing previous works and/or developing new methods.


## Citation 
If you find this project useful in your research, please consider cite:


```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```

## Contribution
Welcome to be a member of the OpenPCDet development team by contributing to this repo, and feel free to contact us for any potential contributions. 

