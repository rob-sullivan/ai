# 3d Deep Learning
## Installation
Install CUDA version on Windows 10 without a Conda environment.

* python3 -m venv venv_3dl
* go to [pytorch's website](https://pytorch.org/) to find the correct install command e.g: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
* make sure cuda installed matches pytorch cuda. Either upgrade or downgrade appropriately
* If CUDA is less than 11.7 go to [CUB](https://github.com/NVIDIA/cub/releases), download zip file and unzip in a desired location: e.g .\venv_3dl\Lib
* If CUDA is 11.7 go to [CUB](https://github.com/NVIDIA/cub/releases) and download [version 1.17](https://github.com/NVIDIA/cub/archive/refs/tags/1.17.0.zip) then go to where CUB folder is e.g C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\cub and replace it with the downloaded 1.17 [source 1](https://github.com/facebookresearch/pytorch3d/issues/1567), [source 2](https://github.com/facebookresearch/pytorch3d/issues/1227)
* If CUDA is 11.7 add: #define THRUST_IGNORE_CUB_VERSION_CHECK true to version.cuh in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\cub
* install pytorch3d: pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
* pip install open3d


## 3d Data Processing

## 3d computer vision and Geometry

## Fitting Mesh models to Point Clouds

## Object Pose Detection and Tracking

## Differentiable Volumetric Rendering

## Neural Radiance Fields (NeRF)

## Controllable Neral Feature Fields

## Modelling Humans in 3d

## SynSin

## Mesh R-CNN