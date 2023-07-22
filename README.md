# 3d Deep Learning
## Installation
Install a Conda environment without installing the full Anaconda distribution, by using Miniconda, which is a lightweight version of Anaconda that only includes Conda and Python.

* Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install
* Create a new Conda environment: conda create --name venv_3dl python=3.8
* conda activate venv_3dl
* conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
* conda install -c fvcore -c iopath -c conda-forge fvcore iopath
* check CUDA version (e.g 11.3): nvcc --version
* CUB build time dependency (CUDA older than 11.7): conda install -c bottler nvidiacub
* conda install pytorch3d -c pytorch3d

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