#ref: https://github.com/amdegroot/ssd.pytorch
#import libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# define detection function