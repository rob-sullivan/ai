# Deep Convolutional GANS

import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variables


# settings
batchSize = 64

imageSize = 64

#transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor()])

#load dataset
dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
dataloarder = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle=True)

#define weights function that takes input from NN m
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)