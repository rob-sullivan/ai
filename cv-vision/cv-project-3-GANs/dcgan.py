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

#define the generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False), #100 is size of input, 512 is size of output feature map, 4 is kernal as in size 4x4, stride and padding and bias.
            nn.BatchNorm2d(512), #normalise each feature map using batch norm.
            nn.ReLU(True), #apply activation to break linearity
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), # input is now output of previous. 256 was done through research
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
