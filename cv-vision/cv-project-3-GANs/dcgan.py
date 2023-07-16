# Deep Convolutional GANS
# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# settings
batchSize = 64 # size of match
imageSize = 64 # size of generated images

#transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

#load dataset
dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle=True)

#define weights function that takes input from NN module
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#define the generator
class Generator(nn.Module):
    """Generates an image from noise"""
    def __init__(self):
        super(Generator, self).__init__() #activates inheritance for nn.Module
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False), #from experiments; 100 is size of input, 512 is size of output feature map, 4 is kernal as in square size 4x4, stride and padding and bias.
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
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        #input will initially be some random vector noise of size 100
        output = self.main(input) # this will return three channels of the generated image
        return output

# Create Generator
netG = Generator()
netG.apply(weights_init) #initialise the weights of the network

class Discriminator(nn.Module):
    """Takes a generated image and returns a discriminating number between 0 and 1"""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), #3 channels of generator,
            #leakyRELU that includes original formula but multiplied by min(0, x) to give negative values. came from experimentation
            nn.LeakyReLU(0.2, inplace=True), #inplace optional operation
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True), #inplace optional operation
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True), #inplace optional operation
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True), #inplace optional operation
            nn.Conv2d(512, 1, 4, 2, 1, bias=False), # output is a vector dimension of 1 because it returns between 0 and 1.
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input) # we need to flatten the generator's impage
        return output.view(-1) # flatten 2d dimension into 1d
    
# Create discriminator
netD = Discriminator()
netD.apply(weights_init) #initialise the weights of the network


#Training discriminator and generator

#update weights of discriminator (what is real 1 and what is fake 0)
criterion = nn.BCELoss()#binary cross entropy loss
#update weights of generator (take fake image, feed into discriminator [0 or 1], set target always to 1 and get output and compute loss, back propagating it to the generator)
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(25):
    for i, data in enumerate(dataloader, 0):
        #update weights of discriminator
        netD.zero_grade()

        #train discriminator to understand real and fake images
        #train discriminator with a real image of dataset
        real, _ = data # contains image, labels
        input = Variable(real) #associate with gradient
        target = Variable(torch.once(input.size()[0])) #0 reject and 1 =accepted, input.size will return size of mini batch
        output = netD(input)
        errD_real = criterion(output, input) #real ground truth

        #train discriminator with a fake image generated by Generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach()) # to save up for memory
        errD_fake = criterion(output, input) #real ground truth

        #backprop total error
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # update weights of Generator
        #update weights of discriminator
        netG.zero_grade()
        target = Variable(torch.once(input.size()[0])) #0 reject and 1 =accepted, input.size will return size of mini batch
        output = netD(fake) #get discriminator values for each fake image
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()

        #print losses and save real and fake images
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0])) # We print les losses of the discriminator (Loss_D) and the generator (Loss_G).
        if i % 100 == 0: # Every 100 steps:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True) # We save the real images of the minibatch.
            fake = netG(noise) # We get our fake generated images.
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True) # We also save the fake generated images of the minibatch.