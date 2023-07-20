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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#added due to deprecated code

#transformations
#transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

#load dataset
#The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
#https://www.cs.toronto.edu/~kriz/cifar.html
dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.

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
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), #we add another inversed convolution
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
            nn.Conv2d(512, 1, 4, 1, 0, bias=False), # output is a vector dimension of 1 because it returns between 0 and 1.
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input) # we need to flatten the generator's image
        return output.view(-1) # flatten 2d dimension into 1d
    
# Create discriminator
netD = Discriminator()
netD.apply(weights_init) #initialise the weights of the network

if __name__ == '__main__':
    #Training discriminator and generator

    #update weights of discriminator (what is real 1 and what is fake 0)
    criterion = nn.BCELoss()#binary cross entropy loss
    #update weights of generator (take fake image, feed into discriminator [0 or 1], set target always to 1 and get output and compute loss, back propagating it to the generator)
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(25): # We iterate over 25 epochs.

        for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset.
            
            # 1st Step: Updating the weights of the neural network of the discriminator

            netD.zero_grad() # We initialize to 0 the gradients of the discriminator with respect to the weights.
            
            # Training the discriminator with a real image of the dataset
            real, _ = data # We get a real image of the dataset which will be used to train the discriminator.
            input = Variable(real) # We wrap it in a variable.
            target = Variable(torch.ones(input.size()[0])) # We get the target.
            output = netD(input) # We forward propagate this real image into the neural network of the discriminator to get the prediction (a value between 0 and 1).
            errD_real = criterion(output, target) # We compute the loss between the predictions (output) and the target (equal to 1).
            
            # Training the discriminator with a fake image generated by the generator
            noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) # We make a random input vector (noise) of the generator.
            fake = netG(noise) # We forward propagate this random input vector into the neural network of the generator to get some fake generated images.
            target = Variable(torch.zeros(input.size()[0])) # We get the target.
            output = netD(fake.detach()) # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).
            errD_fake = criterion(output, target) # We compute the loss between the prediction (output) and the target (equal to 0).

            # Backpropagating the total error
            errD = errD_real + errD_fake # We compute the total error of the discriminator.
            errD.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator.
            optimizerD.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator.

            # 2nd Step: Updating the weights of the neural network of the generator

            netG.zero_grad() # We initialize to 0 the gradients of the generator with respect to the weights.
            target = Variable(torch.ones(input.size()[0])) # We get the target.
            output = netD(fake) # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).
            errG = criterion(output, target) # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 1).
            errG.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator.
            optimizerG.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator.
            
            # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.item(), errG.item())) # We print les losses of the discriminator (Loss_D) and the generator (Loss_G).
            if i % 100 == 0: # Every 100 steps:
                vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True) # We save the real images of the minibatch.
                fake = netG(noise) # We get our fake generated images.
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True) # We also save the fake generated images of the minibatch.