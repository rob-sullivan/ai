#ref: https://github.com/amdegroot/ssd.pytorch
#import libraries
import torch #pytorch to build nn for cv because it has dynamic graphs, so we can efficiently compute gradient composition functions in backward propagation, (i.e update weights through stocastic gradient decent in deep nn hidden layers)
from torch.autograd import Variable #autograd is a package that is responsible for gradient decent. Variable class converts tensors into pytorch variable into tensor and gradient.
import cv2 #drawing rectangles
from data import BaseTransform, VOC_CLASSES as labelmap #base transform converts images so they will be compatible with the NN VOC_CLASSES is just a dictionary that converts text fields to numbers
from ssd import build_ssd #this is the constructor that builds the single shot classifier. it's a pretrained model trained on 30 / 40 objects such as dogs, trains, horses, etc.
import imageio #converts video to images and then ouput rectangles on the video
#detector not based on open cv but on deep NN

# define detection function
    #frame is the image, net is the ssd nn, transform converts images into the correct format, we don't need to work in grey scale
def detect(frame, net, transform):
    #get height and width 
    height, width = frame.shape[:2] #:2=0,1
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1) #rbg, ssd was traind on grb.
    #unsqueeze function is just before filtting nn
    #we use unsqueeze to create a fake dimention of the batch
    #it should always be the the index of dimension of the batch (always first dimension)
    #torch tensor of inputs to a torch variable (tensor + gradient)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    #upper left corner width, height, lower right corner width, height
    scale = torch.Tensor([width, height, width, height]) #needed to normalise images between zero and one
