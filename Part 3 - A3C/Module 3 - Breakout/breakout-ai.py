#envs improves gym environment with universe optimum preprocessing of images (normalises images and values)
#main executes the whole program
#model
# AI for breakout

#import libraries 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std = 1.0):
    out = torch.randn(weights.size()) #initialise a torch tensor that follows a normal distribution
    out *= std / torch.sqrt(output.pow(2).sum(1).expand_as(out)) #gets all weights, sum of sqrd then get sqrd of all //variance of out will be std^2
    return out

#initialising the weights of the neural network for an optimal learning
def weights_init(m):
    #initialise weights in a specific way based on research papers
    classname = m.__class__.__name__ #represents a neural network
    if classname.find('Conv') != -1: #-1 means no
        weights_shape = list(m.weight.data.size())
        fan_in = np.prod(weights_shape[1:4]) #product of dimension 1,2 and 3 of our shape (upper bound not included) dim1*dim2*dim3
        fan_out = np.prod(weights_shape[2:4]) * weights_shape[0] # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / fan_in + fan_out)
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weights_shape = list(m.weight.data.size())
        fan_in = weights_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / fan_in + fan_out)
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

# making the a3c brain - CRNN
class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        #feature detectors 32 features detector (aka kernals) size 3x3 stride of 2 padding of 1 (4 convelutions)

        #eyes of the ai or in Q(s,a) this is S
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)# output 32 convelution images
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)

        #memory in brain
        #learn temporal property of brain (will encode the bounce)
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        #num of possible actions this is A
        num_outputs = action_space.n #take this from gym

        self.critic_linear = nn.Linear(256, 1) # output = V(S)
        self.actor_linear = nn.Linear(256, num_outputs) # output = Q(S,A)
        
        #initialise random weights
        self.apply(weights_init)

        #small standard diviation to actior and large for critic (explotation vs exploration)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.action_linear.bias.data.fill_(0) # may not be necessary # just to make sure

        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1)
        self.critic_linear.bias.data.fill_(0) # may not be necessary # just to make sure

        #init bias for lstm
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        #puts module in train mode (drop outs in batch normalisation)
        self.train()



#my_optim adam optimiser adapted to shared model
#training of the model
#test.py is last one implement a test agent will play breakout without updating the model (seperate from training)