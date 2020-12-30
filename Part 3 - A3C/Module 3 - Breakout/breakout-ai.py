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
    
    #convelutional layers
    def forward(self, inputs): # considering hidden nodes and cell nodes (in a tupple). inputs is the images
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs)) #pytorch.org/docs/nn.html non-linear activations RELU(x) = max(0,x) but explonation linear limit
        x = F.elu(self.conv2(inputs))
        x = F.elu(self.conv3(inputs))
        x = F.elu(self.conv4(inputs))

        #now flatten into one long 1d vector
        x = x.view(-1, 32 * 3 * 3)
        (hx, cx) = self.lstm(x, (hx, cx))
        x = hx
        #two output signals required for actor and critic
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)



#my_optim adam optimiser adapted to shared model 
#stacastic gradiant decent tool used to update weight to how much they contribute and calculate loss from prediction vs actual, previously used ADAM
#do a custom optimiser based on ADAM optimisor so we don't wait ages to train model

# Optimizer

import math
import torch
import torch.optim as optim

# Implementing the Adam optimizer with shared states

class SharedAdam(optim.Adam): # object that inherits from optim.Adam

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay) # inheriting from the tools of optim.Adam
        for group in self.param_groups: # self.param_groups contains all the attributes of the optimizer, including the parameters to optimize (the weights of the network) contained in self.param_groups['params']
            for p in group['params']: # for each tensor p of weights to optimize
                state = self.state[p] # at the beginning, self.state is an empty dictionary so state = {} and self.state = {p:{}} = {p: state}
                state['step'] = torch.zeros(1) # counting the steps: state = {'step' : tensor([0])}
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_() # the update of the adam optimizer is based on an exponential moving average of the gradient (moment 1)
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_() # the update of the adam optimizer is also based on an exponential moving average of the squared of the gradient (moment 2)

    # Sharing the memory
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_() # tensor.share_memory_() acts a little bit like tensor.cuda()
                state['exp_avg'].share_memory_() # tensor.share_memory_() acts a little bit like tensor.cuda()
                state['exp_avg_sq'].share_memory_() # tensor.share_memory_() acts a little bit like tensor.cuda()

    # Performing a single optimization step of the Adam algorithm (see algorithm 1 in https://arxiv.org/pdf/1412.6980.pdf)
    def step(self): #can use super(SharedAdam, self).step()
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step'][0]
                bias_correction2 = 1 - beta2 ** state['step'][0]
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss
#training of the model

import torch
import torch.nn.functional as F
from env import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param.__grad = param.grad

def train(rank, params, shared_model, optimiser): # rank desynchronise each agent
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name)
    env.seed(params.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    #numpy array 1 channel black and white images 42x42 (input is the images)
    state = env.reset()#this resets the state of the image and sets it to 42x42 etc..
    #convert into torch tensor
    state = torch.from_numpy(state)
    done = True #if episode or game is over
    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256)) #vector, num of elements 
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        #now do the training process
        values = [] #output of the critic
        log_probs = []
        rewards = []
        entropies = []
        for step in range(params.num_steps): # loop over exploration steps
            #first, second and third output
            value, action_values, (hx,cx) = model((Variable(state.unsqueeze(0)), (hx, cx))) # get a batch of inputs not an individual input
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)#action values are the Q values
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))
            



    


#test.py is last one implement a test agent will play breakout without updating the model (seperate from training)