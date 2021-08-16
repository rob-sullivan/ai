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

        # loop over exploration steps
        for step in range(params.num_steps): 
            #first, second and third output
            value, action_values, (hx,cx) = model((Variable(state.unsqueeze(0)), (hx, cx))) # get a batch of inputs not an individual input
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)#action values are the Q values
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_prob.append(log_prob)
            state, reward, done = env.step(action.numpy())
            done = (done or episode_length >= params.max_episode_length)
            reward = max(min(reward, 1), -1)
            if done:
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done: #stop exploration and move on to updating shared moddel
                break

        #init cuml reward
        R = torch.zeros(1, 1)
        if not done: # get last state shared by the network
            value, _, _ = model.((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data 
        values.append(Variable(R))
        #loss of actor
        policy_loss = 0
        #loss of critic
        value_loss = 0
        R = Variable(R)
        #generalisaed advantage estimation vation function
        # A(a,s) = Q(a,s) - V(s)
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))): # reversed is used so we can move back in time
            R = params.gamma * R + rewards[i] # R = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^(n-1) * r_{n-1} + gamma^nb_steps * V(last_state)
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2) # this is the value loss minus the critic # Q*(a*,s = V*(s)
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data # temporal difference of state values
            gae = gae * params.gamma * params.tau + TD # gae = sum_i (gamma*tau*)^i * TD(i)
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i] #policy_loss = -sum_i log(pi_i)*gae + 0.01*h_i #0.01 is to prevent falling into an issue where everything has 0 except 1
        optimizer().zero_grad()
        (policy_loss + 0.5 * value_loss).backward()7
        #stop taking extremly large losses and degenerate the algorithm
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()


#test.py is last one implement a test agent will play breakout without updating the model (seperate from training)

# Test Agent

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
import time
from collections import deque

# Making the test agent (won't update the model but will just use the shared model to explore)
def test(rank, params, shared_model):
    torch.manual_seed(params.seed + rank) # asynchronizing the test agent
    env = create_atari_env(params.env_name, video=True) # running an environment with a video
    env.seed(params.seed + rank) # asynchronizing the environment
    model = ActorCritic(env.observation_space.shape[0], env.action_space) # creating one model
    model.eval() # putting the model in "eval" model because it won't be trained
    state = env.reset() # getting the input images as numpy arrays
    state = torch.from_numpy(state) # converting them into torch tensors
    reward_sum = 0 # initializing the sum of rewards to 0
    done = True # initializing done to True
    start_time = time.time() # getting the starting time to measure the computation time
    actions = deque(maxlen=100) # cf https://pymotw.com/2/collections/deque.html
    episode_length = 0 # initializing the episode length to 0
    while True: # repeat
        episode_length += 1 # incrementing the episode length by one
        if done: # synchronizing with the shared model (same as train.py)
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)
        value, action_value, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(action_value)
        action = prob.max(1)[1].data.numpy() # the test agent does not explore, it directly plays the best action
        state, reward, done, _ = env.step(action[0, 0]) # done = done or episode_length >= params.max_episode_length
        reward_sum += reward
        if done: # printing the results at the end of each part
            print("Time {}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            reward_sum = 0 # reinitializing the sum of rewards
            episode_length = 0 # reinitializing the episode length
            actions.clear() # reinitializing the actions
            state = env.reset() # reinitializing the environment
            time.sleep(60) # doing a one minute break to let the other agents practice (if the game is done)
        state = torch.from_numpy(state) # new state and we continue

# Improvement of the Gym environment with universe
import cv2
import gym
import numpy as np
from gym.spaces.box import Box
from gym import wrappers


# Taken from https://github.com/openai/universe-starter-agent


def create_atari_env(env_id, video=False):
    env = gym.make(env_id)
    if video:
        env = wrappers.Monitor(env, 'test', force=True)
    env = MyAtariRescale42x42(env)
    env = MyNormalizedEnv(env)
    return env


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    #frame = np.reshape(frame, [1, 42, 42])
    return frame


class MyAtariRescale42x42(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(MyAtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def _observation(self, observation):
    	return _process_frame42(observation)


class MyNormalizedEnv(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(MyNormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        ret = (observation - unbiased_mean) / (unbiased_std + 1e-8)
        return np.expand_dims(ret, axis=0)

# Main code

from __future__ import print_function #brings print function from Python 3 into Python 2.6+.
import os
import torch
import torch.multiprocessing as mp
from envs import create_atari_env
from model import ActorCritic
from train import train
from test import test
import my_optim

# Gathering all the parameters (that we can modify to explore)
class Params():
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 16
        self.num_steps = 20
        self.max_episode_length = 10000
        self.env_name = 'Breakout-v0'

# Main run
os.environ['OMP_NUM_THREADS'] = '1' # 1 thread per core
params = Params() # creating the params object from the Params class, that sets all the model parameters
torch.manual_seed(params.seed) # setting the seed (not essential)
env = create_atari_env(params.env_name) # we create an optimized environment thanks to universe
shared_model = ActorCritic(env.observation_space.shape[0], env.action_space) # shared_model is the model shared by the different agents (different threads in different cores)
shared_model.share_memory() # storing the model in the shared memory of the computer, which allows the threads to have access to this shared memory even if they are in different cores
optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr) # the optimizer is also shared because it acts on the shared model
optimizer.share_memory() # same, we store the optimizer in the shared memory so that all the agents can have access to this shared memory to optimize the model
processes = [] # initializing the processes with an empty list
p = mp.Process(target=test, args=(params.num_processes, params, shared_model)) # allowing to create the 'test' process with some arguments 'args' passed to the 'test' target function - the 'test' process doesn't update the shared model but uses it on a part of it - torch.multiprocessing.Process runs a function in an independent thread
p.start() # starting the created process p
processes.append(p) # adding the created process p to the list of processes
for rank in range(0, params.num_processes): # making a loop to run all the other processes that will be trained by updating the shared model
    p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
    p.start()
    processes.append(p)
for p in processes: # creating a pointer that will allow to kill all the threads when at least one of the threads, or main.py will be killed, allowing to stop the program safely
    p.join()
