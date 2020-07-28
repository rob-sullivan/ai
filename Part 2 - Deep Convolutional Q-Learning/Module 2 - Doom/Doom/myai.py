# AI for Doom



# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing

# Part 1 - Building the AI

# Making the brain 
#common practice is to start with 32 common feature detectors. That's 32 images
class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernal_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernal_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernal_size = 2)
        #1 hidden layer fc = full connection
        self.fc1 = nn.Linear(in_features = self.count_neurons((1,80,80)), out_features = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)
    def forward(self, x):
        x = F.relu(max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Making the body
class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions
# Making the AI
class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()

# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)

number_actions = doom_env.action_space.n

# Building an AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

#setting up experience replay
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step=10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity=10000)

#part 2 - implementing deep convolutional Q-learning