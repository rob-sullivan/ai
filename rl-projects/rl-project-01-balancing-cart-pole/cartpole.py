import gym
import random

#create and reset the environment
env = gym.make('CartPole-v0')

states = env.observation_space.shape[0]
actions = env.action_space.shape[0]