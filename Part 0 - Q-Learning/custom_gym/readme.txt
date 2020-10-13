pip install -e .

python

import gym
import envs
env = gym.make('CustomEnv-v0) //env initialised
env.step() // success
env.reset() // reset

if you make any changes just do this
pip install -e .

to delete do this
pip uninstall custom_env