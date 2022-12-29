import gym

#create and reset the environment
environment = gym.make('CartPole-v1', render_mode="rgb_array")
environment.reset()

#itterate the environment
for dummy in range(100):
    environment.reset()
    environment.render() 
    environment.step(environment.action_space.sample())