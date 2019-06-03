import gym
import time

env = gym.make("CartPole-v1")
env.reset()
for _ in range(200):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.01)
env.close()