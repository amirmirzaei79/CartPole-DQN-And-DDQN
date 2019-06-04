import gym
import time
import tensorflow as tf
import keras
import numpy as np

env = gym.make('CartPole-v1')

# regression neural network to predict action rewards for each state - in this case each state has 2 actions
model = keras.models.Sequential()
model.add(keras.layers.Dense(32, activation='sigmoid', input_dim=env.observation_space.shape[0]))
model.add(keras.layers.Dense(32, activation='sigmoid'))
model.add(keras.layers.Dense(env.action_space.n, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001)) # 0.001 is learning rate of Adam

env.reset()
env.render()
observations, Reward, done, info = env.step(env.action_space.sample())
env.render()

print(observations, '\n', Reward, done, '\n', info)

time.sleep(1)