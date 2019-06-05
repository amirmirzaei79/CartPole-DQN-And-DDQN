import gym
import time
import keras
import numpy as np
import random

env = gym.make('CartPole-v1')

# regression neural network to predict action rewards for each state - in this case each state has 2 actions
model = keras.models.Sequential()
model.add(keras.layers.Dense(32, activation='relu', input_dim=env.observation_space.shape[0]))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(env.action_space.n, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['mae']) # 0.001 is learning rate of Adam

# deep Q learning init
gamma = 0.95
epsilon = 1.0 # Explore rate
epsilonMin = 0.01
epsilonDecay = 0.95
episodeLimit = 5000
batch_size = 256
memory_limit = 100000

memory = []

# deep Q
for episode in range(episodeLimit):
    currentStateArray = env.reset()
    currentState = np.array([currentStateArray])
    done = False
    while not done:
        # env.render()

        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(currentState)[0])

        newStateArray, reward, done, info = env.step(action)
        newState = np.array([newStateArray])
        if not done:
            target = reward + gamma * np.max(model.predict(newState))
        else:
            target = reward

        targetLabel = model.predict(currentState)[0]
        targetLabel[action] = target
        model.fit(currentState, targetLabel.reshape(1, 2), epochs=1, verbose=0)
        memory.append([currentState, action, reward, done, newState])
        currentState = newState
    else:
        print(episode)

    if epsilon > epsilonMin:
        epsilon *= epsilonDecay

    if len(memory) > memory_limit:
        del memory[0:5000]

    if len(memory) > batch_size:
        mini_batch = random.sample(memory, batch_size)

        for currentState, action, reward, done, newState in mini_batch:
            if not done:
                target = reward + gamma * np.max(model.predict(newState))
            else:
                target = reward

            targetLabel = model.predict(currentState)[0]
            targetLabel[action] = target
            model.fit(currentState, targetLabel.reshape(1, 2), epochs=1, verbose=0)

# Play game
print("\nPlaying Game...")
time.sleep(1)

currentStateArray = env.reset()
currentState = np.array([currentStateArray])
done = False
while not done:
    env.render()
    action = np.argmax(model.predict(currentState)[0])
    currentStateArray, reward, done, info = env.step(action)
    currentState = np.array([currentStateArray])
    time.sleep(0.01)