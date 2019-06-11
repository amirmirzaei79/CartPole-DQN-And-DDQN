import gym
import time
import keras
import numpy as np
import random

env = gym.make('CartPole-v1')

# regression neural network to predict action rewards for each state - in this case each state has 2 actions
model = keras.models.Sequential()
model.add(keras.layers.Dense(32, activation='softmax', input_dim=env.observation_space.shape[0]))
model.add(keras.layers.Dense(32, activation='softmax'))
model.add(keras.layers.Dense(env.action_space.n, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['mae']) # 0.001 is learning rate of Adam

# deep Q learning init
gamma = 0.99
epsilon = 1.0 # Explore rate
epsilonMin = 0.005
episodeLimit = 4000
exploreLimit = 0.95 * episodeLimit // 1
batch_size = 250
memory_limit = 25000

memory = []

# deep Q
for episode in range(episodeLimit):
    currentState = env.reset()
    currentStateArray = np.array([currentState])
    done = False
    score = 0
    while not done:
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(currentStateArray)[0])

        newState, reward, done, info = env.step(action)
        newStateArray = np.array([newState])
        if not done:
            target = reward + gamma * np.max(model.predict(newStateArray))
        else:
            target = reward

        targetLabel = model.predict(currentStateArray)[0]
        targetLabel[action] = target
        memory.append([currentStateArray, action, reward, done, newStateArray])
        currentStateArray = newStateArray
        score += 1
    else:
        print(episode, '-> Score =', score)

    if episode < exploreLimit:
        epsilon = (1 - epsilonMin) * (1 - (episode / exploreLimit)) + epsilonMin
    else:
        epsilon = epsilonMin

    if len(memory) > memory_limit:
        del memory[0:memory_limit // 5]

    if len(memory) > batch_size:
        mini_batch = random.sample(memory, batch_size)

        for currentStateArray, action, reward, done, newStateArray in mini_batch:
            if not done:
                target = reward + gamma * np.max(model.predict(newStateArray))
            else:
                target = reward

            targetLabel = model.predict(currentStateArray)[0]
            targetLabel[action] = target
            model.fit(currentStateArray, targetLabel.reshape(1, 2), epochs=1, verbose=0)

model.save('cartpole.h5')

# Play game
print("\nPlaying Game...")
time.sleep(1)

currentState = env.reset()
currentStateArray = np.array([currentState])
done = False
score = 0
while not done:
    env.render()
    action = np.argmax(model.predict(currentStateArray)[0])
    currentState, reward, done, info = env.step(action)
    currentStateArray = np.array([currentState])
    time.sleep(0.01)
    score += 1

print(score)
