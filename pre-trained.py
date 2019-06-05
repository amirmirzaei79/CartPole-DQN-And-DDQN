import gym
import time
import numpy as np
import keras

model = keras.models.load_model('cartpole.h5')

# Play game
print("\nPlaying Game...")
env = gym.make('CartPole-v1')
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
