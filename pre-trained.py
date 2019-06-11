import gym
import time
import numpy as np
import keras

model = keras.models.load_model('cartpole.h5')

# Play game
print("\nPlaying Game...")
env = gym.make('CartPole-v1')
time.sleep(1)

currentState = env.reset()
currentStateArray = np.array([currentState])
done = False
T = 0
while not done:
    env.render()
    action = np.argmax(model.predict(currentStateArray)[0])
    currentState, reward, done, info = env.step(action)
    currentStateArray = np.array([currentState])
    time.sleep(0.01)
    T += 1

print(T)
