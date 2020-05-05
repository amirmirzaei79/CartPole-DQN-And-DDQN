import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import gym
from collections import deque
import random
from os import path
import numpy as np

use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')

outFile = open("log.txt", "w")

MEMORY_LEN = 10000
BATCH_SIZE = 512

# Discount Factor
GAMMA = 0.995

epsilon = 1
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.05

EPISODE_LIMIT = 2000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(env.observation_space.high), 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


memory = deque(maxlen=MEMORY_LEN)
model = Net().to(device)


def get_action(state: torch.Tensor):
    if random.random() > epsilon:
        with torch.no_grad():
            return torch.argmax(model(state.to(device))).item()
    else:
        return np.random.randint(env.action_space.n)


def normalize(st):
    new_st = [st[0] / 5, st[1] / 5, st[2] / 24, st[3] / 24]
    return new_st


def fit(dataloader: DataLoader, epochs: int = 1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train():
    batch = random.sample(memory, min(len(memory), BATCH_SIZE))

    x = []
    y = []
    for currentState, action, reward, nextState, done in batch:
        if not done:
            target = reward + GAMMA * torch.max(model(nextState.to(device))).detach()
        else:
            target = reward

        q = model(currentState.to(device)).detach()
        q[0][action] = target

        x.append(currentState)
        y.append(q)

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)
    fit(dataloader, 1)


def test():
    state = env.reset()
    current_state = torch.reshape(torch.tensor(normalize(state)), (1, len(env.observation_space.high))).to(device)
    done = False
    t = 0
    while not done:
        action = torch.argmax(model(current_state.to(device))).item()
        state, reward, done, info = env.step(action)
        current_state = torch.reshape(torch.tensor(normalize(state)), (1, len(env.observation_space.high)))
        t += 1
    return t


def main():
    global epsilon

    if path.exists('cartpole.pth'):
        model.load_state_dict(torch.load('cartpole.pth'))

    max_score = 0
    for episode in range(EPISODE_LIMIT):
        state = env.reset()
        score = 0
        done = False

        current_state = torch.tensor(normalize(state)).reshape(-1, len(env.observation_space.high))
        while not done:
            action = get_action(current_state)
            state, reward, done, _ = env.step(action)
            next_state = torch.tensor(normalize(state)).reshape(-1, len(env.observation_space.high))

            if score < 499:
                memory.append((current_state, action, reward, next_state, done))

            score += reward

            train()

            current_state = next_state

            epsilon = max(EPSILON_DECAY * epsilon, MIN_EPSILON)
        else:
            test_score = test()
            if test_score >= max_score:
                max_score = test_score
                torch.save(model.state_dict(), 'cartpole.pth')

            print("episode: " + str(episode + 1) + " -> " + str(score) + " - " + str(test_score))
            outFile.write("episode: " + str(episode + 1) + " -> " + str(score) + " - " + str(test_score) + "\n")


if __name__ == '__main__':
    main()
