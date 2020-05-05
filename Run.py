import torch
import gym
import time
from Train import Net, normalize
env = gym.make('CartPole-v1')


def main():
    use_cuda = False
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load('cartpole.pth'))
    model.to(device)
    model.eval()

    state = env.reset()
    current_state = torch.reshape(torch.tensor(normalize(state)), (1, len(env.observation_space.high)))
    score = 0

    done = False
    while not done:
        env.render()

        action = torch.argmax(model(current_state)).item()
        state, reward, done, _ = env.step(action)
        current_state = torch.reshape(torch.tensor(normalize(state)), (1, len(env.observation_space.high)))
        score += reward

        time.sleep(0.01)

    print(score)


if __name__ == '__main__':
    main()
