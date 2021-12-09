import torch
import gym
import time
from Train_DQN import normalize_state, state_reward
from Model import Model
env = gym.make('CartPole-v1')


def main():
    use_cuda = False
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    policy_net = Model(len(env.observation_space.high), env.action_space.n)
    policy_net.load_state_dict(torch.load('policy_net.pth'))
    policy_net.to(device)
    policy_net.eval()

    state = env.reset()
    normalize_state(state)
    done = False
    score = 0
    reward = 0
    while not done:
        env.render()
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.argmax(policy_net(state)).item()
        state, env_reward, done, _ = env.step(action)
        normalize_state(state)
        score += env_reward
        reward += state_reward(state, env_reward)

        time.sleep(0.01)

    print(f'score: {score} - reward: {reward}')


if __name__ == '__main__':
    main()
