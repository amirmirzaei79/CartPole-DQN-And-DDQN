# DQN & DDQN Algorithms for Open-AI gym Cart pole
Implementation for DQN (Deep Q Network) and DDQN (Double Deep Q Networks) algorithms proposed in 

"Mnih, V., Kavukcuoglu, K., Silver, D. *et al.* Human-level control through deep reinforcement learning.                    *Nature* **518,** 529–533 (2015). https://doi.org/10.1038/nature14236"

and

"Hado van Hasselt, Arthur Guez, David Silver. Deep Reinforcement Learning with Double Q-learning https://arxiv.org/abs/1509.06461"

on Open-AI gym Cart Pole environment.

Also a fraction of pole's base distance to center and pole's angle from center were added as a cost in order to encourage model to keep the pole still and in center. Adding this short term cost should help agent to learn avoiding distance from center and increasing angel (which is the final goal) faster. Although removing these costs won't make it impossible for agent to learn, just makes it harder.

Both methods of training create and save policy model in the same manner, therefore model parameters created by either one of training methods can be used for the Run file.
