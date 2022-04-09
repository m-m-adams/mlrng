import gym
import numpy as np
from break_network import Net
from gym.spaces import MultiBinary, Box

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam


class RNGEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super().__init__()
        # The action and observation spaces need to be gym.spaces objects:
        self.action_space = MultiBinary(32)  # up, left, right, down
        # Here's an observation space for 200 wide x 100 high RGB image inputs:
        self.observation_space = MultiBinary(128)
        self.state = np.zeros(128, dtype=np.integer)
        self.network = Net()
        self.criterion = nn.BCELoss()
        self.episode_loss = 0
        self.outputs = []
        self.actor_optim = Adam(self.network.parameters(), lr=0.001)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float)
        #obs, rew, done, _ = self.env.step(action)
        guess = self.network(self.state)
        self.outputs.append(
            ''.join([str(int(x)) for x in action.detach().cpu().numpy()]))
        self.state = np.append(self.state[32:], action)

        # accumulate the gradient but learn once per episode
        loss = self.criterion(guess, action)
        self.episode_loss += loss
        loss.backward()

        return self.state, loss, False, None

    def reset(self):
        # learn
        self.actor_optim.step()
        self.actor_optim.zero_grad()
        print(f"episode loss for guesser is {self.episode_loss}")
        self.episode_loss = 0
        self.state = np.zeros((128), np.integer)
        #print(*self.outputs, sep="\n")
        return self.state

    def render(self, mode='human', close=False):
        pass
