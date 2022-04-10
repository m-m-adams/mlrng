import gym
import numpy as np
from .break_network import Net
from gym.spaces import MultiBinary, Discrete

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam


class RNGEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # The action and observation spaces need to be gym.spaces objects:
        self.action_space = Discrete(2)  # up, left, right, down
        self.observation_space = MultiBinary(128)
        self.state = np.zeros(128, dtype=np.integer)
        self.network = Net()
        self.criterion = nn.BCELoss()
        self.episode_loss = 0
        self.outputs = []
        self.states = []
        self.actor_optim = Adam(self.network.parameters(), lr=0.0007)

    def step(self, action):

        self.outputs.append(int(action))
        action = torch.tensor(action, dtype=torch.float)
        # obs, rew, done, _ = self.env.step(action)
        guess = self.network(self.state).squeeze()
        self.states.append(self.state)
        self.state = np.append(self.state[1:], action)

        # accumulate the gradient but learn once per episode
        loss = self.criterion(guess, action)
        self.episode_loss += loss
        loss.backward()

        info = {
            'reward': loss.detach().cpu().numpy()
        }

        return self.state, loss, False, info

    def learn(self):
        batch_obs = torch.tensor(np.array(self.states), dtype=torch.float)
        batch_acts = torch.tensor(np.array(self.outputs), dtype=torch.float)
        for _ in range(5):
            guesses = self.network(batch_obs).squeeze()
            #print(guesses.shape, batch_acts.shape)
            loss = self.criterion(guesses, batch_acts)
            print(f"replay loss for guesser is {loss}")

            loss.backward()
            self.actor_optim.step()
            self.actor_optim.zero_grad()
        print(self.outputs[:50])

    def reset(self):

        self.actor_optim.step()
        self.actor_optim.zero_grad()
        if len(self.states) > 1:
            print(
                f"episode loss for guesser is {self.episode_loss/len(self.states)}")

            print(len(self.states))
            # replay the episode a couple times for faster learning
            self.learn()
        self.episode_loss = 0
        self.state = np.zeros((128), np.integer)
        self.states = []
        self.outputs = []
        # print(*self.outputs, sep="\n")
        return self.state

    def render(self, mode='human', close=False):
        print(self.outputs)
