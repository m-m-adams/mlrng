import torch
import numpy as np
from network import FeedForwardNN
from torch.distributions import Bernoulli
from torch.optim import Adam
from torch import log_, nn
try:
    from RLGym.rng_break_env import RNGEnv
except:
    import sys
    import os
    sys.path.append('.')
    from RLGym.rng_break_env import RNGEnv

from stable_baselines3 import DQN

env = RNGEnv()

model = DQN('MlpPolicy', env, verbose=1)

obs = env.reset()
for i in range(50):
    for i in range(1000):
        action, _state = model.predict(obs)
        obs, reward, done, info = env.step(action)

    obs = env.reset()
for i in range(100_000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
env.render()
