
try:
    from RLGym.rng_break_env import RNGEnv
except:
    import sys
    import os
    sys.path.append('.')
    from RLGym.rng_break_env import RNGEnv

from stable_baselines3 import DQN

env = RNGEnv()
print(env.action_space)

model = DQN('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=100_000, log_interval=4)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
env.save("dqn_output.txt")
