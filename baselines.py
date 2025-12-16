# baselines.py
import numpy as np
from typing import List
from env_wrapper import BanditPortfolioEnv

def run_equal_weight_baseline(env: BanditPortfolioEnv) -> List[float]:
    """
    Equal-weighted portfolio emulated by randomly switching among assets.
    For simplicity, treat it as uniform random arm selection.
    """
    rewards = []
    ctx = env.reset()
    done = False
    n_arms = env.action_space_n()
    while not done:
        action = np.random.randint(n_arms)
        ctx, reward, done, info = env.step(action)
        rewards.append(reward)
    return rewards

def run_greedy_past_mean(env: BanditPortfolioEnv) -> List[float]:
    """
    Always choose the asset with the highest mean return in the last lookback window.
    """
    rewards = []
    ctx = env.reset()
    done = False
    n_arms = env.action_space_n()
    lookback = env.lookback

    while not done:
        # ctx has [mean_ret (n_arms), vol (n_arms)]
        mean_ret = ctx[:n_arms]
        action = int(np.argmax(mean_ret))
        ctx, reward, done, info = env.step(action)
        rewards.append(reward)
    return rewards
