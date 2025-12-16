# register_env.py
import gymnasium as gym
from gymnasium.envs.registration import register

from gym_env import PortfolioGymEnv


def register_portfolio_env():
    register(
        id="PortfolioBandit-v0",
        entry_point="gym_env:PortfolioGymEnv",
    )
