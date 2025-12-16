# gym_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

from env_wrapper import BanditPortfolioEnv


class PortfolioGymEnv(gym.Env):
    """
    Gymnasium-compatible environment wrapping BanditPortfolioEnv.

    Observation: Box(low, high, shape=[2 * n_assets]) with mean returns and volatilities.
    Action: Discrete(n_assets), choosing one asset.
    Reward: Next-period return of chosen asset minus transaction cost.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices,
        lookback: int = 20,
        transaction_cost: float = 0.0005,
        render_mode: str = None,
    ):
        super().__init__()
        self.base_env = BanditPortfolioEnv(
            prices=prices,
            lookback=lookback,
            transaction_cost=transaction_cost,
        )
        self.render_mode = render_mode

        n_assets = self.base_env.n_assets
        obs_dim = 2 * n_assets

        # Observation: mean returns and vol per asset; bounded by some reasonable values
        obs_low = np.full((obs_dim,), -1.0, dtype=np.float32)
        obs_high = np.full((obs_dim,), 1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action: choose one asset
        self.action_space = spaces.Discrete(n_assets)

        self._current_obs = None

    def reset(
        self,
        *,
        seed: int = None,
        options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        ctx = self.base_env.reset()
        # Clip context into observation bounds
        ctx = np.clip(ctx, self.observation_space.low, self.observation_space.high)
        self._current_obs = ctx.astype(np.float32)
        info: Dict[str, Any] = {}
        return self._current_obs, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        ctx, reward, done, info = self.base_env.step(int(action))
        ctx = np.clip(ctx, self.observation_space.low, self.observation_space.high)
        self._current_obs = ctx.astype(np.float32)

        terminated = done
        truncated = False  # no explicit time limit here; can add if desired
        return self._current_obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            # Minimal text render: could show current date and last reward
            print("Current observation:", self._current_obs)

    def close(self):
        pass
