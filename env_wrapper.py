# env_wrapper.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

class BanditPortfolioEnv:
    """
    Simple multi-asset environment for bandits.
    - State (context): recent returns and volatility per asset.
    - Action (arm): select one asset (or discrete allocation choice).
    - Reward: next-period portfolio return of chosen asset minus transaction cost.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        lookback: int = 20,
        transaction_cost: float = 0.0005
    ):
        """
        prices: DataFrame indexed by date, columns=assets.
        """
        assert prices.shape[1] >= 2, "Need at least 2 assets."
        self.prices = prices
        self.assets: List[str] = list(prices.columns)
        self.n_assets = len(self.assets)
        self.lookback = lookback
        self.transaction_cost = transaction_cost

        self.returns = prices.pct_change().dropna()
        self.dates = list(self.returns.index)
        self.current_idx = lookback
        self.last_action = None

    def reset(self) -> np.ndarray:
        self.current_idx = self.lookback
        self.last_action = None
        return self._get_context()

    def _get_context(self) -> np.ndarray:
        """
        Build context vector from last `lookback` days:
        - mean return and volatility per asset.
        Shape: [n_assets * 2]
        """
        start = self.current_idx - self.lookback
        end = self.current_idx
        window = self.returns.iloc[start:end]
        mean_ret = window.mean(axis=0).values      # shape: [n_assets]
        vol = window.std(axis=0).values            # shape: [n_assets]
        ctx = np.concatenate([mean_ret, vol]).astype(np.float32)
        return ctx

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        action: index of asset chosen.
        Reward: realized next-day return minus transaction cost if switching.
        """
        assert 0 <= action < self.n_assets
        if self.current_idx >= len(self.returns) - 1:
            # No next return available, terminate.
            return self._get_context(), 0.0, True, {"done_reason": "end_of_data"}

        # Next-day returns for all assets
        next_ret_vec = self.returns.iloc[self.current_idx + 1]
        raw_ret = float(next_ret_vec.iloc[action])

        # Simple transaction cost if we change asset
        cost = 0.0
        if self.last_action is not None and self.last_action != action:
            cost = self.transaction_cost

        reward = raw_ret - cost

        self.last_action = action
        self.current_idx += 1
        done = self.current_idx >= len(self.returns) - 2
        ctx = self._get_context()
        info = {
            "date": self.dates[self.current_idx],
            "raw_ret": raw_ret,
            "transaction_cost": cost
        }
        return ctx, reward, done, info

    def action_space_n(self) -> int:
        return self.n_assets
