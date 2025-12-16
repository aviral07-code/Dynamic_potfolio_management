# evaluation.py
import numpy as np
from typing import Dict

def compute_cumulative_return(rewards: np.ndarray) -> float:
    """
    Approximate cumulative return from per-step returns.
    Assumes rewards are small percentage returns per step.
    """
    gross = np.prod(1.0 + rewards)
    return gross - 1.0

def compute_sharpe_ratio(rewards: np.ndarray, risk_free: float = 0.0, steps_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio: (mean - rf) / std * sqrt(steps_per_year).
    """
    if rewards.std() == 0:
        return 0.0
    mean = rewards.mean()
    std = rewards.std()
    sr = (mean - risk_free) / std * np.sqrt(steps_per_year)
    return sr

def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Max drawdown from equity curve.
    equity_curve: array of cumulative equity values.
    """
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def evaluate_rewards(rewards: np.ndarray) -> Dict[str, float]:
    """
    Convenience function to compute metrics given per-step rewards.
    """
    cum_ret = compute_cumulative_return(rewards)
    equity_curve = np.cumprod(1.0 + rewards)
    mdd = compute_max_drawdown(equity_curve)
    sr = compute_sharpe_ratio(rewards)

    return {
        "cumulative_return": cum_ret,
        "max_drawdown": mdd,
        "sharpe_ratio": sr
    }
def summarize_returns(rewards: np.ndarray):
    """
    Compute mean daily return, std of daily returns, and number of trading steps.
    """
    rewards = np.asarray(rewards, dtype=float)
    mean_daily = rewards.mean()
    std_daily = rewards.std()
    n_steps = rewards.size
    return {
        "mean_daily_return": mean_daily,
        "std_daily_return": std_daily,
        "n_steps": n_steps,
    }