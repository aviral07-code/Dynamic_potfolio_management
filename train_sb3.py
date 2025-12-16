# train_sb3.py
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from data_utils import download_price_data, train_test_split_prices
from gym_env import PortfolioGymEnv
from evaluation import evaluate_rewards, summarize_returns


def make_env(prices):
    # Wrapper function that SB3 expects (for vectorized envs)
    def _init():
        return PortfolioGymEnv(prices=prices, lookback=20, transaction_cost=0.0005)
    return _init


def evaluate_rl_agent(model, env, n_episodes: int = 1):
    """
    Roll out the trained RL agent and compute portfolio metrics.
    """
    all_rewards = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_rewards = []

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_rewards.append(reward)

        all_rewards.extend(ep_rewards)

    rewards_arr = np.array(all_rewards, dtype=float)
    metrics = evaluate_rewards(rewards_arr)
    summary = summarize_returns(rewards_arr)
    print("RL (PPO) TEST metrics:", metrics)
    print("RL (PPO) TEST summary:", summary)
    return metrics,summary

    return evaluate_rewards(rewards_arr)
    


def main():
    tickers = ["SPY", "QQQ", "TLT", "GLD"]
    prices = download_price_data(tickers)
    train_prices, test_prices = train_test_split_prices(prices, train_ratio=0.7)

    # Create training environment
    env = PortfolioGymEnv(prices=train_prices, lookback=20, transaction_cost=0.0005)

    # (Optional) wrap with VecEnv, but SB3 will auto-wrap with DummyVecEnv
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_portfolio_tb/",
    )

    # Train
    model.learn(total_timesteps=100_000)

    # Save model
    model.save("ppo_portfolio_model")

    # Evaluate on test data
    test_env = PortfolioGymEnv(prices=test_prices, lookback=20, transaction_cost=0.0005)
    rl_metrics, r1_summary = evaluate_rl_agent(model, test_env, n_episodes=1)
    print("RL (PPO) TEST metrics:", rl_metrics)


if __name__ == "__main__":
    main()
