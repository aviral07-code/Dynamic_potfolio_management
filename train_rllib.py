# train_rllib.py

import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from data_utils import download_price_data, train_test_split_prices
from gym_env import PortfolioGymEnv
from evaluation import evaluate_rewards, summarize_returns


def env_creator(env_config):
    """
    RLlib env creator. env_config must contain 'prices'.
    """
    prices = env_config["prices"]
    lookback = env_config.get("lookback", 20)
    transaction_cost = env_config.get("transaction_cost", 0.0005)
    return PortfolioGymEnv(prices=prices, lookback=lookback, transaction_cost=transaction_cost)


def get_mean_reward_from_result(result: dict):
    """
    Robustly read a mean episode reward from an RLlib result dict,
    handling both old and new API stack formats.
    """
    # Try legacy key first
    if "episode_reward_mean" in result:
        return result["episode_reward_mean"]

    # Newer API often nests env metrics under "env_runners"
    env_runners = result.get("env_runners", {})
    if isinstance(env_runners, dict):
        # Common new key name: "episode_return_mean"
        if "episode_return_mean" in env_runners:
            return env_runners["episode_return_mean"]

        # Fall back to any float-like entry if available
        for k, v in env_runners.items():
            if isinstance(v, (int, float)):
                return v

    # Fallback: unknown structure, return None
    return None


def main():
    # 1. Load data
    tickers = ["SPY", "QQQ", "TLT", "GLD"]
    prices = download_price_data(tickers)
    train_prices, test_prices = train_test_split_prices(prices, train_ratio=0.7)

    # 2. Init Ray and register env
    ray.init(ignore_reinit_error=True)

    register_env("PortfolioBandit-v0", lambda config: env_creator(config))

    # 3. Configure PPO (old API stack disabled or not, this works with both)
    config = (
        PPOConfig()
        .environment(
            env="PortfolioBandit-v0",
            env_config={"prices": train_prices},
        )
        .framework("torch")
        # You can keep or remove api_stack override; get_mean_reward_from_result handles both.
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    algo = config.build()

    # 4. Train for some iterations
    for i in range(10):
        result = algo.train()
        mean_rew = get_mean_reward_from_result(result)
        print(f"Iteration {i}, episode_reward_mean={mean_rew}")

    # 5. Evaluation on test data (using compute_single_action; works on old stack)
    test_env = PortfolioGymEnv(prices=test_prices, lookback=20, transaction_cost=0.0005)
    obs, info = test_env.reset()
    done = False
    truncated = False
    rewards = []

    while not (done or truncated):
        action = algo.compute_single_action(obs)
        obs, reward, done, truncated, info = test_env.step(action)
        rewards.append(reward)
    rewards_arr = np.array(rewards, dtype=float)
    rl_metrics = evaluate_rewards(rewards_arr)
    rl_summary = summarize_returns(rewards_arr)

    print("RLlib PPO TEST metrics:", rl_metrics)
    print("RLlib PPO TEST summary:", rl_summary)

    # 6. Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
