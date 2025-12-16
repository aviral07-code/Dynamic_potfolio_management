# main.py
import numpy as np
from data_utils import download_price_data, train_test_split_prices
from env_wrapper import BanditPortfolioEnv
from bandit_agent import ContextualBanditPortfolio
from baselines import run_equal_weight_baseline, run_greedy_past_mean
from evaluation import evaluate_rewards, summarize_returns

def run_bandit_on_env(env: BanditPortfolioEnv, n_warmup: int = 50):
    """
    Run contextual bandit online on the environment.
    n_warmup: number of steps of random exploration before using the learned policy.
    """
    ctx = env.reset()
    done = False
    n_arms = env.action_space_n()
    n_features = ctx.shape[0]

    agent = ContextualBanditPortfolio(n_arms=n_arms, n_context_features=n_features)

    contexts_buffer = []
    actions_buffer = []
    rewards_buffer = []
    rewards_all = []

    step = 0
    while not done:
        if step < n_warmup or not agent._is_fit:
            action = np.random.randint(n_arms)
        else:
            action = agent.predict(ctx)

        next_ctx, reward, done, info = env.step(action)

        contexts_buffer.append(ctx)
        actions_buffer.append(action)
        rewards_buffer.append(reward)
        rewards_all.append(reward)

        # Online update at each step
        contexts_np = np.array(contexts_buffer, dtype=np.float32)
        actions_np = np.array(actions_buffer, dtype=int)
        rewards_np = np.array(rewards_buffer, dtype=float)
        agent.partial_fit(contexts_np, actions_np, rewards_np)

        ctx = next_ctx
        step += 1

    return np.array(rewards_all, dtype=float)

def main():
    tickers = ["SPY", "QQQ", "TLT", "GLD"]  # equities, bonds, gold, etc.
    prices = download_price_data(tickers)
    train_prices, test_prices = train_test_split_prices(prices, train_ratio=0.7)

    # Training environment (could be used for hyperparameter tuning or offline analysis)
    train_env = BanditPortfolioEnv(train_prices, lookback=20, transaction_cost=0.0005)
    train_rewards = run_bandit_on_env(train_env, n_warmup=50)
    train_metrics = evaluate_rewards(train_rewards)
    train_summary = summarize_returns(train_rewards)
    print("Bandit TRAIN metrics:", train_metrics)
    print("Bandit TRAIN summary:", train_summary)
    # Test environment for final evaluation
    test_env = BanditPortfolioEnv(test_prices, lookback=20, transaction_cost=0.0005)
    bandit_test_rewards = run_bandit_on_env(test_env, n_warmup=50)
    bandit_test_metrics = evaluate_rewards(bandit_test_rewards)
    bandit_test_summary = summarize_returns(bandit_test_rewards)
    print("Bandit TEST metrics:", bandit_test_metrics)
    print("Bandit TRAIN summary:", bandit_test_summary)

    # Baselines on the same test segment
    test_env_eq = BanditPortfolioEnv(test_prices, lookback=20, transaction_cost=0.0005)
    eq_rewards = run_equal_weight_baseline(test_env_eq)
    eq_metrics = evaluate_rewards(np.array(eq_rewards))
    eq_summary = summarize_returns(eq_rewards)
    print("Equal-weight baseline metrics:", eq_metrics)
    print("Equal-weight baseline summary:", eq_summary)

    test_env_greedy = BanditPortfolioEnv(test_prices, lookback=20, transaction_cost=0.0005)
    greedy_rewards = run_greedy_past_mean(test_env_greedy)
    greedy_metrics = evaluate_rewards(np.array(greedy_rewards))
    greedy_summary = summarize_returns(greedy_rewards)
    print("Greedy past-mean baseline metrics:", greedy_metrics)
    print("Greedy past-mean baseline summary:", greedy_summary)

if __name__ == "__main__":
    main()
