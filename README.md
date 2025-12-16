# Dynamic Portfolio Management with Bandit Algorithms

This project implements a dynamic portfolio allocation framework using contextual bandit algorithms and deep reinforcement learning (PPO). The environment models daily allocation among multiple financial assets with transaction costs, and compares:

- Contextual Bandit (LinUCB, via MABWiser)
- PPO (Stable-Baselines3)
- PPO (RLlib / Ray)
- Equal-weight baseline
- Greedy past-mean baseline

The goal is to evaluate return, risk, and transaction costs in a nonstationary market setting.

---

## 1. Project Structure

dynamic_portfolio_bandits/
bandit_agent.py # Contextual bandit (LinUCB) wrapper
baselines.py # Equal-weight and greedy baselines
data_utils.py # Download and split historical price data
env_wrapper.py # Core multi-asset bandit environment
gym_env.py # Gymnasium-compatible environment wrapper
evaluation.py # Metrics: cumulative return, Sharpe, drawdown, summaries
main.py # Contextual bandit + baselines experiments
train_sb3.py # PPO (Stable-Baselines3) training and evaluation
train_rllib.py # PPO (RLlib) training and evaluation


---

## 2. Installation

### 2.1. Create and activate virtual environment (macOS)

cd Project_SDM
python3 -m venv .venv
source .venv/bin/activate


### 2.2. Install dependencies

pip install --upgrade pip

Core libraries
pip install numpy pandas matplotlib yfinance

Bandit library
pip install mabwiser

Gymnasium and environment tooling
pip install gymnasium

Stable-Baselines3 (PPO)
pip install 'stable-baselines3[extra]>=2.0.0a1'

RLlib / Ray (optional, for RLlib PPO)
pip install 'ray[rllib]'


> Note: On zsh, the quotes around `package[extra]` are necessary to avoid shell globbing errors.

---

## 3. Data

The project downloads daily adjusted close prices for four assets:

- SPY, QQQ, TLT, GLD

using the `yfinance` API. The default date range is 2018-01-01 to 2024-12-31, with a 70/30 train–test split by time.

If you want to change the tickers or date range, edit `download_price_data` in `data_utils.py`.

---

## 4. Running Experiments

### 4.1 Contextual Bandit + Baselines

From the project root:

source .venv/bin/activate
python dynamic_portfolio_bandits/main.py


This script:

- Downloads data and splits into train/test.
- Runs LinUCB contextual bandit on train and test.
- Runs equal-weight and greedy past-mean baselines on the test segment.
- Prints:

  - `Bandit TRAIN metrics` + summary (mean/std daily returns, steps)
  - `Bandit TEST metrics` + summary
  - `Equal-weight baseline metrics` + summary
  - `Greedy past-mean baseline metrics` + summary

These values are used to fill the results tables in the report.

---

## 5. PPO (Stable-Baselines3) Experiment

### 5.1. Train PPO

source .venv/bin/activate
python dynamic_portfolio_bandits/train_sb3.py


This script:

- Builds the `PortfolioGymEnv` from the training price data.
- Trains a PPO agent (MLP policy) for a specified number of timesteps.
- Evaluates PPO on the test period using the same reward as the bandit and baselines.
- Prints:

  - `RL (PPO) TEST metrics: {...}`
  - `RL (PPO) TEST summary: {...}`

These include cumulative return, Sharpe ratio, max drawdown, mean daily return, std daily return, and the number of trading steps.

The trained model is saved (e.g., as `ppo_portfolio_model.zip`) and can be reloaded for further analysis.

---

## 6. PPO (RLlib) Experiment (Optional)

### 6.1. Train and Evaluate PPO with RLlib

source .venv/bin/activate
python dynamic_portfolio_bandits/train_rllib.py


This script:

- Initializes Ray and registers `PortfolioGymEnv` under `PortfolioBandit-v0`.
- Configures PPO to use the same environment and uses the legacy API stack for evaluation.
- Trains for a fixed number of iterations.
- Evaluates the RLlib PPO policy on the test data.
- Prints:

  - `Iteration i, episode_reward_mean=...` during training
  - `RLlib PPO TEST metrics: {...}`
  - `RLlib PPO TEST summary: {...}`

As with the SB3 PPO, these metrics are directly comparable to the bandit and baselines.

---

## 7. Key Files Summary

- `env_wrapper.py`: Implements the core multi-asset bandit environment with rolling mean/volatility context and transaction costs.
- `gym_env.py`: Wraps the environment into a Gymnasium-compatible class for PPO.
- `bandit_agent.py`: Defines the LinUCB-based contextual bandit agent using MABWiser.
- `baselines.py`: Implements equal-weight and greedy past-mean strategies.
- `evaluation.py`: Provides functions to compute cumulative return, Sharpe ratio, max drawdown, and per-step summaries.
- `main.py`: Entry point for bandit and baseline experiments.
- `train_sb3.py`: Entry point for PPO (Stable-Baselines3) experiment.
- `train_rllib.py`: Entry point for PPO (RLlib) experiment.

---

## 8. Reproducing Report Tables

Run:

1. `python dynamic_portfolio_bandits/main.py`
2. `python dynamic_portfolio_bandits/train_sb3.py`
3. `python dynamic_portfolio_bandits/train_rllib.py`

Collect the printed `metrics` and `summary` dictionaries for each method and insert them into your LaTeX report tables:

- Contextual Bandit (TEST)
- PPO (SB3) TEST
- PPO (RLlib) TEST
- Equal-weight baseline TEST
- Greedy past-mean baseline TEST

---

## 9. Notes and Troubleshooting

- If you see SSL or LibreSSL warnings from HTTP libraries, they are usually harmless for this project.
- If installation of extras such as `stable-baselines3[extra]` or `gymnasium[all]` fails on zsh, make sure they are quoted: `'stable-baselines3[extra]'`.
- Large training runs for PPO may take several minutes depending on hardware; you can reduce `total_timesteps` in `train_sb3.py` for quicker tests.

---

