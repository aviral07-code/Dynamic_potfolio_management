# bandit_agent.py
import numpy as np
from typing import Tuple
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

class ContextualBanditPortfolio:
    """
    Wrapper around MABWiser contextual bandit for portfolio allocation.
    """

    def __init__(self, n_arms: int, n_context_features: int):
        """
        n_arms: number of assets.
        n_context_features: dimensionality of context vector.
        """
        self.n_arms = n_arms
        self.n_context_features = n_context_features

        # Use LinUCB as a starting point (parametric contextual bandit).
        # Other choices: LinTS (Thompson sampling), EpsilonGreedy with context, etc.
        learning_policy = LearningPolicy.LinUCB(alpha=1.0)

        # No neighborhood policy for now (global model).
        self.mab = MAB(
            arms=list(range(n_arms)),
            learning_policy=learning_policy
        )

        self._is_fit = False

    def partial_fit(self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """
        Update bandit with observed contexts, actions, and rewards.
        contexts: shape [n_samples, n_features]
        actions: shape [n_samples]
        rewards: shape [n_samples]
        """
        self.mab.partial_fit(contexts=contexts, decisions=actions, rewards=rewards)
        self._is_fit = True

    def predict(self, context: np.ndarray) -> int:
        """
        Select an arm for a single context.
        """
        ctx = context.reshape(1, -1)
        if not self._is_fit:
            # If no data yet, choose random arm.
            return np.random.randint(self.n_arms)
        decision = self.mab.predict(contexts=ctx)
        return int(decision)
