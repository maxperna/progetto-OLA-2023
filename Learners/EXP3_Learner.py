import numpy as np
import math
import random
from Learners.Learner import Learner


class EXP3_Learner(Learner):
    def __init__(self, n_arms, gamma, clicks, cost, production_cost):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf] * n_arms)
        self.production_cost = production_cost
        self.clicks = clicks
        self.cost = cost
        self.gamma = gamma
        self.weights = np.full(n_arms, 1.0 / n_arms)
        self._initial_exploration = np.random.permutation(n_arms)

    @property
    def p(self):
        p = ((1 - self.gamma) * self.weights) + (self.gamma / self.n_arms)
        if not np.all(np.isfinite(p)):
            p[~np.isfinite(p)] = 0
        if np.isclose(np.sum(p), 0):
            p[:] = 1.0 / self.n_arms
        return p / np.sum(p)

    def pull_arm(self):
        if self.t < self.n_arms:
            return self._initial_exploration[self.t]
        else:
            return np.random.choice(self.n_arms, p=self.p)

    def update(self, pulled_arm, reward, price):
        reward = (price - self.production_cost) * reward - self.clicks * self.cost
        self.weights[pulled_arm] *= np.exp(reward * (self.gamma / self.n_arms))
        self.weights /= np.sum(self.weights)
        self.t += 1
        super().update_observations(pulled_arm, reward)
