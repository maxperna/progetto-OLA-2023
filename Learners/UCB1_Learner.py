import numpy as np
import matplotlib.pyplot as plt

from Environments.Users import UserC1, UserC2, UserC3
from Learners.Learner import Learner


class UCB1_Learner(Learner):
    def __init__(self, n_arms, production_cost, n_clicks, cost_of_click, M):
        super().__init__(n_arms)
        self.production_cost = production_cost
        self.n_clicks = n_clicks
        self.cost_of_click = cost_of_click

        self.expected_rewards = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)
        self.upper_confidence_bounds = np.array([np.inf] * n_arms)
        self.M = M  # max gain (for UCB1)

    def pull_arm(self):
        for a in range(self.n_arms):
            if self.n_pulls[a] == 0:
                self.upper_confidence_bounds[a] = np.inf  # in case denominator is 0
            else:
                self.upper_confidence_bounds[a] = self.expected_rewards[
                    a
                ] + self.M * np.sqrt(2 * np.log(self.t) / (self.n_pulls[a]))
        idxs = np.argwhere(
            self.upper_confidence_bounds == self.upper_confidence_bounds.max()
        ).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update(self, pulled_arm, reward, price, current_phase=0):
        # non-stationary extension:
        if isinstance(self.n_clicks, np.ndarray):
            current_n_clicks = self.n_clicks[current_phase]
            current_cost_of_click = self.cost_of_click[current_phase]
        else:
            current_n_clicks = self.n_clicks
            current_cost_of_click = self.cost_of_click

        self.t += 1
        self.n_pulls[pulled_arm] += 1
        gain = (
            price - self.production_cost
        ) * reward - current_cost_of_click * current_n_clicks
        self.update_observations(pulled_arm, gain)
        self.expected_rewards[pulled_arm] = (
            self.expected_rewards[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + gain
        ) / self.n_pulls[pulled_arm]
