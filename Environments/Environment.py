import numpy as np
from Environments.Users import *


class Environment:
    avg_n_users = 100
    n_rounds = 365
    def __init__(self, n_arms = 5, user = UserC1()):
        self.n_arms = n_arms  # 5 prices
        self.user = user

    def round(self, pulled_arm):
        # Observation are communicated at the end of the day for all the customers
        selected_bid = 0.5
        n_clicks = self.user.click_vs_bid(selected_bid)
        reward = np.random.binomial(n_clicks, self.user.probabilities[pulled_arm])/n_clicks
        return reward
