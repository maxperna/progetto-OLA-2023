import numpy as np
from Environments.Users import *


class Environment:
    
    avg_n_users = 100
    n_rounds = 365
    selected_bid = 0.5

    def __init__(self, n_arms = 5, user = UserC1()):
        self.n_arms = n_arms  # 5 prices
        self.user = user
        self.n_clicks = self.user.click_vs_bid(Environment.selected_bid)

    def round(self, pulled_arm):
        # Observation are communicated at the end of the day for all the customers
        reward = np.random.binomial(self.n_clicks, self.user.probabilities[pulled_arm])/self.n_clicks
        return reward
    
    def max_reward(self):
        # Return the maximum reward given the selected bid
        return Environment.avg_n_users * np.max(self.user.reward_of_prices)