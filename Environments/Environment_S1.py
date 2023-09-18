import numpy as np
from Environments.Users import *


class Environment:
    
    def __init__(self, n_arms = 5, user = UserC1(), selected_bid = 0.5, production_cost = 100):
        self.n_arms = n_arms  # 5 prices
        self.user = user
        self.n_clicks = self.user.click_vs_bid(selected_bid)  # Use the known curve
        self.production_cost = production_cost
        self.cost_of_click = self.user.cost_vs_bid(selected_bid)

    def round(self, pulled_arm):
        # Observation are communicated at the end of the day for all the customers
        reward = np.random.binomial(self.n_clicks, self.user.probabilities[pulled_arm])
        return reward       # Number of items sold
    
    def max_reward(self):
        # Return the maximum reward given the selected bid
        opt = max((self.user.prices - self.production_cost) * self.user.probabilities * self.n_clicks - self.cost_of_click*self.n_clicks)
        return opt
    
