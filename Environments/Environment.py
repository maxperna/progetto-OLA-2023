import numpy as np
from Environments.Users import *


class Environment:
    def __init__(self, n_arms = 5, user = UserC1()):
        self.n_rounds = 365  # 365 day long
        self.n_arms = n_arms  # 5 prices
        self.user = user
        self.avg_n_users = 100

    def round(self, pulled_arm):
        # Observation are communicated at the end of the day for all the customers
        reward = np.random.binomial(1, self.user.probabilities[pulled_arm])
        return reward
