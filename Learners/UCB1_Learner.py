import numpy as np
import matplotlib.pyplot as plt 

from Environments.Users import  UserC1, UserC2, UserC3
from Environments.Environment import Environment
from Learners.Learner import Learner

class UCB1_Learner(Learner):

    def __init__(self, n_arms, production_cost, n_clicks, cumulative_cost, M):
        super().__init__(n_arms)
        self.production_cost = production_cost
        self.n_clicks = n_clicks
        self.cumulative_cost = cumulative_cost

        self.expected_rewards = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)
        self.upper_confidence_bounds = np.array([np.inf]*n_arms)
        self.M = M          # max price (for UCB1)

    def pull_arm(self):
        for a in range(self.n_arms):
            if self.n_pulls[a] == 0:
                self.upper_confidence_bounds[a] = np.inf  # in case denominator is 0
            else:
                self.upper_confidence_bounds[a] = self.expected_rewards[a] + self.M * np.sqrt(2 * np.log(self.t) / (self.n_pulls[a]))
        idxs = np.argwhere(self.upper_confidence_bounds == self.upper_confidence_bounds.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm
        
    def update(self, pulled_arm, reward, price):
        self.t += 1
        self.n_pulls[pulled_arm] += 1
        #gain = (price-self.production_cost)*reward*self.n_clicks - self.cumulative_cost
        gain = (price-self.production_cost)*reward*self.n_clicks - self.cumulative_cost*self.n_clicks
        self.update_observations(pulled_arm, gain)      # TODO update with reward*price
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + gain) / self.n_pulls[pulled_arm]  # TODO check t_elemnt