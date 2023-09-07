import numpy as np
from Learners.Learner import *


class TS_Learner(Learner):
    def __init__(self,n_arms, production_cost, n_clicks, cumulative_cost):
        super().__init__(n_arms)
        self.production_cost = production_cost
        self.n_clicks = n_clicks
        self.cumulative_cost = cumulative_cost
        
        # Beta and Alpha parameters
        self.beta_param = np.ones((n_arms,2))

    """
    Method that selects the arm to pull at time t
    """
    def pull_arm(self, prices):
        # Get the index associated to the largest reward
        max_ind = np.argmax(np.random.beta(self.beta_param[:,0],self.beta_param[:,1]) * prices)
        return max_ind

    """
    Method to call update methods of superclass Learner
    """
    def update(self,pulled_arm,reward, price):
        # Increase time
        self.t += 1
        gain = (price-self.production_cost)*reward*self.n_clicks - self.cumulative_cost
        self.update_observations(pulled_arm, gain) # TODO update with reward*price
        # Update parameters
        self.beta_param[pulled_arm,0] = self.beta_param[pulled_arm,0] + reward
        self.beta_param[pulled_arm,1] = self.beta_param[pulled_arm,1] + 1.0 - reward