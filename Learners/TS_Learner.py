import numpy as np
from Learners.Learner import *


class TSLearner(Learner):
    def __init__(self,n_arms):
        super().__init__(n_arms)
        # Beta and Alpha parameters
        self.beta_param = np.ones((n_arms,2))

    """
    Method that selects the arm to pull at time t
    """
    def pull_arm(self):
        # Get the index associated to the largest reward
        max_ind = np.argmax(np.random.beta(self.beta_param[:,0],self.beta_param[:,1]))
        return max_ind

    """
    Method to call update methods of superclass Learner
    """
    def update(self,pulled_arm,reward):
        # Increase time
        self.t += 1
        self.update_observation(pulled_arm,reward)
        # Update parameters
        self.beta_param[pulled_arm,0]=self.beta_param[pulled_arm,1] + 1.0 - reward