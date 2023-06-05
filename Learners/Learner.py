import numpy as np
from abc import ABC
"""
Defining the class Learner as abstract class
"""

class Learner(ABC):
    """
    Abstract class to define the Learner
    :n_arms: the number of arms
    """
    def __init__(self,n_arms):
        self.n_arms = n_arms
        # Round we are currently in
        self.t = 0
        self.reward_per_arm =  [[] for i in range(n_arms)]
        # cumulative rewards for each arm
        self.collected_rewards = np.array([])

    """
    Method that updates the observations coming from the environment
    :pulled_arm: the played arm
    :reward: the reward given by the pulled arm
    """
    def update_observation(self,pulled_arm,reward):
        # Update the rewards given by an arm
        self.reward_per_arm[pulled_arm].append(reward)
        # Update the cumulative reward for the played arm
        self.collected_rewards = np.append(self.collected_rewards,reward)
