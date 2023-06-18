from Learners.Learner import *

class Greedy_Learner (Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)

    def pull_arm(self):
        """
        Method that selects the arm to pull at time t
        """
        if (self.t < self.n_arms):
            return self.t
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm
    
    def update (self, pulled_arm, reward, price):
        self.t += 1
        self.update_observations(pulled_arm, reward*price)   # update with reward*price
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) + reward*price) / self.t
    
