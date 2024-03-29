from Learners.Learner import *

class Greedy_Learner (Learner):
    def __init__(self, n_arms, production_cost, n_clicks, cost_of_click):
        super().__init__(n_arms)
        self.production_cost = production_cost
        self.n_clicks = n_clicks
        self.cost_of_click = cost_of_click
    
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
        gain = (price-self.production_cost)*reward - self.cost_of_click*self.n_clicks
        self.update_observations(pulled_arm, gain)  
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) + gain) / self.t
    
