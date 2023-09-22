import numpy as np
from Learners.Learner import Learner
from Learners.CUSUM import CUSUM


class CUSUM_UCB1_Learner(Learner):

    
    def __init__(self, n_arms, production_cost, n_clicks, cost_of_click, M, N, eps, threshold, alpha = 0.01): 
        super().__init__(n_arms) 
        self.production_cost = production_cost
        self.n_clicks = n_clicks
        self.cost_of_click = cost_of_click

        self.expected_rewards = np.zeros(n_arms) # sample means of rewards collected starting from the last reset
        self.n_pulls = np.zeros(n_arms) # number of pulls for each arm starting from the last reset
        self.upper_confidence_bounds = np.array([np.inf]*n_arms) 
        self.M = M 
        
        self.change_detection = [CUSUM(N, eps, threshold) for _ in range(n_arms)]
        self.alpha = alpha

        self.n_arms = n_arms
        self.N = N
        self.eps = eps
        self.threshold = threshold

        self.counter = 0
        self.detections = [] # vector storing changing instants 

        # RMK. from the superclass we inherit:
        # - self.reward_per_arm =  [[] for i in range(n_arms)] -> vector of vectors, one for each arm, storing the reward values collected pulling each arm
        # - self.collected_rewards = np.array([]) -> vector storing all rewards obtained
        # RMK. each time a change is detected, self.t is reset to 0
        
    def pull_arm(self):
        
        if np.random.binomial(1, 1-self.alpha):                  # i.e. with probability 1-alpha:
            for a in range(self.n_arms):
                if self.n_pulls[a] == 0: 
                    self.upper_confidence_bounds[a] = np.inf  
                else:               
                    self.upper_confidence_bounds[a] = self.expected_rewards[a] + self.M * np.sqrt(2 * np.log(self.t) / (self.n_pulls[a])) 
            idxs = np.argwhere(self.upper_confidence_bounds == self.upper_confidence_bounds.max()).reshape(-1)
            pulled_arm = np.random.choice(idxs)
        
        else:                                                    # i.e. with probability alpha:
            pulled_arm = np.random.choice(range(0, self.n_arms)) # select a random arm

        return pulled_arm
    
  
    def update(self, pulled_arm, reward, price, current_phase = 0):

        if isinstance(self.n_clicks, np.ndarray):
            current_n_clicks = self.n_clicks[current_phase]
            current_cost_of_click = self.cost_of_click[current_phase]
        else:
            current_n_clicks = self.n_clicks
            current_cost_of_click = self.cost_of_click
        
        gain = (price-self.production_cost)*reward - current_cost_of_click*current_n_clicks
        normalized_reward = reward/current_n_clicks 
        #normalized_gain = gain / ((price-self.production_cost) * current_n_clicks - current_cost_of_click*current_n_clicks)
        reset = self.change_detection[pulled_arm].update(normalized_reward, pulled_arm) 
        if reset == True:
            self.detections = np.append(self.detections, self.t if len(self.detections) == 0 else self.detections[-1] + self.t)
            self.counter += 1
            self.reset()
        
        self.t += 1
        self.n_pulls[pulled_arm] += 1
        self.update_observations(pulled_arm, gain)  
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + gain) / self.n_pulls[pulled_arm] 


    def reset(self):
        self.t = 0 
        self.expected_rewards = np.zeros(self.n_arms)
        self.n_pulls = np.zeros(self.n_arms)
        self.upper_confidence_bounds = np.array([np.inf]*self.n_arms)
        self.change_detection = [CUSUM(self.N, self.eps, self.threshold) for _ in range(self.n_arms)] 
        
        