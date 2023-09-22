import numpy as np
from Learners.Learner import Learner


class SW_UCB1_Learner(Learner):

    
    def __init__(self, n_arms, production_cost, n_clicks, cost_of_click, M, Tau): 
        super().__init__(n_arms) 
        self.production_cost = production_cost
        self.n_clicks = n_clicks
        self.cost_of_click = cost_of_click

        self.expected_rewards = np.zeros(n_arms) # sample means of rewards from previous Tau rounds 
        self.n_pulls = np.zeros(n_arms) # number of pulls for each arm in the previous Tau rounds
        self.upper_confidence_bounds = np.array([np.inf]*n_arms) 
        self.M = M 

        self.Tau = Tau # sliding window's size 
        self.pulled_arms = [] # vector storing the sequence of pulled arms in previous Tau rounds
        self.reward_per_arm_tau =  [[] for i in range(n_arms)] # vector of vectors, one for each arm, storing the reward values collected pulling each arm in the previous Tau rounds
        self.collected_rewards_tau = np.array([]) # vector storing all rewards obtained in the previous Tau rounds

        # RMK. from the superclass we inherit:
        # - self.reward_per_arm =  [[] for i in range(n_arms)] -> vector of vectors, one for each arm, storing the reward values collected pulling each arm  
        # - self.collected_rewards = np.array([]) -> vector storing all rewards obtained 

    def pull_arm(self):
        for a in range(self.n_arms):
            if self.n_pulls[a] == 0: # in the last Tau rounds arm a has never been pulled
                self.upper_confidence_bounds[a] = np.inf  
            else: 
                self.upper_confidence_bounds[a] = self.expected_rewards[a] + self.M * np.sqrt(2 * np.log(self.t) / (self.n_pulls[a])) 
        idxs = np.argwhere(self.upper_confidence_bounds == self.upper_confidence_bounds.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm
    
    def update(self, pulled_arm, reward, price, current_phase = 0):

        if isinstance(self.n_clicks, np.ndarray):
            current_n_clicks = self.n_clicks[current_phase]
            current_cost_of_click = self.cost_of_click[current_phase]
        else:
            current_n_clicks = self.n_clicks
            current_cost_of_click = self.cost_of_click


        self.t += 1
        self.n_pulls[pulled_arm] += 1
        gain = (price-self.production_cost)*reward - current_cost_of_click*current_n_clicks
        self.update_observations(pulled_arm, gain)  

        self.reward_per_arm_tau[pulled_arm].append(gain)
        self.collected_rewards_tau = np.append(self.collected_rewards_tau, gain)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + gain) / self.n_pulls[pulled_arm] 
        self.pulled_arms.append(pulled_arm)

        if self.t > self.Tau: # update learner's attributes removing the contribution of the oldest round
            # TODO: verifica che così i vettori abbiano esattamente Tau elementi
            a_old = self.pulled_arms[0]
            self.pulled_arms.pop(0) 
            self.n_pulls[a_old] -= 1
            self.reward_per_arm_tau[a_old].pop(0)
            self.collected_rewards_tau = np.delete(self.collected_rewards_tau, 0)

            if self.n_pulls[a_old] == 0: 
                self.expected_rewards[a_old] = 0
            else:
                self.expected_rewards[a_old] = sum(self.reward_per_arm_tau[a_old]) / self.n_pulls[a_old] 

        