import numpy as np
import math
import random
from Learners.Learner import Learner

class EXP3_Learner(Learner):

    def __init__(self, n_arms, gamma): #, rewardMin = 0, rewardMax = 1): 
        super().__init__(n_arms) # chiamo il constructor della superclass Learner

        self.gamma = gamma
        #self.rewardMin = rewardMin 
        #self.rewardMax = rewardMax
        self.weights = [1.0] * n_arms
        self.probs = np.zeros(n_arms)
        #self.estimated_rewards = np.zeros(n_arms)


    def pull_arm(self):

        for a in range(self.n_arms):
            self.probs[a] = (1.0-self.gamma) * self.weights[a] * (1/float(sum(self.weights))) + (self.gamma/self.n_arms)

        # arm to be pooled: 
        pulled_arm = self.draw(self.probs)
        
        return pulled_arm
    
    
    def update(self, pulled_arm, reward, price, prices):
        
        self.t += 1

        rewardMax = np.max(prices) 
        rewardMin = 0 # rewardMin deve essere zero in quanto il reward, in caso di non vendita è nullo -> da cambiare

        scaled_reward = (reward * price - rewardMin) / (rewardMax - rewardMin) # rewards scaled to 0,1

        estimated_reward = 1.0 * scaled_reward / self.probs[pulled_arm]
        self.weights[pulled_arm] *= math.exp(estimated_reward * self.gamma / self.n_arms) # important that we use estimated reward here!

        #gain = (price-self.production_cost)*reward*self.n_clicks - self.cumulative_cost
        self.update_observations(pulled_arm, reward * price)     
        
    
    # draw: [float] -> int
    # pick an index from the given list of floats proportionally
    # to the size of the entry (i.e. normalize to a probability
    # distribution and draw according to the probabilities).
    def draw(weights):
        choice = random.uniform(0, sum(weights))
        choiceIndex = 0

        for weight in weights:
            choice -= weight
            if choice <= 0:
                return choiceIndex
            choiceIndex += 1