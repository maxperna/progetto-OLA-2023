import numpy as np
from Learners.Learner import Learner

class UCB1_Learner_old(Learner):

    def __init__(self, n_arms, M): 
        super().__init__(n_arms) # chiamo il constructor della superclass Learner

        self.expected_rewards = np.zeros(n_arms) # vettore in cui salvo le medie campionarie dei reward per ciascun arm
        self.n_pulls = np.zeros(n_arms) # vettore contnente il numero di pull fatti per ciascun arm
        self.upper_confidence_bounds = np.array([np.inf]*n_arms) # vettore contenente gli upper confidence bound della conversion probability (o del reward) per ciascun arm 
        self.M = M          # max price (for UCB1)

    def pull_arm(self):

        for a in range(self.n_arms):
            
            # play once every arm:
            if self.n_pulls[a] == 0: # ancora non ho mai pullato l'arm a
                self.upper_confidence_bounds[a] = np.inf  # in case denominator is 0, i.e. the number of times arm a has been pulled is 0 -> set the UCB to inf in order to be sure to pull such arm at least once 
            
            # compute the updated UCB for each arm:
            else: # RMK. gli upper conf. bounds di ogni arm vanno aggiornati ad ogni time-step, non solo se l'arm associato viene giocato
                self.upper_confidence_bounds[a] = self.expected_rewards[a] + self.M * np.sqrt(2 * np.log(self.t) / (self.n_pulls[a])) 
        
        # arm to be pooled:
        idxs = np.argwhere(self.upper_confidence_bounds == self.upper_confidence_bounds.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs) 
        
        return pulled_arm
    
    def update(self, pulled_arm, reward, price):
        self.t += 1
        self.n_pulls[pulled_arm] += 1

        #gain = (price-self.production_cost)*reward*self.n_clicks - self.cumulative_cost
        
        self.update_observations(pulled_arm, reward * price)     
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + reward * price) / self.n_pulls[pulled_arm]  