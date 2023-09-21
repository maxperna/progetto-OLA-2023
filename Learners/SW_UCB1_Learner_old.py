import numpy as np
from Learners.Learner import Learner


class SW_UCB1_Learner_old(Learner):

    
    def __init__(self, n_arms, M, Tau): 
        super().__init__(n_arms) 

        # N.B. il significato di expected_rewards e n_pulls è cambiato, quello di reward_per_arm e collected_rewards no

        self.expected_rewards = np.zeros(n_arms) # vettore contente le medie campionarie del reward per ciascun arm NEI PRECEDENTI TAU ROUNDS
        self.n_pulls = np.zeros(n_arms) # vettore contenente il numero di pull fatti per ciascun arm NEI PRECEDENTI TAU ROUNDS
        
        self.upper_confidence_bounds = np.array([np.inf]*n_arms) 
        self.M = M 

        self.Tau = Tau # sliding window's size 
        self.pulled_arms = [] # vector storing the sequence of pulled arms in previous Tau rounds

        self.reward_per_arm_tau =  [[] for i in range(n_arms)] # vector of vectors, one for each arm, storing the reward values collected pulling each arm IN THE PREVIOUS TAU ROUNDS
        self.collected_rewards_tau = np.array([]) # vector storing all rewards obtained IN THE PREVIOUS TAU ROUNDS

        # RMK. from the superclass we inherit:
        # - self.reward_per_arm =  [[] for i in range(n_arms)] -> vector of vectors, one for each arm, storing the reward values collected pulling each arm  
        # - self.collected_rewards = np.array([]) -> vector storing all rewards obtained 



        

    def pull_arm(self):

        for a in range(self.n_arms):

            if self.n_pulls[a] == 0: # nei precedenti Tau rounds non ho mai pullato l'arm a:
                self.upper_confidence_bounds[a] = np.inf  

            else: 
                self.upper_confidence_bounds[a] = self.expected_rewards[a] + self.M * np.sqrt(2 * np.log(self.t) / (self.n_pulls[a])) 

        
        idxs = np.argwhere(self.upper_confidence_bounds == self.upper_confidence_bounds.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm
    

        
    def update(self, pulled_arm, reward, price):

        self.t += 1
        self.n_pulls[pulled_arm] += 1

        #gain = (price-self.production_cost)*reward - self.cumulative_cost
        
        self.update_observations(pulled_arm, reward * price)  

        self.reward_per_arm_tau[pulled_arm].append(reward * price)
        self.collected_rewards_tau = np.append(self.collected_rewards_tau, reward * price)

        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + reward * price) / self.n_pulls[pulled_arm] 

        self.pulled_arms.append(pulled_arm) # aggiungo l'indice del nuovo arm pulled in coda alla lista 

        if self.t > self.Tau: # aggiorno gli attributi del learner eliminando il contributo del round più vecchio (NB. facendolo ogni volta dal round Tau in poi, i contributii considerati rimarranno per sempre Tau da lì in avanti)
            # TODO: verifica che così i vettori abbiano esattamente Tau elementi
            
            a_old = self.pulled_arms[0] # indice dell'arm pullato nel round più vecchio 

            self.pulled_arms.pop(0) 
            self.n_pulls[a_old] -= 1
            self.reward_per_arm_tau[a_old].pop(0)
            self.collected_rewards_tau = np.delete(self.collected_rewards_tau, 0)

            if self.n_pulls[a_old] == 0: # era l'unico round in cui avevo pullato l'arm a_old -> riporto a zero l'expected reward
                self.expected_rewards[a_old] = 0
            else:
                self.expected_rewards[a_old] = sum(self.reward_per_arm_tau[a_old]) / self.n_pulls[a_old] # considero i rewards ottenuti pullando tale arm negli scorsi Tau rounds

        '''
        # printing: 
        print('time:')
        print(self.t)
        print('UCBs:')
        print(self.upper_confidence_bounds)
        print('n_pulls:')
        print(self.n_pulls)
        #print('collected_rewards:')
        #print(self.collected_rewards)
        print('collected_rewards_tau:')
        print(self.collected_rewards_tau)
        print('pulled_arms:')
        print(self.pulled_arms)
        print('expected_rewards:')
        print(self.expected_rewards)
        print('')
        '''
        