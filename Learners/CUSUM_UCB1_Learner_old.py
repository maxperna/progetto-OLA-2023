import numpy as np
from Learners.Learner import Learner
from Learners.CUSUM import CUSUM


class CUSUM_UCB1_Learner_old(Learner):

    
    def __init__(self, n_arms, M, N, eps, threshold, alpha = 0.01): 
        super().__init__(n_arms) 

        self.M = M 
        self.upper_confidence_bounds = np.array([np.inf]*n_arms) 
        self.expected_rewards = np.zeros(n_arms) # vettore contente le medie campionarie del reward per ciascun arm A PARTIRE DALL'ULTIMO RESET
        self.n_pulls = np.zeros(n_arms) # vettore contenente il numero di pull fatti per ciascun arm A PARTIRE DALL'ULTIMO RESET
        
        self.change_detection = [CUSUM(N, eps, threshold) for _ in range(n_arms)]
        self.alpha = alpha

        self.n_arms = n_arms
        self.N = N
        self.eps = eps
        self.threshold = threshold

        self.detections = [] # vector storing changing instants 

        # RMK. from the superclass we inherit:
        # - self.reward_per_arm =  [[] for i in range(n_arms)] -> vector of vectors, one for each arm, storing the reward values collected pulling each arm
        # - self.collected_rewards = np.array([]) -> vector storing all rewards obtained
        # N.B. il significato di expected_rewards e n_pulls è cambiato, quello di reward_per_arm e collected_rewards no
        # N.B. anche il significato di t è cambiato, ora non rappresenta più l'instante globale ma l'istante di tempo A PARTIRE DALL'ULTIMO RESET 

        
    def pull_arm(self):
        
        if np.random.binomial(1, 1-self.alpha): # i.e. with probability 1-alpha:
            for a in range(self.n_arms):

                if self.n_pulls[a] == 0: # a partire dall'ultimo reset non ho mai pullato l'arm a:
                    self.upper_confidence_bounds[a] = np.inf  

                else: 
                    # RMK. nella formula usiamo self.t, che viene resettato ogni volta che viene rilevato un cambiamento della distribuzione
                    self.upper_confidence_bounds[a] = self.expected_rewards[a] + self.M * np.sqrt(2 * np.log(self.t) / (self.n_pulls[a])) 

            
            idxs = np.argwhere(self.upper_confidence_bounds == self.upper_confidence_bounds.max()).reshape(-1)
            pulled_arm = np.random.choice(idxs)

        else: # i.e. with probability alpha:
            pulled_arm = np.random.choice(range(0, self.n_arms)) # select a random arm

        return pulled_arm
    
  
    def update(self, pulled_arm, reward, price):
        
        print(pulled_arm)
        reset = self.change_detection[pulled_arm].update(reward) # oppure (price * reward) -> ciò che cambia è la scala, così è normalizzato nell'ingtervallo [0,1]

        if reset == True:
            self.detections = np.append(self.detections, self.detections[-1] + self.t)
            self.reset()
        
        self.t += 1
        self.n_pulls[pulled_arm] += 1

        #gain = (price-self.production_cost)*reward*self.n_clicks - self.cumulative_cost
        self.update_observations(pulled_arm, reward * price)  
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.n_pulls[pulled_arm] - 1) + reward * price) / self.n_pulls[pulled_arm] 

        
        # TODO: delete after check

        #self.valid_rewards_per_arm[pulled_arm].append(reward * price)
        #self.valid_collected_rewards = np.append(self.valid_collected_rewards, reward * price)
        
        
        #self.pulled_arms.append(pulled_arm) # aggiungo l'indice del nuovo arm pulled in coda alla lista 

        # if self.t > self.Tau: # aggiorno gli attributi del learner eliminando il contributo del round più vecchio (NB. facendolo ogni volta dal round Tau in poi, i contributii considerati rimarranno per sempre Tau da lì in avanti)
            
        #     a_old = self.pulled_arms[0] # indice dell'arm pullato nel round più vecchio 

        #     self.pulled_arms.pop(0) 
        #     self.n_pulls[a_old] -= 1
        #     self.reward_per_arm_tau[a_old].pop(0)
        #     self.collected_rewards_tau = np.delete(self.collected_rewards_tau, 0)

        #     if self.n_pulls[a_old] == 0: # era l'unico round in cui avevo pullato l'arm a_old -> riporto a zero l'expected reward
        #         self.expected_rewards[a_old] = 0
        #     else:
        #         self.expected_rewards[a_old] = sum(self.reward_per_arm_tau[a_old]) / self.n_pulls[a_old] # considero i rewards ottenuti pullando tale arm negli scorsi Tau rounds

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

    def reset(self):
        self.t = 0 
        self.expected_rewards = np.zeros(self.n_arms)
        self.n_pulls = np.zeros(self.n_arms)
        self.upper_confidence_bounds = np.array([np.inf]*self.n_arms)
        self.change_detection = [CUSUM(self.N, self.eps, self.threshold) for _ in range(self.n_arms)] # resettiamo manualmente tutte le istanze di CUSUM in quanto solo quella associata all'ultimo pulled_arm si resetta automaticamente

        
        