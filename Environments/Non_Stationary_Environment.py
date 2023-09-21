import numpy as np
from Environments.Environment_S1 import Environment 
from Environments.Users import *

class Non_Stationary_Environment(Environment): 
    
    def __init__(self, selected_bid, production_cost, n_arms, user, horizon):
        super().__init__(selected_bid, production_cost, n_arms, user) # TODO: verificare se è necessario eliminare la default initialization dal costruttore della superclass, che lo inizializza con la classe C1 stazionaria di user
        self.t = 0
        n_phases = len(self.user._probabilities) 
        self.phase_size = np.ceil(horizon/n_phases)

    def round(self, pulled_arm): 
        # N.B. stesso nome e stessi input del metodo della superclass, va bene? Chiamato da un oggetto della classe figli dovrebbe avere il comportamento corretto
        current_phase = int(self.t / self.phase_size) # FIXME
        p = self.user._probabilities[current_phase][pulled_arm]
        reward = np.random.binomial(self.n_clicks, p) 
        self.t += 1
        return reward
    
    def max_reward(self):
        # Return the maximum reward given the selected bid
        current_phase = int(self.t / self.phase_size)
        opt = max((self.user.prices - self.production_cost) * self.user.probabilities[current_phase] * self.n_clicks - self.cost_of_click*self.n_clicks)
        return opt
