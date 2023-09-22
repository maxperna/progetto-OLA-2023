import numpy as np
from Environments.Environment_S1 import Environment 
from Environments.Users import *

class Non_Stationary_Environment(Environment): 
    
    def __init__(self, selected_bid, production_cost, n_arms, user, horizon):
        super().__init__(selected_bid, production_cost, n_arms, user) 
        self.t = 0
        n_phases = len(self.user._probabilities) 
        self.phase_size = np.ceil(horizon/n_phases)

    def round(self, pulled_arm): 
        current_phase = int(self.t / self.phase_size)
        current_n_clicks = self.n_clicks[current_phase]
        p = self.user._probabilities[current_phase][pulled_arm]
        reward = np.random.binomial(current_n_clicks, p) 
        self.t += 1
        return reward