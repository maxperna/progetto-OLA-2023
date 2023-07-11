import numpy as np

class PricingBiddingEnvironment():
    '''Pricing and Bidding Environment Class'''

    def __init__(self, actions, bids, sigma, user):  
        """Initialize the Bidding Environment Class with the bids, means and sigmas."""

        # Assignments and Initializations
        self.actions = actions
        step=5
        self.means=[]
        for i in range(1,101):
            self.means[step*(i-1):step*i] = user.demand_curve(user.prices) * np.repeat(user.click_vs_bid(bids[i-1]),step)
        self.sigmas = np.ones(len(actions)) * sigma 

    def round(self, pulled_arm):
        '''Simulate the current round of bidding with the given pulled arm. Returns the realization of a random normal with set mean and std.'''
        reward = np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
        return reward
    