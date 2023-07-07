import numpy as np

class PricingBiddingEnvironment():
    '''Pricing and Bidding Environment Class'''

    def __init__(self, actions, bids, sigma, user):  
        """Initialize the Bidding Environment Class with the bids, means and sigmas."""

        # Assignments and Initializations
        self.actions = actions
        step=5
        for i in range(1,50,1):
            self.means[1:step*i] = user.demand_curve(user.prices) + user.click_vs_bid(bids[1:step]) # real function # I try with the sum
        self.sigmas = np.ones(len(bids)) * sigma 

    def round(self, pulled_arm):
        '''Simulate the current round of bidding with the given pulled arm. Returns the realization of a random normal with set mean and std.'''
        reward = np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
        return reward
