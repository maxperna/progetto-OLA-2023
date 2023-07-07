import numpy as np

class BiddingEnvironment():
    '''Bidding Environment Class'''

    def __init__(self, bids, sigma, user):  
        """Initialize the Bidding Environment Class with the bids, means and sigmas."""

        # Assignments and Initializations
        self.bids = bids
        self.means = user.demand_curve(user.prices) + user.click_vs_bid(bids) # real function # I try with the sum
        self.sigmas = np.ones(len(bids)) * sigma 

    def round(self, pulled_arm):
        '''Simulate the current round of bidding with the given pulled arm. Returns the realization of a random normal with set mean and std.'''
        reward = np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
        return reward
