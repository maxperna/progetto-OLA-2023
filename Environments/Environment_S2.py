import numpy as np

from param import std_noise_general

class BiddingEnvironment():
    '''Bidding Environment Class'''

    def __init__(self, bids, margin, rate, user):  
        """Initialize the Bidding Environment Class with the bids, means and sigmas."""

        # Assignments and Initializations
        self.bids = bids
        self.user = user
        self.margin = margin
        self.rate = rate
        self.means = margin*rate*user.click_vs_bid(bids) - user.cost_vs_bid(bids)*user.click_vs_bid(bids)  # real function
        self.sigmas = np.ones(len(bids)) * std_noise_general

    def round(self, pulled_arm):
        '''Simulate the current round of bidding with the given pulled arm. Returns the realization of a random normal with set mean and std.'''
        selected_bid = self.bids[pulled_arm]
        clicks = self.user.generate_click_bid_observations(selected_bid)
        costs = self.user.generate_cost_bid_observations(selected_bid)
        reward = self.margin*self.rate*clicks - costs*clicks
        return reward