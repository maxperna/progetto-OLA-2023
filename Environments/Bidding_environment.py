import numpy as np

class BiddingEnvironment():
    '''Bidding Environment Class'''

    def __init__(self, bids, sigma, user):
        '''Initialize the Bidding Environment Class with a list of budgets for each subcampaign, sigma and the current subcampaign'''

        # Assignments and Initializations
        self.bids = bids
        self.means = user.generate_observations(bids)
        self.sigmas = np.ones(len(bids)) * sigma 

    def round(self, pulled_arm):
        '''Simulate the current round of bidding with the given pulled arm. Returns the realization of a random normal with set mean and std.'''
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
