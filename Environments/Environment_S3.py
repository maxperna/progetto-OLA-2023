import numpy as np

class PricingBiddingEnvironment():
    '''Pricing and Bidding Environment Class'''

    def __init__(self, actions, bids, sigma, user, production_cost):  
        """
        Initialize the Bidding Environment Class with the actions, means and sigmas.
        """

        # Assignments and Initializations
        self.actions = actions
        step=int(actions.shape[0]/len(bids))
        self.means=[]
        for i in range(1,len(bids)+1):
            self.means[step*(i-1):step*i] = (user.prices - production_cost) * user.demand_curve(user.prices) * np.repeat(user.click_vs_bid(bids[i-1]),step) - np.repeat(user.cumulative_cost_vs_bid(bids[i-1])*user.click_vs_bid(bids[i-1]),step)
        self.sigmas = np.ones(len(actions)) * sigma
        production_cost = production_cost

    def round(self, pulled_arm):
        '''Simulate the current round of pricing and bidding with the given pulled arm. Returns the realization of a random normal with set mean and std.'''
        reward = np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
        return reward
    