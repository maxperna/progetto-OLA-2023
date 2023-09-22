import numpy as np

from param import production_cost

class ContextEnvironment():
    def __init__(self,actions,bids,sigma,user_set):
        """
                Initialize the Bidding Environment Class with the actions, means and sigmas.
        """

        # Assignments and Initializations
        self.actions = actions
        self.customers = [item for sublist in list(user_set.values()) for item in sublist]
        self.means=[]

        step = int(actions.shape[0] / len(bids))
        for user in self.customers:
            tmp = []
            for i in range(1, len(bids) + 1):
                #tmp[step * (i - 1):step * i] = user.demand_curve(user.prices) * np.repeat(user.click_vs_bid(bids[i - 1]), step)  # FIXME change this line with new reward (suggestion below)
                tmp[step*(i-1):step*i] = (user.prices - production_cost) * user.demand_curve(user.prices) * np.repeat(user.click_vs_bid(bids[i-1]),step) - np.repeat(user.cost_vs_bid(bids[i-1])*user.click_vs_bid(bids[i-1]),step)
            self.means.append(tmp)

        self.sigmas = np.ones(len(actions)) * sigma
        self._current_feature = None
        self._customer_id = None

    def get_current_features(self):
        customer_type = np.random.choice(range(len(self.customers)))
        self._customer_id = customer_type
        self._current_feature = self.customers[customer_type].get_features
        return self._current_feature


    def round(self, pulled_arm):
        '''Simulate the current round of pricing and bidding with the given pulled arm. Returns the realization
        of a random normal with set mean and std.'''
        #Select a random customer from a uniform distribution
        reward = np.random.normal(self.means[self._customer_id][pulled_arm], self.sigmas[pulled_arm])

        return reward