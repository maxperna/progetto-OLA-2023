import numpy as np

class ContextEnvironment():
    def __init__(self,actions,bids,sigma,user_set):
        """
                Initialize the Bidding Environment Class with the actions, means and sigmas.
        """

        # Assignments and Initializations
        self.actions = actions
        self.customers = user_set
        self.means=[]

        step = int(actions.shape[0] / len(bids))
        for user in user_set:
            tmp = []
            for i in range(1, len(bids) + 1):
                tmp[step * (i - 1):step * i] = user.demand_curve(user.prices) * np.repeat(user.click_vs_bid(bids[i - 1]), step)
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