import numpy as np
from Users import UserC1,UserC2,UserC3


class ContextEnvironment():
    def __init__(self,actions,bids,sigma):
        """
                Initialize the Bidding Environment Class with the actions, means and sigmas.
        """

        # Assignments and Initializations
        self.actions = actions
        step = int(actions.shape[0] / len(bids))
        self.means_C1 = []
        self.means_C2 = []
        self.means_C3 = []
        for i in range(1, len(bids) + 1):
            self.means_C1[step * (i - 1):step * i] = UserC1.demand_curve(UserC1.prices) * np.repeat(UserC1.click_vs_bid(bids[i - 1]), step)
            self.means_C2[step * (i - 1):step * i] = UserC2.demand_curve(UserC2.prices) * np.repeat(UserC2.click_vs_bid(bids[i - 1]), step)
            self.means_C3[step * (i - 1):step * i] = UserC3.demand_curve(UserC3.prices) * np.repeat(UserC3.click_vs_bid(bids[i - 1]), step)
        self.sigmas = np.ones(len(actions)) * sigma

    def assess_user_type(self,user):
        """
        Method used to assess the type of user and return the right means depending on the relative demand curve
        """
        if user.f1_value:
            if user.f2_value:
                return "C1"
            else:
                return "C2"
        else:
            return "C3"


    def round(self, pulled_arm,user):
        '''Simulate the current round of pricing and bidding with the given pulled arm. Returns the realization
        of a random normal with set mean and std.'''
        if(self.assess_user_type(user)=="C1"):
            reward = np.random.normal(self.means_C1[pulled_arm], self.sigmas[pulled_arm])
        if(self.assess_user_type(user)=="C2"):
            reward = np.random.normal(self.means_C2[pulled_arm], self.sigmas[pulled_arm])
        else:
            reward = np.random.normal(self.means_C3[pulled_arm], self.sigmas[pulled_arm])

        return reward