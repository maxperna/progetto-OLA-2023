import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Learners.Learner import Learner


class GPTS_Contextual(Learner):
    '''Gaussian Process Thompson Sampling Learner inheriting from the Learner class.'''

    def __init__(self, n_arms, bids,context):
        '''Initialize the Gaussian Process Thompson Sampling Learner with a number of arms, the arms and a kernel.'''

        super().__init__(n_arms) # supercharges the init from the learner

        # Assignments and Initializations
        self.arms = bids
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10.0
        self.pulled_arms = []
        self.context = context
        self.features = []
        self.rewards_per_context = []
        self.pulled_arms_context = []
        alpha = 0.5  # alpha

        # The kernel is set as the product of a constant and a Radial-basis with values 1 and range 1e-3 to 1e3
        theta = 1.0
        l = 1.0
        kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
        n_restarts = 9

        # Sets the Gaussian Process Regressor from the given kernel
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True,
                                        n_restarts_optimizer=n_restarts)

        self.update_context(self.context)

    def update_observations(self, pulled_arm, reward,features):


        '''Updates the information on the rewards keeping track of the pulled arm (supercharges update_observations in Learner).'''
        super().update_observations(pulled_arm, reward)
        index = self.get_context_section(features)
        self.rewards_per_context[index].append(reward)
        self.pulled_arms_context[index].append(self.arms[pulled_arm])

        self.features.append(features)
        self.pulled_arms.append(self.arms[pulled_arm])
        # Keeps track of the pulled arm
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self,features):
        '''Updates the model with the new means and sigmas.'''
        # Sets the trimmed pulled arms vs rewards
        #x = np.atleast_2d(self.pulled_arms).T
        #y = self.collected_rewards

        index = self.get_context_section(features)
        # x = np.atleast_2d(self.pulled_arms).T
        x = np.atleast_2d(self.pulled_arms_context[index])
        # x=np.array(self.pulled_arms)
        y = self.rewards_per_context[index]

        # Fits the Gaussian process
        self.gp.fit(x, y)

        # Evaluates current means and sigmas with a lower bound on the standard deviation of 0.01 (for convergence)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms), return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward,features):
        '''Proceeds of 1 time step updating both the observations and the model.'''
        self.t += 1
        self.update_observations(pulled_arm, reward,features)
        self.update_model(features)

    def pull_arm(self):
        '''Pulls the arm from the current multidimensional random normal distribution, returning the index of the best arm.'''

        sampled_values = np.random.normal(self.means, self.sigmas) # pulls some random arms basing on current means and sigmas
        return np.argmax(sampled_values) # returns the index of the best arm

    def update_context(self, context):
        self.context = context
        self.rewards_per_context = []
        self.pulled_arms_context = []

        contexts_number = len(context)
        if self.context is None:
            contexts_number = 1
        for i in range(contexts_number):
            self.rewards_per_context.append([])
            self.pulled_arms_context.append([])
            self.rewards_per_context.append([])

    def get_context_section(self, features):
        """
        Method used to return the index for a right update of the context
        """
        if self.context is None:
            return 0
        else:
            for split in self.context.keys():
                for customer in self.context[split]:
                    if customer.get_features == features:
                        return list(self.context.keys()).index(split)
        return -1