import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Learners.Learner import Learner


class GPTS_Learner(Learner):
    '''Gaussian Process Thompson Sampling Learner inheriting from the Learner class.'''

    def __init__(self, n_arms, bids):
        '''Initialize the Gaussian Process Thompson Sampling Learner with a number of arms, the arms and a kernel.'''

        super().__init__(n_arms) # supercharges the init from the learner

        # Assignments and Initializations
        self.arms = bids
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10.0
        self.pulled_arms = []
        alpha = 0.5  # alpha
        theta = 1.0
        l = 1.0
        # kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
        kernel = C(theta, constant_value_bounds="fixed") * RBF(l, length_scale_bounds="fixed") # This works way better
        n_restarts = 9

        # Sets the Gaussian Process Regressor from the given kernel
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True,
                                        n_restarts_optimizer=n_restarts)

    def update_observations(self, pulled_arm, reward):
        '''Updates the information on the rewards keeping track of the pulled arm (supercharges update_observations in Learner).'''
        super().update_observations(pulled_arm, reward) 
        # Keeps track of the pulled arm
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        '''Updates the model with the new means and sigmas.'''
        # Sets the trimmed pulled arms vs rewards
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards

        # Fits the Gaussian process
        self.gp.fit(x, y)

        # Evaluates current means and sigmas with a lower bound on the standard deviation of 0.01 (for convergence)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        '''Proceeds of 1 time step updating both the observations and the model.'''
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        '''Pulls the arm from the current multidimensional random normal distribution, returning the index of the best arm.'''

        sampled_values = np.random.normal(self.means, self.sigmas) # pulls some random arms basing on current means and sigmas
        return np.argmax(sampled_values) # returns the index of the best arm
