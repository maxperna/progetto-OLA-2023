import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Learners.Learner import Learner

from param import sigma_gp

class GPUCB1_Learner(Learner):
    """
    GPUCB implementation
    """

    def __init__(self, n_arms, bids, M):
        super().__init__(n_arms)
        self.arms = bids
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * sigma_gp
        self.pulled_arms = []
        self.M = M
        alpha = .5
        theta = 1.0
        l = 1.0

        # kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
        kernel = C(theta, constant_value_bounds="fixed") * RBF(l, length_scale_bounds="fixed")
        n_restarts = 9

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha ** 2,
            normalize_y=True,
            n_restarts_optimizer=n_restarts
        )

    def update_observations(self, pulled_arm, reward):
        """
        Update internal state given last action and its outcome
        """
        super().update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        """
        This method update internal model retraining the GP and predicting again the values for every arm
        """
        x = np.atleast_2d(self.pulled_arms)
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms), return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

        return

    def update(self, pulled_arm, reward):
        """
        This method update the GPTS state and internal model
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

        return

    def pull_arm(self):

        beta = np.sqrt(2*np.log2(self.n_arms*(self.t+1)*(self.t+1)*np.pi*np.pi/(6*0.05))) 
        upper_bounds = self.means + beta * self.sigmas

        return np.argmax(upper_bounds)
