import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Learners.Learner import Learner

class GPUCB1_Contextual(Learner):
    """
    Contextual GPUCB implementation
    """
    def __init__(self, n_arms, bids, M,context):
        """
        Context = None if no context is given
        """

        super().__init__(n_arms)
        self.arms = bids
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10.0
        self.pulled_arms = []
        self.M = M
        self.features = []
        self.context = context     #contex as set of customers
        self.rewards_per_context = []
        self.pulled_arms_context = []

        alpha = .5
        theta = 1.0
        l = 1.0
        kernel = C(theta, constant_value_bounds="fixed") * RBF(l, length_scale_bounds="fixed")
        n_restarts = 9

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha ** 2,
            normalize_y=True,
            n_restarts_optimizer=n_restarts
        )

        self.update_context(self.context)

    def update_observations(self, pulled_arm, reward, features):
        """
        Update internal state given last action and its outcome
        """
        super().update_observations(pulled_arm, reward)

        index = self.get_context_section(features)
        self.rewards_per_context[index].append(reward)
        self.pulled_arms_context[index].append(self.arms[pulled_arm])

        self.features.append(features)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self,features):
        """
        This method update internal model retraining the GP and predicting again the values for every arm
        """
        # Prepare X, y for GP
        #Get the right slice of context
        index = self.get_context_section(features)
        #x = np.atleast_2d(self.pulled_arms).T
        x = np.atleast_2d(self.pulled_arms_context[index])
        #x=np.array(self.pulled_arms)
        y = self.rewards_per_context[index]

        # Retrain the GP
        self.gp.fit(x, y)

        # Retrieve predictions from GP
        #self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms), return_std=True)
        #self.means, self.sigmas = self.gp.predict(np.array(self.arms), return_std=True)

        # sigma lower bound
        self.sigmas = np.maximum(self.sigmas, 1e-2)

        return

    def update(self, pulled_arm, reward, features):
        """
        This method update the GPTS state and internal model
        """
        self.t += 1
        self.update_observations(pulled_arm, reward,features)
        self.update_model(features)

        return

    def pull_arm(self):

        if self.t < self.n_arms:
            return self.t # TODO: check this
        else:
            beta = np.sqrt(2*np.log2(self.n_arms*(self.t+1)*(self.t+1)*np.pi*np.pi/(6*0.05))) # TODO WTF is this
            upper_bounds = self.means + beta * self.sigmas

        return np.argmax(upper_bounds)

    def update_context(self,context):
        self.context = context
        self.rewards_per_context = []
        self.pulled_arms_context = []

        if self.context is None:
            contexts_number = 1
        else:
            contexts_number = len(context)
        for i in range(contexts_number):
            self.rewards_per_context.append([])
            self.pulled_arms_context.append([])
            self.rewards_per_context.append([])

    def get_context_section(self,features):
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
