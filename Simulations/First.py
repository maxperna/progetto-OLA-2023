import numpy as np
import matplotlib.pyplot as plt
from Environments.Environment import *
from Learners.TS_Learner import *


"""
First simulation environment where test the settings
"""
env = Environment()
ts_rewards_per_experiment = []
# Optimal price is the number 3
optimal_price = env.user.probabilities[2]

for t in range(env.n_rounds):
    ts_learner = TSLearner(n_arms=env.n_arms)
    for t in range(env.avg_n_users):
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm,reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(optimal_price-ts_rewards_per_experiment,axis=0)), 'r')
plt.show()