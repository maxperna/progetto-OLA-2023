import numpy as np
import matplotlib.pyplot as plt
from Environments.Environment import Environment
from Environments.Bidding_environment import BiddingEnvironment
from Environments.Users import UserC1
from Learners.GPTS_Learner import GPTS_Learner
from Learners.GPUCB1_Learner import GPUCB1_Learner
from Learners.TS_Learner import TS_Learner
from Learners.UCB1_Learner import UCB1_Learner

#%% Parameters
n_bids = 100
min_bid = 0.0
max_bid = 1.0
Collector = UserC1()

bids = np.linspace(min_bid, max_bid, n_bids)
prices = Collector.prices
sigma = 10

#Generate an action space with both the bids and the price
action_space = np.array([[bid,price] for bid in bids for price in prices])
n_arms = action_space.shape[0]

T = 365

n_experiments = 1000
gpts_rewards_per_experiment = []
gpucb1_rewards_per_experiment = []
bids_made_per_experiment = []   #the bids made by the learner

#Contextual multi-armed bandit to be implemented
for e in range(0, n_experiments):
    env = BiddingEnvironment(bids=bids, sigma = sigma, user=UserC1())
    gpts_learner = GPTS_Learner(n_arms = n_arms, bids = action_space)
    gpucb1_learner = GPUCB1_Learner(n_arms = n_arms, bids = action_space, M = max_bid)
    for t in range(0, T):
        # GP UCB1 Learner
        pulled_arm = gpucb1_learner.pull_arm()
        reward = env.round(pulled_arm)
        gpucb1_learner.update(pulled_arm, reward)

        # GP Thompson Sampling Learner
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gpts_learner.update(pulled_arm, reward)

    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    gpucb1_rewards_per_experiment.append(gpucb1_learner.collected_rewards)