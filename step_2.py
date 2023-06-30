# %%
import numpy as np
import matplotlib.pyplot as plt

from Environments.Users import  UserC1, UserC2, UserC3
from Environments.Bidding_environment import BiddingEnvironment
from Learners.GPTS_Learner import GPTS_Learner
from Learners.GPUCB1_Learner import GPUCB1_Learner

#%load_ext autoreload
#%autoreload 2

# %% Parameters
n_arms = 100
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10

T = 365
n_experiments = 4
gpts_rewards_per_experiment = []
gpucb1_rewards_per_experiment = []

# %% Run the experiments
for e in range(0, n_experiments):
    env = BiddingEnvironment(bids=bids, sigma = sigma, user=UserC1())
    gpts_learner = GPTS_Learner(n_arms = n_arms, bids = bids)
    gpucb1_learner = GPUCB1_Learner(n_arms = n_arms, bids = bids, M = max_bid)
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

# %% Compute the regret
# opt = np.max(env.means)
# opt = np.max([np.max(gpts_rewards_per_experiment,axis=1), np.max(gpucb1_rewards_per_experiment, axis=1)])
opt = np.array(120)
avg_regret_gpts = np.mean(opt - gpts_rewards_per_experiment, axis=0)
cum_regret_gpts = np.cumsum(avg_regret_gpts)
std_regret_gpts = np.std(opt - gpts_rewards_per_experiment, axis=0)
avg_regret_gpucb1 = np.mean(opt - gpucb1_rewards_per_experiment, axis=0)
cum_regret_gpucb1 = np.cumsum(avg_regret_gpucb1)
std_regret_gpucb1 = np.std(opt - gpucb1_rewards_per_experiment, axis=0)

# %% Plot the cumulative regret
fig = plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(cum_regret_gpucb1, 'b')
plt.plot(cum_regret_gpts, 'r')
plt.fill_between(range(len(cum_regret_gpucb1)), cum_regret_gpucb1 - np.cumsum(std_regret_gpucb1), cum_regret_gpucb1 + np.cumsum(std_regret_gpucb1), alpha=0.2, color='b')
plt.fill_between(range(len(cum_regret_gpts)), cum_regret_gpts - np.cumsum(std_regret_gpts), cum_regret_gpts + np.cumsum(std_regret_gpts), alpha=0.2, color='r')
plt.legend(["GP-UCB1", "GPTS"])
fig = plt.gcf()
plt.show()

fig.savefig("results/S2_cumulative_regret.png")

# %% Plot the instantaneous regret
fig = plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_gpucb1, 'b')
plt.plot(avg_regret_gpts, 'r')
plt.fill_between(range(len(avg_regret_gpucb1)), avg_regret_gpucb1 - std_regret_gpucb1, avg_regret_gpucb1 + std_regret_gpucb1, alpha=0.2, color='b')
plt.fill_between(range(len(avg_regret_gpts)), avg_regret_gpts - std_regret_gpts, avg_regret_gpts + std_regret_gpts, alpha=0.2, color='r')
plt.legend(["GP-UCB1", "GPTS"])
fig = plt.gcf()
plt.show()

fig.savefig("results/S2_instantaneous_regret.png")

# %% Plot the cumulative reward
plt.figure(1)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.cumsum(np.mean(gpucb1_rewards_per_experiment, axis=0)), 'b')
plt.plot(np.cumsum(np.mean(gpts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["Greedy", "UCB1", "TS"])
plt.title("Cumulative Reward")
plt.show()

# %% Plot the instantaneous reward
plt.figure(3)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(gpucb1_rewards_per_experiment, axis=0), 'b')
plt.plot(np.mean(gpts_rewards_per_experiment, axis=0), 'r')
plt.hlines(y=opt, xmin=0, xmax=T, colors='k', linestyles='dashed')
plt.legend(["Greedy", "UCB1", "TS", "Clairvoyant"])
plt.title("Instantaneous Reward")
plt.show()

# %% Plot the instantaneous regret with standard deviation
plt.figure(4)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_gpucb1, 'b')
plt.plot(avg_regret_gpts, 'r')
plt.hlines(y=0, xmin=0, xmax=T, colors='k', linestyles='dashed')
plt.fill_between(range(len(avg_regret_gpucb1)), avg_regret_gpucb1 - std_regret_gpucb1, avg_regret_gpucb1 + std_regret_gpucb1, color='b', alpha=0.2)
plt.fill_between(range(len(avg_regret_gpts)), avg_regret_gpts - std_regret_gpts, avg_regret_gpts + std_regret_gpts, color='r', alpha=0.2)
plt.legend(["Greedy", "UCB1", "TS", "Clairvoyant"])
plt.title("Instantaneous Regret with Standard Deviation")
plt.show()


# %% Plot of cumulative regret with variance
avg_cum_regret_ucb1 = np.cumsum(avg_regret_gpucb1)
avg_cum_regret_ts = np.cumsum(avg_regret_gpts)

std_cum_regret_ucb1 = np.cumsum(std_regret_gpucb1)
std_cum_regret_ts = np.cumsum(std_regret_gpts)

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_cum_regret_ucb1, 'b')
plt.plot(avg_cum_regret_ts, 'r')
plt.fill_between(range(len(avg_cum_regret_ucb1)), avg_cum_regret_ucb1 - std_cum_regret_ucb1, avg_cum_regret_ucb1 + std_cum_regret_ucb1, alpha=0.2, color='b')
plt.fill_between(range(len(avg_cum_regret_ts)), avg_cum_regret_ts - std_cum_regret_ts, avg_cum_regret_ts + std_cum_regret_ts, alpha=0.2, color='r')
plt.legend(["Greedy", "UCB1", "TS"])
plt.title("Cumulative Regret with standard deviation")
plt.show()