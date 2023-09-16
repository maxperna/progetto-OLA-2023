# %%
import numpy as np
import matplotlib.pyplot as plt

from Environments.Context_environment import ContextEnvironment
from Algorithm.Splitting_Context import Context
from Environments.Users import UserC1,UserC2,UserC3
from Learners.GPTS_Learner_s3 import GPTS_Learner
from Learners.GPUCB1_Learner_s3 import GPUCB1_Learner

from Algorithm import Splitting_Context

# %% Parameters
n_bids = 100
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_bids)

Collector_1 = UserC1()
Collector_2 = UserC2()
Collector_3 = UserC3()

user_set = (Collector_1,Collector_2,Collector_3)
user_collected =[]  #list of users that have been collected

prices = Collector_1.prices
sigma = 10

production_cost = 75

# Generate an action space with both bids and prices
action_space = np.array([[bid,price] for bid in bids for price in prices]) # all combinations of bid and price
n_arms = action_space.shape[0]

T = 365

n_experiments = 4
gpts_rewards_per_experiment = []
gpucb1_rewards_per_experiment = []
#bids_made_per_experiment = [] #the bids made by the learner

# %% Run the experiments
for e in range(0, n_experiments):
    env = ContextEnvironment(actions=action_space, bids=bids, sigma = sigma,user_set=user_set)
    gpts_learner = GPTS_Learner(n_arms = n_arms, bids = action_space)
    gpucb1_learner = GPUCB1_Learner(n_arms = n_arms, bids = action_space, M = np.max(action_space[:,0]*action_space[:,1]))
    for t in range(0, T):
        #Pick users
        pick = np.random.randint(1, 4)
        user_collected.append(pick)
        if pick == 1:
            user = Collector_1
        elif pick == 2:
            user = Collector_2
        else:
            user = Collector_3


        # GP UCB1 Learner
        pulled_arm = gpucb1_learner.pull_arm()
        reward = env.round(pulled_arm,user)
        gpucb1_learner.update(pulled_arm, reward)

        # GP Thompson Sampling Learner
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(pulled_arm,user)
        gpts_learner.update(pulled_arm, reward)

        if(T%14==0):
            #DO SPLITTING
            pass

    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    gpucb1_rewards_per_experiment.append(gpucb1_learner.collected_rewards)

# %% Compute the regret
opt = np.max(env.means)
avg_regret_gpts = np.mean(opt - gpts_rewards_per_experiment, axis=0)
cum_regret_gpts = np.cumsum(avg_regret_gpts)
std_regret_gpts = np.std(opt - gpts_rewards_per_experiment, axis=0)
avg_regret_gpucb1 = np.mean(opt - gpucb1_rewards_per_experiment, axis=0)
cum_regret_gpucb1 = np.cumsum(avg_regret_gpucb1)
std_regret_gpucb1 = np.std(opt - gpucb1_rewards_per_experiment, axis=0)


# %%
print(opt)

print(np.shape(gpucb1_rewards_per_experiment))
print(gpucb1_rewards_per_experiment[0])

aux = opt - gpucb1_rewards_per_experiment
print(np.shape(aux))

print(aux)
# %% Plot the cumulative regret
fig = plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(cum_regret_gpucb1, 'b')
plt.plot(cum_regret_gpts, 'r')
plt.fill_between(range(len(cum_regret_gpucb1)), cum_regret_gpucb1 - np.cumsum(std_regret_gpucb1), cum_regret_gpucb1 + np.cumsum(std_regret_gpucb1), alpha=0.2, color='b')
plt.fill_between(range(len(cum_regret_gpts)), cum_regret_gpts - np.cumsum(std_regret_gpts), cum_regret_gpts + np.cumsum(std_regret_gpts), alpha=0.2, color='r')
plt.legend(["GP-UCB1", "GP-TS"])
fig = plt.gcf()
plt.show()

fig.savefig("results/S3_cumulative_regret.png")

# %% Plot the instantaneous regret
fig = plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_gpucb1, 'b')
plt.plot(avg_regret_gpts, 'r')
plt.fill_between(range(len(avg_regret_gpucb1)), avg_regret_gpucb1 - std_regret_gpucb1, avg_regret_gpucb1 + std_regret_gpucb1, alpha=0.2, color='b')
plt.fill_between(range(len(avg_regret_gpts)), avg_regret_gpts - std_regret_gpts, avg_regret_gpts + std_regret_gpts, alpha=0.2, color='r')
plt.hlines(0, 0, T, colors='black', linestyles='dashed')
plt.legend(["GP-UCB1", "GP-TS"])
fig = plt.gcf()
plt.show()

fig.savefig("results/S3_instantaneous_regret.png")

# %% Plot the cumulative reward
plt.figure(1)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.cumsum(np.mean(gpucb1_rewards_per_experiment, axis=0)), 'b')
plt.plot(np.cumsum(np.mean(gpts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["GP-UCB1", "GP-TS"])
plt.title("Cumulative Reward")
plt.show()

# %% Plot the instantaneous reward
plt.figure(3)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(gpucb1_rewards_per_experiment, axis=0), 'b')
plt.plot(np.mean(gpts_rewards_per_experiment, axis=0), 'r')
plt.hlines(y=opt, xmin=0, xmax=T, colors='k', linestyles='dashed')
plt.legend(["GP-UCB1", "GP-TS", "Clairvoyant"])
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
plt.legend(["GP-UCB1", "GP-TS", "Clairvoyant"])
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
plt.legend(["GP-UCB1", "GP-TS"])
plt.title("Cumulative Regret with standard deviation")
plt.show()

# %%
