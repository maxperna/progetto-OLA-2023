# %%
#ENVIRONMENT WITH KNOWN USERS CONTEXT
import numpy as np
import matplotlib.pyplot as plt

import logging

from Environments.Context_environment import ContextEnvironment
from Environments.Users import UserC1,UserC2,UserC3
from Learners.GPTS_Learner_s3 import GPTS_Learner
from Learners.GPUCB1_Learner_s3 import GPUCB1_Learner
# %% Parameters
n_bids = 100
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_bids)

Collector_1 = UserC1()
Collector_2 = UserC2()
Collector_3 = UserC3()

user_set = (Collector_1,Collector_2,Collector_3)

prices = Collector_1.prices

sigma = 10

# Generate an action space with both bids and prices
action_space = np.array([[bid,price] for bid in bids for price in prices]) # all combinations of bid and price

n_arms = action_space.shape[0]

T = 365

n_experiments = 4
gpts_rewards_per_experiment_c1 = []
gpucb1_rewards_per_experiment_c1 = []

gpts_rewards_per_experiment_c2 = []
gpucb1_rewards_per_experiment_c2 = []

gpts_rewards_per_experiment_c3 = []
gpucb1_rewards_per_experiment_c3 = []
#bids_made_per_experiment = [] #the bids made by the learner

number_of_c1 = 0    #number of C1 users
number_of_c2 = 0    #number of C2 users
number_of_c3 = 0    #number of C3 users

# %% Run the experiments
for e in range(0, n_experiments):
    env = ContextEnvironment(actions=action_space, bids=bids, sigma = sigma,user_set=user_set)
    #User c1 learner
    gpts_learner_c1 = GPTS_Learner(n_arms = n_arms, bids = action_space)
    gpucb1_learner_c1 = GPUCB1_Learner(n_arms = n_arms, bids = action_space, M = np.max(action_space[:,0]*action_space[:,1]))
    #User c2 learner
    gpts_learner_c2 = GPTS_Learner(n_arms = n_arms, bids = action_space)
    gpucb1_learner_c2 = GPUCB1_Learner(n_arms = n_arms, bids = action_space, M = np.max(action_space[:,0]*action_space[:,1]))
    #User c3 learner
    gpts_learner_c3 = GPTS_Learner(n_arms = n_arms, bids = action_space)
    gpucb1_learner_c3 = GPUCB1_Learner(n_arms = n_arms, bids = action_space, M = np.max(action_space[:,0]*action_space[:,1]))


    for t in range(0, T):
        pick = np.random.randint(1,4)

        if pick==1:
            number_of_c1 += 1
            # GP UCB1 Learner
            pulled_arm = gpucb1_learner_c1.pull_arm()
            reward = env.round(pulled_arm,Collector_1)
            gpucb1_learner_c1.update(pulled_arm, reward)

            # GP Thompson Sampling Learner
            pulled_arm = gpts_learner_c2.pull_arm()
            reward = env.round(pulled_arm,Collector_1)
            gpts_learner_c1.update(pulled_arm, reward)

        elif pick==2:
            number_of_c2 += 1
            # GP UCB1 Learner
            pulled_arm = gpucb1_learner_c2.pull_arm()
            reward = env.round(pulled_arm,Collector_2)
            gpucb1_learner_c2.update(pulled_arm, reward)

            # GP Thompson Sampling Learner
            pulled_arm = gpts_learner_c2.pull_arm()
            reward = env.round(pulled_arm,Collector_2)
            gpts_learner_c2.update(pulled_arm, reward)
        else:
            number_of_c3 += 1
            # GP UCB1 Learner
            pulled_arm = gpucb1_learner_c3.pull_arm()
            reward = env.round(pulled_arm,Collector_3)
            gpucb1_learner_c3.update(pulled_arm, reward)

            # GP Thompson Sampling Learner
            pulled_arm = gpts_learner_c3.pull_arm()
            reward = env.round(pulled_arm,Collector_3)
            gpts_learner_c3.update(pulled_arm, reward)

    gpts_rewards_per_experiment_c1.append(gpts_learner_c1.collected_rewards)
    gpucb1_rewards_per_experiment_c1.append(gpucb1_learner_c1.collected_rewards)

    gpts_rewards_per_experiment_c2.append(gpts_learner_c2.collected_rewards)
    gpucb1_rewards_per_experiment_c2.append(gpucb1_learner_c2.collected_rewards)

    gpts_rewards_per_experiment_c3.append(gpts_learner_c3.collected_rewards)
    gpucb1_rewards_per_experiment_c3.append(gpucb1_learner_c3.collected_rewards)

# %% Compute the probability of choosing each user for disaggregated model
tot_users = number_of_c1 + number_of_c2 + number_of_c3
#Probability of have each user
prob_c1 = number_of_c1/tot_users
prob_c2 = number_of_c2/tot_users
prob_c3 = number_of_c3/tot_users

print(f"Experiments with f{number_of_c1} C1 users, {number_of_c2} C2 users and {number_of_c3} C3 users")
print(f"Probability C1 {prob_c1}, probability C2 {prob_c2}, probability C3 {prob_c3}")

#Padding to make all the array of the same size
resize_array = lambda arr: np.pad(arr, (0, max(0, T - len(arr))), mode='constant')

gpts_rewards_per_experiment_c1 = [resize_array(arr) for arr in gpts_rewards_per_experiment_c1]
gpucb1_rewards_per_experiment_c1 = [resize_array(arr) for arr in gpucb1_rewards_per_experiment_c1]

gpts_rewards_per_experiment_c2 = [resize_array(arr) for arr in gpts_rewards_per_experiment_c2]
gpucb1_rewards_per_experiment_c2 = [resize_array(arr) for arr in gpucb1_rewards_per_experiment_c2]

gpts_rewards_per_experiment_c3 = [resize_array(arr) for arr in gpts_rewards_per_experiment_c3]
gpucb1_rewards_per_experiment_c3 = [resize_array(arr) for arr in gpucb1_rewards_per_experiment_c3]

#Aggregate the three demand curves
gpts_rewards_per_experiment = np.array(gpts_rewards_per_experiment_c1)*prob_c1 + np.array(gpts_rewards_per_experiment_c2)*prob_c2 + np.array(gpts_rewards_per_experiment_c3)*prob_c3
gpucb1_rewards_per_experiment = np.array(gpucb1_rewards_per_experiment_c1)*prob_c1 + np.array(gpucb1_rewards_per_experiment_c2)*prob_c2 + np.array(gpucb1_rewards_per_experiment_c3)*prob_c3

# %% Compute the regret
opt = np.max([max(lst) for lst in [env.means_C1,env.means_C2,env.means_C3]])
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