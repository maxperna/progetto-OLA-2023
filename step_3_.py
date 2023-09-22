# %%
import numpy as np
import matplotlib.pyplot as plt

from Environments.Environment_S3 import PricingBiddingEnvironment
from Environments.Users import UserC1
from Learners.GPTS_Learner_s3 import GPTS_Learner
from Learners.GPUCB1_Learner_s3 import GPUCB1_Learner

from param import n_arms_bidding, min_bid, max_bid, T, n_experiments_S3, production_cost, std_noise_general

# %% Parameters
bids = np.linspace(min_bid, max_bid, n_arms_bidding)

Collector = UserC1()
prices = Collector.prices

best_price = Collector.clairvoyant()[0]
conversion_rate = Collector.probabilities[np.where(Collector.prices == best_price)][0]

margin = best_price - production_cost

# Generate an action space with both bids and prices
action_space = np.array([[bid,price] for bid in bids for price in prices]) # all combinations of bid and price
n_arms = action_space.shape[0]

gpts_rewards_per_experiment = []
gpucb1_rewards_per_experiment = []

# %% Run the experiments
for e in range(0, n_experiments_S3):
    env = PricingBiddingEnvironment(actions=action_space, sigma = std_noise_general, bids=bids, user=UserC1(), production_cost=production_cost)
    gpts_learner = GPTS_Learner(n_arms = n_arms, bids = action_space)
    gpucb1_learner = GPUCB1_Learner(n_arms = n_arms, bids = action_space, M = Collector.clairvoyant()[2])
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
opt = np.max(env.means)
regret_gpucb1 = opt - gpucb1_rewards_per_experiment  # row = exp, col = t
avg_regret_gpucb1 = np.mean(regret_gpucb1, axis=0)
std_regret_gpucb1 = np.std(regret_gpucb1, axis=0)
cum_avg_regret_gpucb1 = np.mean(np.cumsum(regret_gpucb1, axis=1), axis=0)
cum_std_regret_gpucb1 = np.std(np.cumsum(regret_gpucb1, axis=1), axis=0)

regret_gpts = opt - gpts_rewards_per_experiment  # row = exp, col = t
avg_regret_gpts = np.mean(regret_gpts, axis=0)
std_regret_gpts = np.std(regret_gpts, axis=0)
cum_avg_regret_gpts = np.mean(np.cumsum(regret_gpts, axis=1), axis=0)
cum_std_regret_gpts = np.std(np.cumsum(regret_gpts, axis=1), axis=0)

# %% Plot the cumulative regret
fig = plt.figure(0, facecolor='white')
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(cum_avg_regret_gpucb1, 'b')
plt.plot(cum_avg_regret_gpts, 'r')
plt.fill_between(range(len(cum_avg_regret_gpucb1)), cum_avg_regret_gpucb1 - cum_std_regret_gpucb1, cum_avg_regret_gpucb1 + cum_std_regret_gpucb1, alpha=0.2, color='b')
plt.fill_between(range(len(cum_avg_regret_gpts)), cum_avg_regret_gpts - cum_std_regret_gpts, cum_avg_regret_gpts + cum_std_regret_gpts, alpha=0.2, color='r')
plt.legend(["GP-UCB1", "GP-TS"])
plt.title("Cumulative Regret")
fig = plt.gcf()
plt.show()

fig.savefig("results/S3_cumulative_regret.png")

# %% Plot the instantaneous regret
fig = plt.figure(1,facecolor='white')
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_gpucb1, 'b')
plt.plot(avg_regret_gpts, 'r')
plt.fill_between(range(len(avg_regret_gpucb1)), avg_regret_gpucb1 - std_regret_gpucb1, avg_regret_gpucb1 + std_regret_gpucb1, alpha=0.2, color='b')
plt.fill_between(range(len(avg_regret_gpts)), avg_regret_gpts - std_regret_gpts, avg_regret_gpts + std_regret_gpts, alpha=0.2, color='r')
plt.hlines(0, 0, T, colors='black', linestyles='dashed')
plt.legend(["GP-UCB1", "GP-TS"])
plt.title("Instantaneous Regret")
fig = plt.gcf()
plt.show()

fig.savefig("results/S3_instantaneous_regret.png")

# %% Compute the reward
avg_reward_gpucb1 = np.mean(gpucb1_rewards_per_experiment, axis=0)
std_reward_gpucb1 = np.std(gpucb1_rewards_per_experiment, axis=0)
cum_avg_reward_gpucb1 = np.mean(np.cumsum(gpucb1_rewards_per_experiment, axis=1), axis=0)
cum_std_reward_gpucb1 = np.std(np.cumsum(gpucb1_rewards_per_experiment, axis=1), axis=0)

avg_reward_gpts = np.mean(gpts_rewards_per_experiment, axis=0)
std_reward_gpts = np.std(gpts_rewards_per_experiment, axis=0)
cum_avg_reward_gpts = np.mean(np.cumsum(gpts_rewards_per_experiment, axis=1), axis=0)
cum_std_rreward_gpts = np.std(np.cumsum(gpts_rewards_per_experiment, axis=1), axis=0)

# %% Plot the cumulative reward
plt.figure(2, facecolor='white')
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(cum_avg_reward_gpucb1, 'b')
plt.plot(cum_avg_reward_gpts, 'r')
plt.fill_between(range(len(cum_avg_reward_gpucb1)), cum_avg_reward_gpucb1 - cum_std_reward_gpucb1, cum_avg_reward_gpucb1 + cum_std_reward_gpucb1, alpha=0.2, color='b')
plt.fill_between(range(len(cum_avg_reward_gpts)), cum_avg_reward_gpts - cum_std_rreward_gpts, cum_avg_reward_gpts + cum_std_rreward_gpts, alpha=0.2, color='r')
plt.legend(["GP-UCB1", "GP-TS"])
plt.title("Cumulative Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S3_cumulative_reward.png")

# %% Plot the instantaneous reward
plt.figure(3, facecolor='white')
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(avg_reward_gpucb1, 'b')
plt.plot(avg_reward_gpts, 'r')
plt.hlines(y=opt, xmin=0, xmax=T, colors='k', linestyles='dashed')
plt.fill_between(range(len(avg_reward_gpucb1)), avg_reward_gpucb1 - std_reward_gpucb1, avg_reward_gpucb1 + std_reward_gpucb1, alpha=0.2, color='b')
plt.fill_between(range(len(avg_reward_gpts)), avg_reward_gpts - std_reward_gpts, avg_reward_gpts + std_reward_gpts, alpha=0.2, color='r')
plt.legend(["GP-UCB1", "GP-TS",  "Clairvoyant"])
plt.title("Instantaneous Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S3_instantaneous_reward.png")
# %%
