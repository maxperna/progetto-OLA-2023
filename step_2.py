# %%
import numpy as np
import matplotlib.pyplot as plt

from Environments.Users import  UserC1, UserC2, UserC3
from Environments.Bidding_environment import BiddingEnvironment
from Learners.GPTS_Learner import GPTS_Learner
from Learners.GPUCB1_Learner import GPUCB1_Learner

#%load_ext autoreload
#%autoreload 2

# TODO use scipy std since consider element as sample from population
# TODO refactor reward 

# %% Parameters
n_arms = 100  #number of possible bids
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10  # Noise of the environment TODO consider using %

T = 60
n_experiments = 20
gpts_rewards_per_experiment = []
gpucb1_rewards_per_experiment = []

Collector = UserC1()

#selected_price = np.argmax(Collector.demand_curve())
selected_price = 300
#conversion_rate = np.max(Collector.demand_curve())
conversion_rate = Collector.demand_curve(selected_price)
production_cost = 75
margin = selected_price - production_cost

# %% Run the experiments
for e in range(0, n_experiments):
    env = BiddingEnvironment(bids=bids, sigma = sigma, margin = margin, rate = conversion_rate, user=UserC1())
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
fig = plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(cum_avg_regret_gpucb1, 'b')
plt.plot(cum_avg_regret_gpts, 'r')
plt.fill_between(range(len(cum_avg_regret_gpucb1)), cum_avg_regret_gpucb1 - cum_std_regret_gpucb1, cum_avg_regret_gpucb1 + cum_std_regret_gpucb1, alpha=0.2, color='b')
plt.fill_between(range(len(cum_avg_regret_gpts)), cum_avg_regret_gpts - cum_std_regret_gpts, cum_avg_regret_gpts + cum_std_regret_gpts, alpha=0.2, color='r')
plt.legend(["GP-UCB1", "GPTS"])
plt.title("Cumulative Regret")
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
plt.hlines(0, 0, T, colors='black', linestyles='dashed')
plt.legend(["GP-UCB1", "GPTS"])
plt.title("Instantaneous Regret")
fig = plt.gcf()
plt.show()

fig.savefig("results/S2_instantaneous_regret.png")

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
plt.figure(2)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(cum_avg_reward_gpucb1, 'b')
plt.plot(cum_avg_reward_gpts, 'r')
plt.fill_between(range(len(cum_avg_reward_gpucb1)), cum_avg_reward_gpucb1 - cum_std_reward_gpucb1, cum_avg_reward_gpucb1 + cum_std_reward_gpucb1, alpha=0.2, color='r')
plt.fill_between(range(len(cum_avg_reward_gpts)), cum_avg_reward_gpts - cum_std_rreward_gpts, cum_avg_reward_gpts + cum_std_rreward_gpts, alpha=0.2, color='b')
plt.legend(["GP-UCB1", "GPTS"])
plt.title("Cumulative Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S2_cumulative_reward.png")

# %% Plot the instantaneous reward
plt.figure(3)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(avg_reward_gpucb1, 'b')
plt.plot(avg_reward_gpts, 'r')
plt.hlines(y=opt, xmin=0, xmax=T, colors='k', linestyles='dashed')
plt.fill_between(range(len(avg_reward_gpucb1)), avg_reward_gpucb1 - std_reward_gpucb1, avg_reward_gpucb1 + std_reward_gpucb1, alpha=0.2, color='b')
plt.fill_between(range(len(avg_reward_gpts)), avg_reward_gpts - std_reward_gpts, avg_reward_gpts + std_reward_gpts, alpha=0.2, color='r')
plt.legend(["GP-UCB1", "GPTS",  "Clairvoyant"])
plt.title("Instantaneous Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S2_instantaneous_reward.png")
# %%
