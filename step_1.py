# %%
import numpy as np
import matplotlib.pyplot as plt

from Environments.Users import  UserC1, UserC2, UserC3
from Environments.Environment import Environment
from Learners.TS_Learner import TS_Learner
from Learners.Greedy_Learner import Greedy_Learner
from Learners.UCB1_Learner import UCB1_Learner

# %% Create the three classes of users
n_arm = 5
Collector = UserC1()

#selected_bid = np.argmax(Collector.click_vs_bid())
selected_bid = 0.5
#n_clicks = np.max(Collector.click_vs_bid())
#cumulative_cost=Collector.cumulative_cost_vs_bid(selected_bid)
n_clicks = Collector.click_vs_bid(selected_bid)
cumulative_cost=Collector.cumulative_cost_vs_bid(selected_bid)
production_cost = 75

avg_n_users = 1     # for future optimization
#opt = max(Collector.prices * Collector.probabilities) * avg_n_users
opt = max((Collector.prices-production_cost)*Collector.probabilities*n_clicks - cumulative_cost*n_clicks)

T = 365

n_experiments = 100
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []
ucb1_rewards_per_experiment = []

# %% Run the experiments
for e in range(0, n_experiments):
    env = Environment(n_arms=n_arm, user=Collector)
    ts_learner = TS_Learner(n_arms=n_arm, production_cost=production_cost, n_clicks=n_clicks, cumulative_cost=cumulative_cost)
    gr_learner = Greedy_Learner(n_arms=n_arm, production_cost=production_cost, n_clicks=n_clicks, cumulative_cost=cumulative_cost)
    ucb1_learner = UCB1_Learner(n_arms=n_arm, production_cost=production_cost, n_clicks=n_clicks, cumulative_cost=cumulative_cost, M = Collector._max_price)
    for t in range(0, T):
        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        # UCB1 Learner
        pulled_arm = ucb1_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm(Collector.prices)
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])


    gr_rewards_per_experiment.append(gr_learner.collected_rewards)
    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)

# %% Compute the regret
# opt = env.max_reward()  # TODO implement this

regret_greedy = opt - gr_rewards_per_experiment  # row = exp, col = t
avg_regret_greedy = np.mean(regret_greedy, axis=0)
std_regret_greedy = np.std(regret_greedy, axis=0)
cum_avg_regret_greedy = np.mean(np.cumsum(regret_greedy, axis=1), axis=0)
cum_std_regret_greedy = np.std(np.cumsum(regret_greedy, axis=1), axis=0)

regret_ucb1 = opt - ucb1_rewards_per_experiment  # row = exp, col = t
avg_regret_ucb1 = np.mean(regret_ucb1, axis=0)
std_regret_ucb1 = np.std(regret_ucb1, axis=0)
cum_avg_regret_ucb1 = np.mean(np.cumsum(regret_ucb1, axis=1), axis=0)
cum_std_regret_ucb1 = np.std(np.cumsum(regret_ucb1, axis=1), axis=0)

regret_ts = opt - ts_rewards_per_experiment  # row = exp, col = t
avg_regret_ts = np.mean(regret_ts, axis=0)
std_regret_ts = np.std(regret_ts, axis=0)
cum_avg_regret_ts = np.mean(np.cumsum(regret_ts, axis=1), axis=0)
cum_std_regret_ts = np.std(np.cumsum(regret_ts, axis=1), axis=0)


# %% Plot the cumulative regret
fig = plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(cum_avg_regret_greedy, 'g')
plt.plot(cum_avg_regret_ucb1, 'b')
plt.plot(cum_avg_regret_ts, 'r')
plt.fill_between(range(len(cum_avg_regret_greedy)), cum_avg_regret_greedy - cum_std_regret_greedy, cum_avg_regret_greedy + cum_std_regret_greedy, alpha=0.2, color='g')
plt.fill_between(range(len(cum_avg_regret_ucb1)), cum_avg_regret_ucb1 - cum_std_regret_ucb1, cum_avg_regret_ucb1 + cum_std_regret_ucb1, alpha=0.2, color='b')
plt.fill_between(range(len(cum_avg_regret_ts)), cum_avg_regret_ts - cum_std_regret_ts, cum_avg_regret_ts + cum_std_regret_ts, alpha=0.2, color='r')
plt.legend(["Greedy", "UCB1", "TS"])
plt.title("Cumulative Regret")
fig = plt.gcf()
plt.show()

fig.savefig("results/S1_cumulative_regret.png")

# %% Plot the instantaneous regret
fig = plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_greedy, 'g')
plt.plot(avg_regret_ucb1, 'b')
plt.plot(avg_regret_ts, 'r')
plt.fill_between(range(len(avg_regret_greedy)), avg_regret_greedy - std_regret_greedy, avg_regret_greedy + std_regret_greedy, alpha=0.2, color='g')
plt.fill_between(range(len(avg_regret_ucb1)), avg_regret_ucb1 - std_regret_ucb1, avg_regret_ucb1 + std_regret_ucb1, alpha=0.2, color='b')
plt.fill_between(range(len(avg_regret_ts)), avg_regret_ts - std_regret_ts, avg_regret_ts + std_regret_ts, alpha=0.2, color='r')
plt.hlines(0, 0, T, colors='black', linestyles='dashed')
plt.legend(["Greedy", "UCB1", "TS"])
plt.title("Instantaneous Regret")
fig = plt.gcf()
plt.show()

fig.savefig("results/S1_instantaneous_regret.png")

# %% Compute the reward
avg_reward_greedy = np.mean(gr_rewards_per_experiment, axis=0)
std_reward_greedy = np.std(gr_rewards_per_experiment, axis=0)
cum_avg_reward_greedy = np.mean(np.cumsum(gr_rewards_per_experiment, axis=1), axis=0)
cum_std_rreward_greedy = np.std(np.cumsum(gr_rewards_per_experiment, axis=1), axis=0)

avg_reward_ucb1 = np.mean(ucb1_rewards_per_experiment, axis=0)
std_reward_ucb1 = np.std(ucb1_rewards_per_experiment, axis=0)
cum_avg_reward_ucb1 = np.mean(np.cumsum(ucb1_rewards_per_experiment, axis=1), axis=0)
cum_std_reward_ucb1 = np.std(np.cumsum(ucb1_rewards_per_experiment, axis=1), axis=0)

avg_reward_ts = np.mean(ts_rewards_per_experiment, axis=0)
std_reward_ts = np.std(ts_rewards_per_experiment, axis=0)
cum_avg_reward_ts = np.mean(np.cumsum(ts_rewards_per_experiment, axis=1), axis=0)
cum_std_rreward_ts = np.std(np.cumsum(ts_rewards_per_experiment, axis=1), axis=0)


# %% Plot the cumulative reward
plt.figure(2)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(cum_avg_reward_greedy, 'g')
plt.plot(cum_avg_reward_ucb1, 'b')
plt.plot(cum_avg_reward_ts, 'r')
plt.fill_between(range(len(cum_avg_reward_greedy)), cum_avg_reward_greedy - cum_std_rreward_greedy, cum_avg_reward_greedy + cum_std_rreward_greedy, alpha=0.2, color='g')
plt.fill_between(range(len(cum_avg_reward_ucb1)), cum_avg_reward_ucb1 - cum_std_reward_ucb1, cum_avg_reward_ucb1 + cum_std_reward_ucb1, alpha=0.2, color='r')
plt.fill_between(range(len(cum_avg_reward_ts)), cum_avg_reward_ts - cum_std_rreward_ts, cum_avg_reward_ts + cum_std_rreward_ts, alpha=0.2, color='b')
plt.legend(["Greedy", "UCB1", "TS"])
plt.title("Cumulative Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S1_cumulative_reward.png")

# %% Plot the instantaneous reward
plt.figure(3)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(avg_reward_greedy, 'g')
plt.plot(avg_reward_ucb1, 'b')
plt.plot(avg_reward_ts, 'r')
plt.hlines(y=opt, xmin=0, xmax=T, colors='k', linestyles='dashed')
plt.fill_between(range(len(avg_reward_greedy)), avg_reward_greedy - std_reward_greedy, avg_reward_greedy + std_reward_greedy, alpha=0.2, color='g')
plt.fill_between(range(len(avg_reward_ucb1)), avg_reward_ucb1 - std_reward_ucb1, avg_reward_ucb1 + std_reward_ucb1, alpha=0.2, color='b')
plt.fill_between(range(len(avg_reward_ts)), avg_reward_ts - std_reward_ts, avg_reward_ts + std_reward_ts, alpha=0.2, color='r')
plt.legend(["Greedy", "UCB1", "TS", "Clairvoyant"])
plt.title("Instantaneous Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S1_instantaneous_reward.png")


# %%
