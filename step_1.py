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
opt = max((Collector.prices-production_cost)*Collector.probabilities*n_clicks - cumulative_cost)

T = 365

n_experiments = 1000
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
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm(Collector.prices)
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        # UCB1 Learner
        pulled_arm = ucb1_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

    gr_rewards_per_experiment.append(gr_learner.collected_rewards)
    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)

# %% Compute the regret
avg_regret_gr = np.mean(opt - gr_rewards_per_experiment, axis=0)
avg_regret_ucb1 = np.mean(opt - ucb1_rewards_per_experiment, axis=0)
avg_regret_ts = np.mean(opt - ts_rewards_per_experiment, axis=0)

std_regret_gr = np.std(opt - gr_rewards_per_experiment, axis=0)
std_regret_ucb1 = np.std(opt - ucb1_rewards_per_experiment, axis=0)
std_regret_ts = np.std(opt - ts_rewards_per_experiment, axis=0)


# %% Plot the cumulative regret
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(avg_regret_gr), 'g')
plt.plot(np.cumsum(avg_regret_ucb1), 'b')
plt.plot(np.cumsum(avg_regret_ts), 'r')
plt.legend(["Greedy", "UCB1", "TS"])
plt.title("Cumulative Regret")
plt.show()

# %% Plot the cumulative reward
plt.figure(1)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.cumsum(np.mean(gr_rewards_per_experiment, axis=0)), 'g')
plt.plot(np.cumsum(np.mean(ucb1_rewards_per_experiment, axis=0)), 'b')
plt.plot(np.cumsum(np.mean(ts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["Greedy", "UCB1", "TS"])
plt.title("Cumulative Reward")
plt.show()

# %% Plot the instantaneous regret
plt.figure(2)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_gr, 'g')
plt.plot(avg_regret_ucb1, 'b')
plt.plot(avg_regret_ts, 'r')
plt.hlines(y=0, xmin=0, xmax=T, colors='k', linestyles='dashed')
#plt.ylim(-0.1, 1)
plt.legend(["Greedy", "UCB1", "TS", "Clairvoyant"])
plt.title("Instantaneous Regret")
plt.show()

# %% Plot the instantaneous reward
plt.figure(3)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(gr_rewards_per_experiment, axis=0), 'g')
plt.plot(np.mean(ucb1_rewards_per_experiment, axis=0), 'b')
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.hlines(y=opt, xmin=0, xmax=T, colors='k', linestyles='dashed')
plt.legend(["Greedy", "UCB1", "TS", "Clairvoyant"])
plt.title("Instantaneous Reward")
plt.show()


# %% Plot the instantaneous regret with standard deviation
plt.figure(4)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_gr, 'g')
plt.plot(avg_regret_ucb1, 'b')
plt.plot(avg_regret_ts, 'r')
plt.hlines(y=0, xmin=0, xmax=T, colors='k', linestyles='dashed')
plt.fill_between(range(len(avg_regret_gr)), avg_regret_gr - std_regret_gr, avg_regret_gr + std_regret_gr, color='g', alpha=0.2)
plt.fill_between(range(len(avg_regret_ucb1)), avg_regret_ucb1 - std_regret_ucb1, avg_regret_ucb1 + std_regret_ucb1, color='b', alpha=0.2)
plt.fill_between(range(len(avg_regret_ts)), avg_regret_ts - std_regret_ts, avg_regret_ts + std_regret_ts, color='r', alpha=0.2)
plt.legend(["Greedy", "UCB1", "TS", "Clairvoyant"])
plt.title("Instantaneous Regret with Standard Deviation")
plt.show()




# %% Plot of cumulative regret with variance
avg_cum_regret_gr = np.cumsum(avg_regret_gr)
avg_cum_regret_ucb1 = np.cumsum(avg_regret_ucb1)
avg_cum_regret_ts = np.cumsum(avg_regret_ts)

std_cum_regret_gr = np.cumsum(std_regret_gr)
std_cum_regret_ucb1 = np.cumsum(std_regret_ucb1)
std_cum_regret_ts = np.cumsum(std_regret_ts)

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_cum_regret_gr, 'g')
plt.plot(avg_cum_regret_ucb1, 'b')
plt.plot(avg_cum_regret_ts, 'r')
plt.fill_between(range(len(avg_cum_regret_gr)), avg_cum_regret_gr - std_cum_regret_gr, avg_cum_regret_gr + std_cum_regret_gr, alpha=0.2, color='g')
plt.fill_between(range(len(avg_cum_regret_ucb1)), avg_cum_regret_ucb1 - std_cum_regret_ucb1, avg_cum_regret_ucb1 + std_cum_regret_ucb1, alpha=0.2, color='b')
plt.fill_between(range(len(avg_cum_regret_ts)), avg_cum_regret_ts - std_cum_regret_ts, avg_cum_regret_ts + std_cum_regret_ts, alpha=0.2, color='r')
plt.legend(["Greedy", "UCB1", "TS"])
plt.title("Cumulative Regret with standard deviation")
plt.show()


# %%
