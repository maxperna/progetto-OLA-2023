# %%
import numpy as np
import matplotlib.pyplot as plt

from Environments.Users import  UserC1
from Environments.Environment_S5 import Non_Stationary_Environment

from Learners.CUSUM_UCB1_Learner import CUSUM_UCB1_Learner

from param import n_arms_pricing, T, n_experiments_S5, production_cost, p1_non_stationary
from param import N, eps, threshold, alpha 


# %% Non-Stationary setting:

Collector = UserC1(True, True, p1_non_stationary) 

n_phases = len(p1_non_stationary)
phases_len = np.ceil(T/n_phases)

optimum  = Collector.clairvoyant()

opt = np.array([])
for i in range(0,n_phases):
    opt = np.append(opt,optimum[i][2])

opt_vec = np.array([])
for i in range(0, T):
    current_phase = int(i/phases_len)
    opt_vec = np.append(opt_vec, opt[current_phase])

selected_bid = np.array([])
n_clicks = np.array([])
cost_of_click = np.array([])
for i in range(0,n_phases):
    selected_bid = np.append(selected_bid, optimum[i][1])
    n_clicks = np.append(n_clicks, Collector.click_vs_bid(optimum[i][1]))
    cost_of_click = np.append(cost_of_click, Collector.cost_vs_bid(optimum[i][1]))

# %% Optimization on N
N_cd = np.array([1, 2, 5, 7, 10, 15, 20]).astype(int)

avg_regret_cd = []
std_regret_cd = []
cum_avg_regret_cd = []
cum_std_regret_cd = []

for n in N_cd:

    cd_ucb1_rewards_per_experiment = []

    for e in range(0, n_experiments_S5):
        ns_env = Non_Stationary_Environment(selected_bid, production_cost, n_arms_pricing, Collector, T)
        cd_ucb1_learner = CUSUM_UCB1_Learner(n_arms_pricing, production_cost, n_clicks, cost_of_click, Collector._max_price, n, eps, threshold, alpha)


        for t in range(0, T):
            current_phase = int(t / phases_len)
            pulled_arm = cd_ucb1_learner.pull_arm()
            reward = ns_env.round(pulled_arm)
            cd_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm], current_phase)

        cd_ucb1_rewards_per_experiment.append(cd_ucb1_learner.collected_rewards)
    
    regret_sw = [opt_vec - result for result in cd_ucb1_rewards_per_experiment]
    avg_regret_cd.append(np.mean(opt_vec - cd_ucb1_rewards_per_experiment, axis=0))
    std_regret_cd.append(np.std(opt_vec - cd_ucb1_rewards_per_experiment, axis=0))
    cum_avg_regret_cd.append(np.mean(np.cumsum(opt_vec - cd_ucb1_rewards_per_experiment, axis=1), axis=0))
    cum_std_regret_cd.append(np.std(np.cumsum(opt_vec - cd_ucb1_rewards_per_experiment, axis=1), axis=0))

# %% Plot the cumulative regret
fig = plt.figure(0, facecolor="white")
plt.xlabel("t")
plt.ylabel("Regret")
for i in range(len(N_cd)):
    plt.plot(cum_avg_regret_cd[i])
for i in range(len(N_cd)):
    plt.fill_between(
        range(len(cum_avg_regret_cd[i])),
        cum_avg_regret_cd[i] - cum_std_regret_cd[i],
        cum_avg_regret_cd[i] + cum_std_regret_cd[i],
        alpha=0.2,
    )
plt.legend(N_cd)

plt.title("Cumulative Regret optimizing N")
fig = plt.gcf()
plt.show()

fig.savefig("results/S5_SA_CD_N_cumulative_regret.png")
# %%
# %% Optimization on eps
eps_cd = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8])

avg_regret_cd = []
std_regret_cd = []
cum_avg_regret_cd = []
cum_std_regret_cd = []

for eps_i in eps_cd:

    cd_ucb1_rewards_per_experiment = []

    for e in range(0, n_experiments_S5):
        ns_env = Non_Stationary_Environment(selected_bid, production_cost, n_arms_pricing, Collector, T)
        cd_ucb1_learner = CUSUM_UCB1_Learner(n_arms_pricing, production_cost, n_clicks, cost_of_click, Collector._max_price, N, eps_i, threshold, alpha)


        for t in range(0, T):
            current_phase = int(t / phases_len)
            pulled_arm = cd_ucb1_learner.pull_arm()
            reward = ns_env.round(pulled_arm)
            cd_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm], current_phase)

        cd_ucb1_rewards_per_experiment.append(cd_ucb1_learner.collected_rewards)
    
    regret_sw = [opt_vec - result for result in cd_ucb1_rewards_per_experiment]
    avg_regret_cd.append(np.mean(opt_vec - cd_ucb1_rewards_per_experiment, axis=0))
    std_regret_cd.append(np.std(opt_vec - cd_ucb1_rewards_per_experiment, axis=0))
    cum_avg_regret_cd.append(np.mean(np.cumsum(opt_vec - cd_ucb1_rewards_per_experiment, axis=1), axis=0))
    cum_std_regret_cd.append(np.std(np.cumsum(opt_vec - cd_ucb1_rewards_per_experiment, axis=1), axis=0))

# %% Plot the cumulative regret
fig = plt.figure(0, facecolor="white")
plt.xlabel("t")
plt.ylabel("Regret")
for i in range(len(eps_cd)):
    plt.plot(cum_avg_regret_cd[i])
for i in range(len(eps_cd)):
    plt.fill_between(
        range(len(cum_avg_regret_cd[i])),
        cum_avg_regret_cd[i] - cum_std_regret_cd[i],
        cum_avg_regret_cd[i] + cum_std_regret_cd[i],
        alpha=0.2,
    )
plt.legend(eps_cd)

plt.title("Cumulative Regret optimizing eps")
fig = plt.gcf()
plt.show()

fig.savefig("results/S5_SA_CD_eps_cumulative_regret.png")

# %%
# %% Optimization on threshold
THRESHOLDS = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1])

avg_regret_cd = []
std_regret_cd = []
cum_avg_regret_cd = []
cum_std_regret_cd = []

for thre in THRESHOLDS:

    cd_ucb1_rewards_per_experiment = []

    for e in range(0, n_experiments_S5):
        ns_env = Non_Stationary_Environment(selected_bid, production_cost, n_arms_pricing, Collector, T)
        cd_ucb1_learner = CUSUM_UCB1_Learner(n_arms_pricing, production_cost, n_clicks, cost_of_click, Collector._max_price, N, eps, thre, alpha)


        for t in range(0, T):
            current_phase = int(t / phases_len)
            pulled_arm = cd_ucb1_learner.pull_arm()
            reward = ns_env.round(pulled_arm)
            cd_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm], current_phase)

        cd_ucb1_rewards_per_experiment.append(cd_ucb1_learner.collected_rewards)
    
    regret_sw = [opt_vec - result for result in cd_ucb1_rewards_per_experiment]
    avg_regret_cd.append(np.mean(opt_vec - cd_ucb1_rewards_per_experiment, axis=0))
    std_regret_cd.append(np.std(opt_vec - cd_ucb1_rewards_per_experiment, axis=0))
    cum_avg_regret_cd.append(np.mean(np.cumsum(opt_vec - cd_ucb1_rewards_per_experiment, axis=1), axis=0))
    cum_std_regret_cd.append(np.std(np.cumsum(opt_vec - cd_ucb1_rewards_per_experiment, axis=1), axis=0))

# %% Plot the cumulative regret
fig = plt.figure(0, facecolor="white")
plt.xlabel("t")
plt.ylabel("Regret")
for i in range(len(THRESHOLDS)):
    plt.plot(cum_avg_regret_cd[i])
for i in range(len(THRESHOLDS)):
    plt.fill_between(
        range(len(cum_avg_regret_cd[i])),
        cum_avg_regret_cd[i] - cum_std_regret_cd[i],
        cum_avg_regret_cd[i] + cum_std_regret_cd[i],
        alpha=0.2,
    )
plt.legend(THRESHOLDS)

plt.title("Cumulative Regret optimizing threshold")
fig = plt.gcf()
plt.show()

fig.savefig("results/S5_SA_CD_threshold_cumulative_regret.png")
# %%
