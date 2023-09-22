# %%
import numpy as np
import matplotlib.pyplot as plt

from Environments.Users import  UserC1
from Environments.Environment_S5 import Non_Stationary_Environment

from Learners.SW_UCB1_Learner import SW_UCB1_Learner

from param import n_arms_pricing, T, n_experiments_S5, production_cost, p1_non_stationary, Tau 

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


# %% Plot the demand curves and the expected reward curves for each phase:
fig = plt.figure()
Collector.plot_demand_curve()
plt.legend(['phase 1', 'phase 2', 'phase 3'])
plt.title("Conversion Rate Curves")
plt.show()

# %% Optimization on Slididing Window length
n_experiments_S5 = 100

TAU = (np.sqrt(T) * np.array([0.5, 1, 1.5, 2, 4, 8, 12, 16, 19])).astype(int)

avg_regret_sw = []
std_regret_sw = []
cum_avg_regret_sw = []
cum_std_regret_sw = []

for tau in TAU:

    sw_ucb1_rewards_per_experiment = []

    for e in range(0, n_experiments_S5):
        ns_env = Non_Stationary_Environment(selected_bid, production_cost, n_arms_pricing, Collector, T)
        sw_ucb1_learner = SW_UCB1_Learner(n_arms_pricing, production_cost, n_clicks, cost_of_click, Collector._max_price, tau)

        for t in range(0, T):
            current_phase = int(t / phases_len)
            pulled_arm = sw_ucb1_learner.pull_arm()
            reward = ns_env.round(pulled_arm)
            sw_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm], current_phase)

        sw_ucb1_rewards_per_experiment.append(sw_ucb1_learner.collected_rewards)
    
    regret_sw = [opt_vec - result for result in sw_ucb1_rewards_per_experiment]
    avg_regret_sw.append(np.mean(opt_vec - sw_ucb1_rewards_per_experiment, axis=0))
    std_regret_sw.append(np.std(opt_vec - sw_ucb1_rewards_per_experiment, axis=0))
    cum_avg_regret_sw.append(np.mean(np.cumsum(opt_vec - sw_ucb1_rewards_per_experiment, axis=1), axis=0))
    cum_std_regret_sw.append(np.std(np.cumsum(opt_vec - sw_ucb1_rewards_per_experiment, axis=1), axis=0))

# %% Plot the cumulative regret
fig = plt.figure(0, facecolor="white")
plt.xlabel("t")
plt.ylabel("Regret")
for i in range(len(TAU)):
    plt.plot(cum_avg_regret_sw[i])
for i in range(len(TAU)):
    plt.fill_between(
        range(len(cum_avg_regret_sw[i])),
        cum_avg_regret_sw[i] - cum_std_regret_sw[i],
        cum_avg_regret_sw[i] + cum_std_regret_sw[i],
        alpha=0.2,
    )
plt.legend(TAU)

plt.title("Cumulative Regret")
fig = plt.gcf()
plt.show()

fig.savefig("results/S5_SA_sw_cumulative_regret.png")
# %%
