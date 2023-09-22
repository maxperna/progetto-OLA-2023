# %%
import numpy as np
import matplotlib.pyplot as plt

from Environments.Users import  UserC1
from Environments.Environment_S5 import Non_Stationary_Environment

from Learners.UCB1_Learner import UCB1_Learner
from Learners.SW_UCB1_Learner import SW_UCB1_Learner
from Learners.CUSUM_UCB1_Learner import CUSUM_UCB1_Learner

from param import n_arms_pricing, T, n_experiments_S5, production_cost, p1_non_stationary
from param import Tau, N, eps, threshold, alpha 

# %% Non-Stationary setting:

Collector = UserC1(True, True, p1_non_stationary) 

n_phases = len(p1_non_stationary)
phases_len = np.ceil(T/n_phases)

optimum  = Collector.clairvoyant()
print(optimum)

opt = np.array([])
for i in range(0,n_phases):
    opt = np.append(opt,optimum[i][2])
#print(opt)

opt_vec = np.array([])
for i in range(0, T):
    current_phase = int(i/phases_len)
    opt_vec = np.append(opt_vec, opt[current_phase])
#print(opt_vec)


selected_bid = np.array([])
n_clicks = np.array([])
cost_of_click = np.array([])
for i in range(0,n_phases):
    selected_bid = np.append(selected_bid, optimum[i][1])
    n_clicks = np.append(n_clicks, Collector.click_vs_bid(optimum[i][1]))
    cost_of_click = np.append(cost_of_click, Collector.cost_vs_bid(optimum[i][1]))
print(selected_bid)
print(n_clicks)
print(cost_of_click)


ucb1_rewards_per_experiment = []
sw_ucb1_rewards_per_experiment = []
cd_ucb1_rewards_per_experiment = []


# %% Plot the demand curves and the expected reward curves for each phase:
fig = plt.figure()
Collector.plot_demand_curve()
plt.legend(['phase 1', 'phase 2', 'phase 3'])
plt.title("Conversion Rate Curves")
plt.show()

# %% Run the experiments
n_experiments_S5 = 100
for e in range(0, n_experiments_S5):
    
    ns_env_1 = Non_Stationary_Environment(selected_bid, production_cost, n_arms_pricing, Collector, T)
    ns_env_2 = Non_Stationary_Environment(selected_bid, production_cost, n_arms_pricing, Collector, T)
    ns_env_3 = Non_Stationary_Environment(selected_bid, production_cost, n_arms_pricing, Collector, T)

    ucb1_learner = UCB1_Learner(n_arms_pricing, production_cost, n_clicks, cost_of_click, Collector._max_price)
    sw_ucb1_learner = SW_UCB1_Learner(n_arms_pricing, production_cost, n_clicks, cost_of_click, Collector._max_price, Tau)
    cd_ucb1_learner = CUSUM_UCB1_Learner(n_arms_pricing, production_cost, n_clicks, cost_of_click, Collector._max_price, N, eps, threshold, alpha)

    for t in range(0, T):

        current_phase = int(t / phases_len)
        
        # UCB1 Learner:
        pulled_arm = ucb1_learner.pull_arm() 
        reward = ns_env_1.round(pulled_arm) 
        ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm], current_phase) 
        
        # SW_UCB1 Learner:
        pulled_arm = sw_ucb1_learner.pull_arm()
        reward = ns_env_2.round(pulled_arm)
        sw_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm], current_phase)

        # CD_UCB1 Learner:
        pulled_arm = cd_ucb1_learner.pull_arm()
        reward = ns_env_3.round(pulled_arm)
        cd_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm], current_phase)


    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    sw_ucb1_rewards_per_experiment.append(sw_ucb1_learner.collected_rewards)
    cd_ucb1_rewards_per_experiment.append(cd_ucb1_learner.collected_rewards)


# %% Compute the regret
regret_ucb1 = [opt_vec - result for result in ucb1_rewards_per_experiment]
avg_regret_ucb1 = np.mean(regret_ucb1, axis=0)
std_regret_ucb1 = np.std(regret_ucb1, axis=0)
cum_avg_regret_ucb1 = np.mean(np.cumsum(regret_ucb1, axis=1), axis=0)
cum_std_regret_ucb1 = np.std(np.cumsum(regret_ucb1, axis=1), axis=0)

# sw regret
regret_sw_ucb1 = [opt_vec - result for result in sw_ucb1_rewards_per_experiment]
avg_regret_sw_ucb1 = np.mean(regret_sw_ucb1, axis=0)
std_regret_sw_ucb1 = np.std(regret_sw_ucb1, axis=0)
cum_avg_regret_sw_ucb1 = np.mean(np.cumsum(regret_sw_ucb1, axis=1), axis=0)
cum_std_regret_sw_ucb1 = np.std(np.cumsum(regret_sw_ucb1, axis=1), axis=0)

# cd regret
regret_cd_ucb1 = [opt_vec - result for result in cd_ucb1_rewards_per_experiment]
avg_regret_cd_ucb1 = np.mean(regret_cd_ucb1, axis=0)
std_regret_cd_ucb1 = np.std(regret_cd_ucb1, axis=0)
cum_avg_regret_cd_ucb1 = np.mean(np.cumsum(regret_cd_ucb1, axis=1), axis=0)
cum_std_regret_cd_ucb1 = np.std(np.cumsum(regret_cd_ucb1, axis=1), axis=0)


# %% Plot the cumulative regret
fig = plt.figure(0, facecolor="white")
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(cum_avg_regret_ucb1, "b")
plt.plot(cum_avg_regret_sw_ucb1, "orange")
plt.plot(cum_avg_regret_cd_ucb1, 'r')
plt.fill_between(
    range(len(cum_avg_regret_ucb1)),
    cum_avg_regret_ucb1 - cum_std_regret_ucb1,
    cum_avg_regret_ucb1 + cum_std_regret_ucb1,
    alpha=0.2,
    color="b",
)
plt.fill_between(
    range(len(cum_avg_regret_sw_ucb1)),
    cum_avg_regret_sw_ucb1 - cum_std_regret_sw_ucb1,
    cum_avg_regret_sw_ucb1 + cum_std_regret_sw_ucb1,
    alpha=0.2,
    color="orange",
)
plt.fill_between(
    range(len(cum_avg_regret_cd_ucb1)),
    cum_avg_regret_cd_ucb1 - cum_std_regret_cd_ucb1,
    cum_avg_regret_cd_ucb1 + cum_std_regret_cd_ucb1,
    alpha=0.2,
    color="r",
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1"]) 
plt.title("Cumulative Regret")
fig = plt.gcf()
plt.show()

fig.savefig("results/S5_cumulative_regret.png")

# %% Plot the instantaneous regret with standard deviation
fig = plt.figure(1, facecolor="white")
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_ucb1, "b")
plt.plot(avg_regret_sw_ucb1, "orange")
plt.plot(avg_regret_cd_ucb1, "r")
plt.hlines(0, 0, T, colors="black", linestyles="dashed")
plt.fill_between(
    range(len(avg_regret_ucb1)),
    avg_regret_ucb1 - std_regret_ucb1,
    avg_regret_ucb1 + std_regret_ucb1,
    alpha=0.2,
    color="b",
)
plt.fill_between(
    range(len(avg_regret_sw_ucb1)),
    avg_regret_sw_ucb1 - std_regret_sw_ucb1,
    avg_regret_sw_ucb1 + std_regret_sw_ucb1,
    alpha=0.2,
    color="orange",
)
plt.fill_between(
    range(len(avg_regret_cd_ucb1)),
    avg_regret_cd_ucb1 - std_regret_cd_ucb1,
    avg_regret_cd_ucb1 + std_regret_cd_ucb1,
    alpha=0.2,
    color='r',
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1"])  
plt.title("Instantaneous Regret with Standard Deviation")
fig = plt.gcf()
plt.show()

fig.savefig("results/S5_instantaneous_regret.png")

# %% Compute the reward
avg_reward_ucb1 = np.mean(ucb1_rewards_per_experiment, axis=0)
std_reward_ucb1 = np.std(ucb1_rewards_per_experiment, axis=0)
cum_avg_reward_ucb1 = np.mean(np.cumsum(ucb1_rewards_per_experiment, axis=1), axis=0)
cum_std_reward_ucb1 = np.std(np.cumsum(ucb1_rewards_per_experiment, axis=1), axis=0)

avg_reward_sw_ucb1 = np.mean(sw_ucb1_rewards_per_experiment, axis=0)
std_reward_sw_ucb1 = np.std(sw_ucb1_rewards_per_experiment, axis=0)
cum_avg_reward_sw_ucb1 = np.mean(np.cumsum(sw_ucb1_rewards_per_experiment, axis=1), axis=0)
cum_std_rreward_sw_ucb1 = np.std(np.cumsum(sw_ucb1_rewards_per_experiment, axis=1), axis=0)

avg_reward_cd_ucb1 = np.mean(cd_ucb1_rewards_per_experiment, axis=0)
std_reward_cd_ucb1 = np.std(cd_ucb1_rewards_per_experiment, axis=0)
cum_avg_reward_cd_ucb1 = np.mean(np.cumsum(cd_ucb1_rewards_per_experiment, axis=1), axis=0)
cum_std_rreward_cd_ucb1 = np.std(np.cumsum(cd_ucb1_rewards_per_experiment, axis=1), axis=0)

# %% Plot the cumulative reward
plt.figure(2, facecolor="white")
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(cum_avg_reward_ucb1, "b")
plt.plot(cum_avg_reward_sw_ucb1, "orange")
plt.plot(cum_avg_reward_cd_ucb1, 'r')
plt.fill_between(
    range(len(cum_avg_reward_ucb1)),
    cum_avg_reward_ucb1 - cum_std_reward_ucb1,
    cum_avg_reward_ucb1 + cum_std_reward_ucb1,
    alpha=0.2,
    color="b",
)
plt.fill_between(
    range(len(cum_avg_reward_sw_ucb1)),
    cum_avg_reward_sw_ucb1 - cum_std_rreward_sw_ucb1,
    cum_avg_reward_sw_ucb1 + cum_std_rreward_sw_ucb1,
    alpha=0.2,
    color="orange",
)
plt.fill_between(
    range(len(cum_avg_reward_cd_ucb1)),
    cum_avg_reward_cd_ucb1 - cum_std_rreward_cd_ucb1,
    cum_avg_reward_cd_ucb1 + cum_std_rreward_cd_ucb1,
    alpha=0.2,
    color="r",
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1"]) 
plt.title("Cumulative Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S5_cumulative_reward.png")

# %% Plot the instantaneous reward
plt.figure(3, facecolor="white")
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(avg_reward_ucb1, "b")
plt.plot(avg_reward_sw_ucb1, "orange")
plt.plot(avg_reward_cd_ucb1, 'r')
plt.plot(range(T), opt_vec, "k--")
plt.fill_between(
    range(len(avg_reward_ucb1)),
    avg_reward_ucb1 - std_reward_ucb1,
    avg_reward_ucb1 + std_reward_ucb1,
    alpha=0.2,
    color="b",
)
plt.fill_between(
    range(len(avg_reward_sw_ucb1)),
    avg_reward_sw_ucb1 - std_reward_sw_ucb1,
    avg_reward_sw_ucb1 + std_reward_sw_ucb1,
    alpha=0.2,
    color="orange",
)
plt.fill_between(
    range(len(avg_reward_cd_ucb1)),
    avg_reward_cd_ucb1 - std_reward_cd_ucb1,
    avg_reward_cd_ucb1 + std_reward_cd_ucb1,
    alpha=0.2,
    color="r",
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1"])  
plt.title("Instantaneous Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S5_instantaneous_reward.png")

# %%
