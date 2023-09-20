# %%
import numpy as np
import matplotlib.pyplot as plt

from Environments.Users import  UserC1
from Environments.Non_Stationary_Environment import Non_Stationary_Environment

from Learners.UCB1_Learner import UCB1_Learner
#from Learners.SW_UCB1_Learner import SW_UCB1_Learner
#from Learners.CUSUM_UCB1_Learner import CUSUM_UCB1_Learner

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
print(opt)

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

fig = plt.figure()
Collector.plot_expected_reward()
plt.legend(['phase 1', 'phase 2', 'phase 3'])
plt.title("Expected Rewards Curves")
plt.show()


# %% Run the experiments

for e in range(0, n_experiments):
    
    ns_env_1 = Non_Stationary_Environment(n_arm, Collector, T)
    ns_env_2 = Non_Stationary_Environment(n_arm, Collector, T)
    ns_env_3 = Non_Stationary_Environment(n_arm, Collector, T)

    ucb1_learner = UCB1_Learner_old(n_arm, Collector._max_price)
    sw_ucb1_learner = SW_UCB1_Learner_old(n_arm, Collector._max_price, Tau)
    cd_ucb1_learner = CUSUM_UCB1_Learner_old(n_arm, Collector._max_price, N, eps, threshold, alpha)

    for t in range(0, T):
        
        # UCB1 Learner:
        pulled_arm = ucb1_learner.pull_arm() # determino l'arm da pullare
        reward = ns_env_1.round(pulled_arm) # osservo il feedback corrispondente 
        ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm]) # aggiorno
        
        # aux:
        #ucb1_instantaneous_regret.append(opt_vec[t] - reward * Collector.prices[pulled_arm])
        #ucb1_cumulative_regret.append(np.sum(ucb1_instantaneous_regret))

        # SW_UCB1 Learner:
        pulled_arm = sw_ucb1_learner.pull_arm()
        reward = ns_env_2.round(pulled_arm)
        sw_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        # CD_UCB1 Learner:
        pulled_arm = cd_ucb1_learner.pull_arm()
        reward = ns_env_3.round(pulled_arm)
        cd_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])


    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    sw_ucb1_rewards_per_experiment.append(sw_ucb1_learner.collected_rewards)
    cd_ucb1_rewards_per_experiment.append(cd_ucb1_learner.collected_rewards)



# %% Compute the regret
avg_regret_ucb1 = np.mean(opt_vec - ucb1_rewards_per_experiment, axis=0)
avg_regret_sw_ucb1 = np.mean(opt_vec - sw_ucb1_rewards_per_experiment, axis=0)
avg_regret_cd_ucb1 = np.mean(opt_vec - cd_ucb1_rewards_per_experiment, axis=0)

std_regret_ucb1 = np.std(opt_vec - ucb1_rewards_per_experiment, axis=0)
std_regret_sw_ucb1 = np.std(opt_vec - sw_ucb1_rewards_per_experiment, axis=0)
std_regret_cd_ucb1 = np.std(opt_vec - cd_ucb1_rewards_per_experiment, axis=0)

# print(np.shape(ucb1_rewards_per_experiment))
# print(np.shape(sw_ucb1_rewards_per_experiment))

# print(np.shape(avg_regret_ucb1))
# print(np.shape(avg_regret_sw_ucb1))

# %% Plot the cumulative regret
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(avg_regret_ucb1), 'g')
plt.plot(np.cumsum(avg_regret_sw_ucb1), 'b')
plt.plot(np.cumsum(avg_regret_cd_ucb1), 'r')
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1"])
plt.title("Cumulative Regret")
plt.show()

# %% Plot the cumulative reward
plt.figure(1)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.cumsum(np.mean(ucb1_rewards_per_experiment, axis=0)), 'g')
plt.plot(np.cumsum(np.mean(sw_ucb1_rewards_per_experiment, axis=0)), 'b')
plt.plot(np.cumsum(np.mean(cd_ucb1_rewards_per_experiment, axis=0)), 'r')
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1"])
plt.title("Cumulative Reward")
plt.show()

# %% Plot the instantaneous regret
plt.figure(2)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_ucb1, 'g')
plt.plot(avg_regret_sw_ucb1, 'b')
plt.plot(avg_regret_cd_ucb1, 'r')
plt.hlines(y=0, xmin=0, xmax=T, colors='k', linestyles='dashed')
#plt.ylim(-0.1, 1)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1", "Clairvoyant"])
plt.title("Instantaneous Regret")
plt.show()

# %% Plot the instantaneous reward
plt.figure(3)
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(ucb1_rewards_per_experiment, axis=0), 'g')
plt.plot(np.mean(sw_ucb1_rewards_per_experiment, axis=0), 'b')
plt.plot(np.mean(cd_ucb1_rewards_per_experiment, axis=0), 'r')
plt.hlines(y=opt[0], xmin=0, xmax=phases_len, colors='k', linestyles='dashed')
plt.hlines(y=opt[1], xmin=phases_len+1, xmax=phases_len*2, colors='k', linestyles='dashed')
plt.hlines(y=opt[2], xmin=phases_len*2+1, xmax=T, colors='k', linestyles='dashed')
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1", "Clairvoyant"])
plt.title("Instantaneous Reward")
plt.show()


# %% Plot the instantaneous regret with standard deviation
plt.figure(4)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_ucb1, 'g')
plt.plot(avg_regret_sw_ucb1, 'b')
plt.plot(avg_regret_sw_ucb1, 'r')
plt.hlines(y=0, xmin=0, xmax=T, colors='k', linestyles='dashed')
plt.fill_between(range(len(avg_regret_ucb1)), avg_regret_ucb1 - std_regret_ucb1, avg_regret_ucb1 + std_regret_ucb1, color='g', alpha=0.2)
plt.fill_between(range(len(avg_regret_sw_ucb1)), avg_regret_sw_ucb1 - std_regret_sw_ucb1, avg_regret_sw_ucb1 + std_regret_sw_ucb1, color='b', alpha=0.2)
plt.fill_between(range(len(avg_regret_cd_ucb1)), avg_regret_cd_ucb1 - std_regret_cd_ucb1, avg_regret_cd_ucb1 + std_regret_cd_ucb1, color='r', alpha=0.2)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1", "Clairvoyant"])
plt.title("Instantaneous Regret with Standard Deviation")
plt.show()


# %% Plot of cumulative regret with variance
avg_cum_regret_ucb1 = np.cumsum(avg_regret_ucb1)
avg_cum_regret_sw_ucb1 = np.cumsum(avg_regret_sw_ucb1)
avg_cum_regret_cd_ucb1 = np.cumsum(avg_regret_cd_ucb1)

std_cum_regret_ucb1 = np.cumsum(std_regret_ucb1)
std_cum_regret_sw_ucb1 = np.cumsum(std_regret_sw_ucb1)
std_cum_regret_cd_ucb1 = np.cumsum(std_regret_cd_ucb1)

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_cum_regret_ucb1, 'g')
plt.plot(avg_cum_regret_sw_ucb1, 'b')
plt.plot(avg_cum_regret_cd_ucb1, 'r')
plt.fill_between(range(len(avg_cum_regret_ucb1)), avg_cum_regret_ucb1 - std_cum_regret_ucb1, avg_cum_regret_ucb1 + std_cum_regret_ucb1, alpha=0.2, color='g')
plt.fill_between(range(len(avg_cum_regret_sw_ucb1)), avg_cum_regret_sw_ucb1 - std_cum_regret_sw_ucb1, avg_cum_regret_sw_ucb1 + std_cum_regret_sw_ucb1, alpha=0.2, color='b')
plt.fill_between(range(len(avg_cum_regret_cd_ucb1)), avg_cum_regret_cd_ucb1 - std_cum_regret_cd_ucb1, avg_cum_regret_cd_ucb1 + std_cum_regret_cd_ucb1, alpha=0.2, color='b')
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1"])
plt.title("Cumulative Regret with standard deviation")
plt.show()




# # %%
# plt.figure()
# plt.xlabel("t")
# plt.ylabel("Regret")
# plt.plot(ucb1_instantaneous_regret, 'g')
# plt.hlines(y=0, xmin=0, xmax=T, colors='k', linestyles='dashed')
# #plt.ylim(-0.1, 1)
# plt.legend(["UCB1", "Clairvoyant"])
# plt.title("Instantaneous Regret")
# plt.show()

# # %%
# plt.figure()
# plt.xlabel("t")
# plt.ylabel("Regret")
# plt.plot(ucb1_cumulative_regret, 'g')
# plt.hlines(y=0, xmin=0, xmax=T, colors='k', linestyles='dashed')
# #plt.ylim(-0.1, 1)
# plt.legend(["UCB1", "Clairvoyant"])
# plt.title("Cumulative Regret")
# plt.show()
