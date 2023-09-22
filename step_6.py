# %%
import numpy as np
import math
import matplotlib.pyplot as plt

from Environments.Environment_S6 import Non_Stationary_Environment
from Environments.Users import UserC1

from Learners.UCB1_Learner import UCB1_Learner
from Learners.SW_UCB1_Learner import SW_UCB1_Learner
from Learners.CUSUM_UCB1_Learner import CUSUM_UCB1_Learner
from Learners.EXP3_Learner import EXP3_Learner

from param import T, n_arms_pricing, n_arms_pricing, p1_step_6, n_experiments_S6
from param import production_cost, selected_bid_S6, p1_non_stationary
from param import N, eps, threshold, alpha
from param import Tau


# %% Test EXP3 with 3 changes
##################################################
#                 EXP3 3 changes                 #
##################################################


Collector = UserC1(True, True, p1_non_stationary)

n_phases = len(p1_non_stationary)
phases_len = np.ceil(T / n_phases)

optimum = Collector.clairvoyant()

opt = np.array([])
for i in range(0, n_phases):
    opt = np.append(opt, optimum[i][2])


opt_vec = np.array([])
for i in range(0, T):
    current_phase = int(i / phases_len)
    opt_vec = np.append(opt_vec, opt[current_phase])

n_clicks = Collector.click_vs_bid(selected_bid_S6)
cost_of_click = Collector.cost_vs_bid(selected_bid_S6)

ucb1_rewards_per_experiment = []
sw_ucb1_rewards_per_experiment = []
cd_ucb1_rewards_per_experiment = []
exp3_rewards_per_experiment = []

# %% Optimization of gamma in EXP3
gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1]

Collector_S5 = UserC1(True, True, p1_non_stationary)

avg_regret_exp3 = []
std_regret_exp3 = []
cum_avg_regret_exp3 = []
cum_std_regret_exp3 = []


for g in gamma:
    exp3_rewards_per_experiment = []
    for e in range(0, n_experiments_S6):
        ns_env = Non_Stationary_Environment(
            selected_bid=selected_bid_S6,
            production_cost=production_cost,
            n_arms=n_arms_pricing,
            user=Collector_S5,
            horizon=T,
        )

        exp3_learner = EXP3_Learner(
            n_arms=n_arms_pricing,
            gamma=g,
            clicks=n_clicks,
            cost=cost_of_click,
            production_cost=production_cost,
        )

        for t in range(0, T):
            # EXP3 Learner:
            pulled_arm = exp3_learner.pull_arm()  # determino l'arm da pullare
            reward = ns_env.round(pulled_arm)  # osservo il feedback corrispondente
            exp3_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        exp3_rewards_per_experiment.append(exp3_learner.collected_rewards)

    regret_exp3 = [opt_vec - result for result in exp3_rewards_per_experiment]
    avg_regret_exp3.append(np.mean(regret_exp3, axis=0))
    std_regret_exp3.append(np.std(regret_exp3, axis=0))
    cum_avg_regret_exp3.append(np.mean(np.cumsum(regret_exp3, axis=1), axis=0))
    cum_std_regret_exp3.append(np.std(np.cumsum(regret_exp3, axis=1), axis=0))


# %% Plot the cumulative regret for different values of gamma
fig = plt.figure(0, facecolor="white")
plt.xlabel("t")
plt.ylabel("Regret")
for i in range(len(gamma)):
    plt.plot(cum_avg_regret_exp3[i])
for i in range(len(gamma)):
    plt.fill_between(
        range(len(cum_avg_regret_exp3[i])),
        cum_avg_regret_exp3[i] - cum_std_regret_exp3[i],
        cum_avg_regret_exp3[i] + cum_std_regret_exp3[i],
        alpha=0.2,
    )
plt.legend(gamma)

plt.title("Cumulative Regret by Gamma")
fig = plt.gcf()
plt.show()

fig.savefig("results/S6_3_phases_cumulative_regret_by_gamma.png")

# %%
##################################################
#           EXP3 3 changes comparison            #
##################################################
gamma = 0.01

ucb1_rewards_per_experiment = []
sw_ucb1_rewards_per_experiment = []
cd_ucb1_rewards_per_experiment = []
exp3_rewards_per_experiment = []

for e in range(0, n_experiments_S6):
    ns_env_1 = Non_Stationary_Environment(
        selected_bid_S6, production_cost, n_arms_pricing, Collector, T
    )
    ns_env_2 = Non_Stationary_Environment(
        selected_bid_S6, production_cost, n_arms_pricing, Collector, T
    )
    ns_env_3 = Non_Stationary_Environment(
        selected_bid_S6, production_cost, n_arms_pricing, Collector, T
    )
    ns_env_4 = Non_Stationary_Environment(
        selected_bid_S6, production_cost, n_arms_pricing, Collector, T
    )

    ucb1_learner = UCB1_Learner(
        n_arms_pricing, production_cost, n_clicks, cost_of_click, 600
    )
    sw_ucb1_learner = SW_UCB1_Learner(
        n_arms_pricing,
        production_cost,
        n_clicks,
        cost_of_click,
        Collector._max_price,
        Tau,
    )
    cd_ucb1_learner = CUSUM_UCB1_Learner(
        n_arms_pricing,
        production_cost,
        n_clicks,
        cost_of_click,
        Collector._max_price,
        N,
        eps,
        threshold,
        alpha,
    )
    exp3_learner = EXP3_Learner(
        n_arms_pricing, gamma, n_clicks, cost_of_click, production_cost
    )

    for t in range(0, T):
        # UCB1 Learner:
        pulled_arm = ucb1_learner.pull_arm()  # determino l'arm da pullare
        reward = ns_env_1.round(pulled_arm)  # osservo il feedback corrispondente
        ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        # SW_UCB1 Learner:
        pulled_arm = sw_ucb1_learner.pull_arm()
        reward = ns_env_2.round(pulled_arm)
        sw_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        # CD_UCB1 Learner:
        pulled_arm = cd_ucb1_learner.pull_arm()
        reward = ns_env_3.round(pulled_arm)
        cd_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        # EXP3 Learner:
        pulled_arm = exp3_learner.pull_arm()  # determino l'arm da pullare
        reward = ns_env_4.round(pulled_arm)  # osservo il feedback corrispondente
        exp3_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    sw_ucb1_rewards_per_experiment.append(sw_ucb1_learner.collected_rewards)
    cd_ucb1_rewards_per_experiment.append(cd_ucb1_learner.collected_rewards)
    exp3_rewards_per_experiment.append(exp3_learner.collected_rewards)

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

# exp3 regret
regret_exp3 = [opt_vec - result for result in exp3_rewards_per_experiment]
avg_regret_exp3 = np.mean(regret_exp3, axis=0)
std_regret_exp3 = np.std(regret_exp3, axis=0)
cum_avg_regret_exp3 = np.mean(np.cumsum(regret_exp3, axis=1), axis=0)
cum_std_regret_exp3 = np.std(np.cumsum(regret_exp3, axis=1), axis=0)


# %% Plot the cumulative regret
fig = plt.figure(0, facecolor="white")
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(cum_avg_regret_ucb1, "b")
plt.plot(cum_avg_regret_sw_ucb1, "cyan")
plt.plot(cum_avg_regret_cd_ucb1, "r")
plt.plot(cum_avg_regret_exp3, "g")
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
    color="cyan",
)
plt.fill_between(
    range(len(cum_avg_regret_cd_ucb1)),
    cum_avg_regret_cd_ucb1 - cum_std_regret_cd_ucb1,
    cum_avg_regret_cd_ucb1 + cum_std_regret_cd_ucb1,
    alpha=0.2,
    color="r",
)
plt.fill_between(
    range(len(cum_avg_regret_exp3)),
    cum_avg_regret_exp3 - cum_std_regret_exp3,
    cum_avg_regret_exp3 + cum_std_regret_exp3,
    alpha=0.2,
    color="g",
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1", "EXP3"])
plt.title("Cumulative Regret")
fig = plt.gcf()
plt.show()

fig.savefig("results/S6_3_phases_cumulative_regret.png")

# %% Plot the instantaneous regret with standard deviation
fig = plt.figure(1, facecolor="white")
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_ucb1, "b")
plt.plot(avg_regret_sw_ucb1, "cyan")
plt.plot(avg_regret_cd_ucb1, "r")
plt.plot(avg_regret_exp3, "g")
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
    color="cyan",
)
plt.fill_between(
    range(len(avg_regret_cd_ucb1)),
    avg_regret_cd_ucb1 - std_regret_cd_ucb1,
    avg_regret_cd_ucb1 + std_regret_cd_ucb1,
    alpha=0.2,
    color="r",
)
plt.fill_between(
    range(len(avg_regret_exp3)),
    avg_regret_exp3 - std_regret_exp3,
    avg_regret_exp3 + std_regret_exp3,
    alpha=0.2,
    color="g",
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1", "EXP3"])
plt.title("Instantaneous Regret with Standard Deviation")
fig = plt.gcf()
plt.show()

fig.savefig("results/S6_3_phases_instantaneous_regret.png")

# %% Compute the reward
avg_reward_ucb1 = np.mean(ucb1_rewards_per_experiment, axis=0)
std_reward_ucb1 = np.std(ucb1_rewards_per_experiment, axis=0)
cum_avg_reward_ucb1 = np.mean(np.cumsum(ucb1_rewards_per_experiment, axis=1), axis=0)
cum_std_reward_ucb1 = np.std(np.cumsum(ucb1_rewards_per_experiment, axis=1), axis=0)

avg_reward_sw_ucb1 = np.mean(sw_ucb1_rewards_per_experiment, axis=0)
std_reward_sw_ucb1 = np.std(sw_ucb1_rewards_per_experiment, axis=0)
cum_avg_reward_sw_ucb1 = np.mean(
    np.cumsum(sw_ucb1_rewards_per_experiment, axis=1), axis=0
)
cum_std_rreward_sw_ucb1 = np.std(
    np.cumsum(sw_ucb1_rewards_per_experiment, axis=1), axis=0
)

avg_reward_cd_ucb1 = np.mean(cd_ucb1_rewards_per_experiment, axis=0)
std_reward_cd_ucb1 = np.std(cd_ucb1_rewards_per_experiment, axis=0)
cum_avg_reward_cd_ucb1 = np.mean(
    np.cumsum(cd_ucb1_rewards_per_experiment, axis=1), axis=0
)
cum_std_rreward_cd_ucb1 = np.std(
    np.cumsum(cd_ucb1_rewards_per_experiment, axis=1), axis=0
)

avg_reward_exp3 = np.mean(exp3_rewards_per_experiment, axis=0)
std_reward_exp3 = np.std(exp3_rewards_per_experiment, axis=0)
cum_avg_reward_exp3 = np.mean(np.cumsum(exp3_rewards_per_experiment, axis=1), axis=0)
cum_std_rreward_exp3 = np.std(np.cumsum(exp3_rewards_per_experiment, axis=1), axis=0)


# %% Plot the cumulative reward
plt.figure(2, facecolor="white")
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(cum_avg_reward_ucb1, "b")
plt.plot(cum_avg_reward_sw_ucb1, "cyan")
plt.plot(cum_avg_reward_cd_ucb1, "r")
plt.plot(cum_avg_reward_exp3, "g")
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
    color="cyan",
)
plt.fill_between(
    range(len(cum_avg_reward_cd_ucb1)),
    cum_avg_reward_cd_ucb1 - cum_std_rreward_cd_ucb1,
    cum_avg_reward_cd_ucb1 + cum_std_rreward_cd_ucb1,
    alpha=0.2,
    color="r",
)
plt.fill_between(
    range(len(cum_avg_reward_exp3)),
    cum_avg_reward_exp3 - cum_std_rreward_exp3,
    cum_avg_reward_exp3 + cum_std_rreward_exp3,
    alpha=0.2,
    color="g",
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1", "EXP3"])
plt.title("Cumulative Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S6_3_phases_cumulative_reward.png")

# %% Plot the instantaneous reward
plt.figure(3, facecolor="white")
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(avg_reward_ucb1, "b")
plt.plot(avg_reward_sw_ucb1, "cyan")
plt.plot(avg_reward_cd_ucb1, "r")
plt.plot(avg_reward_exp3, "g")
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
    color="cyan",
)
plt.fill_between(
    range(len(avg_reward_cd_ucb1)),
    avg_reward_cd_ucb1 - std_reward_cd_ucb1,
    avg_reward_cd_ucb1 + std_reward_cd_ucb1,
    alpha=0.2,
    color="r",
)
plt.fill_between(
    range(len(avg_reward_exp3)),
    avg_reward_exp3 - std_reward_exp3,
    avg_reward_exp3 + std_reward_exp3,
    alpha=0.2,
    color="g",
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1", "EXP3"])
plt.title("Instantaneous Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S6_3_phases_instantaneous_reward.png")


# %% Test EXP3 with many changes
##################################################
#               EXP3 many changes                #
##################################################
Collector = UserC1(True, True, p1_step_6)

n_phases = len(p1_step_6)
phases_len = np.ceil(T / n_phases)

optimum = Collector.clairvoyant()

opt = np.array([])
for i in range(0, n_phases):
    opt = np.append(opt, optimum[i][2])


opt_vec = np.array([])
for i in range(0, T):
    current_phase = int(i / phases_len)
    opt_vec = np.append(opt_vec, opt[current_phase])

n_clicks = Collector.click_vs_bid(selected_bid_S6)
cost_of_click = Collector.cost_vs_bid(selected_bid_S6)


# %% Plot the demand curves and the expected reward curves for each phase:
fig = plt.figure()
Collector_aux = UserC1(True, True, p1_step_6[:5])
Collector_aux.plot_demand_curve()
plt.legend(["phase 1", "phase 2", "phase 3", "phase 4", "phase 5"])
plt.title("Conversion Rate Curves")
plt.show()


# %% Run the experiments
gamma = 0.005

ucb1_rewards_per_experiment = []
sw_ucb1_rewards_per_experiment = []
cd_ucb1_rewards_per_experiment = []
exp3_rewards_per_experiment = []

for e in range(0, n_experiments_S6):
    ns_env_1 = Non_Stationary_Environment(
        selected_bid_S6, production_cost, n_arms_pricing, Collector, T
    )
    ns_env_2 = Non_Stationary_Environment(
        selected_bid_S6, production_cost, n_arms_pricing, Collector, T
    )
    ns_env_3 = Non_Stationary_Environment(
        selected_bid_S6, production_cost, n_arms_pricing, Collector, T
    )
    ns_env_4 = Non_Stationary_Environment(
        selected_bid_S6, production_cost, n_arms_pricing, Collector, T
    )

    ucb1_learner = UCB1_Learner(
        n_arms_pricing, production_cost, n_clicks, cost_of_click, 600
    )
    sw_ucb1_learner = SW_UCB1_Learner(
        n_arms_pricing,
        production_cost,
        n_clicks,
        cost_of_click,
        Collector._max_price,
        Tau,
    )
    cd_ucb1_learner = CUSUM_UCB1_Learner(
        n_arms_pricing,
        production_cost,
        n_clicks,
        cost_of_click,
        Collector._max_price,
        N,
        eps,
        threshold,
        alpha,
    )
    exp3_learner = EXP3_Learner(
        n_arms_pricing, gamma, n_clicks, cost_of_click, production_cost
    )

    for t in range(0, T):
        # UCB1 Learner:
        pulled_arm = ucb1_learner.pull_arm()  # determino l'arm da pullare
        reward = ns_env_1.round(pulled_arm)  # osservo il feedback corrispondente
        ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        # SW_UCB1 Learner:
        pulled_arm = sw_ucb1_learner.pull_arm()
        reward = ns_env_2.round(pulled_arm)
        sw_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        # CD_UCB1 Learner:
        pulled_arm = cd_ucb1_learner.pull_arm()
        reward = ns_env_3.round(pulled_arm)
        cd_ucb1_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

        # EXP3 Learner:
        pulled_arm = exp3_learner.pull_arm()  # determino l'arm da pullare
        reward = ns_env_4.round(pulled_arm)  # osservo il feedback corrispondente
        exp3_learner.update(pulled_arm, reward, Collector.prices[pulled_arm])

    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    sw_ucb1_rewards_per_experiment.append(sw_ucb1_learner.collected_rewards)
    cd_ucb1_rewards_per_experiment.append(cd_ucb1_learner.collected_rewards)
    exp3_rewards_per_experiment.append(exp3_learner.collected_rewards)

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

# exp3 regret
regret_exp3 = [opt_vec - result for result in exp3_rewards_per_experiment]
avg_regret_exp3 = np.mean(regret_exp3, axis=0)
std_regret_exp3 = np.std(regret_exp3, axis=0)
cum_avg_regret_exp3 = np.mean(np.cumsum(regret_exp3, axis=1), axis=0)
cum_std_regret_exp3 = np.std(np.cumsum(regret_exp3, axis=1), axis=0)


# %% Plot the cumulative regret
fig = plt.figure(0, facecolor="white")
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(cum_avg_regret_ucb1, "b")
plt.plot(cum_avg_regret_sw_ucb1, "cyan")
plt.plot(cum_avg_regret_cd_ucb1, "r")
plt.plot(cum_avg_regret_exp3, "g")
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
    color="cyan",
)
plt.fill_between(
    range(len(cum_avg_regret_cd_ucb1)),
    cum_avg_regret_cd_ucb1 - cum_std_regret_cd_ucb1,
    cum_avg_regret_cd_ucb1 + cum_std_regret_cd_ucb1,
    alpha=0.2,
    color="r",
)
plt.fill_between(
    range(len(cum_avg_regret_exp3)),
    cum_avg_regret_exp3 - cum_std_regret_exp3,
    cum_avg_regret_exp3 + cum_std_regret_exp3,
    alpha=0.2,
    color="g",
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1", "EXP3"])
plt.title("Cumulative Regret")
fig = plt.gcf()
plt.show()

fig.savefig("results/S6_cumulative_regret.png")

# %% Plot the instantaneous regret with standard deviation
fig = plt.figure(1, facecolor="white")
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(avg_regret_ucb1, "b")
plt.plot(avg_regret_sw_ucb1, "cyan")
plt.plot(avg_regret_cd_ucb1, "r")
plt.plot(avg_regret_exp3, "g")
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
    color="cyan",
)
plt.fill_between(
    range(len(avg_regret_cd_ucb1)),
    avg_regret_cd_ucb1 - std_regret_cd_ucb1,
    avg_regret_cd_ucb1 + std_regret_cd_ucb1,
    alpha=0.2,
    color="r",
)
plt.fill_between(
    range(len(avg_regret_exp3)),
    avg_regret_exp3 - std_regret_exp3,
    avg_regret_exp3 + std_regret_exp3,
    alpha=0.2,
    color="g",
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1", "EXP3"])
plt.title("Instantaneous Regret with Standard Deviation")
fig = plt.gcf()
plt.show()

fig.savefig("results/S6_instantaneous_regret.png")

# %% Compute the reward
avg_reward_ucb1 = np.mean(ucb1_rewards_per_experiment, axis=0)
std_reward_ucb1 = np.std(ucb1_rewards_per_experiment, axis=0)
cum_avg_reward_ucb1 = np.mean(np.cumsum(ucb1_rewards_per_experiment, axis=1), axis=0)
cum_std_reward_ucb1 = np.std(np.cumsum(ucb1_rewards_per_experiment, axis=1), axis=0)

avg_reward_sw_ucb1 = np.mean(sw_ucb1_rewards_per_experiment, axis=0)
std_reward_sw_ucb1 = np.std(sw_ucb1_rewards_per_experiment, axis=0)
cum_avg_reward_sw_ucb1 = np.mean(
    np.cumsum(sw_ucb1_rewards_per_experiment, axis=1), axis=0
)
cum_std_rreward_sw_ucb1 = np.std(
    np.cumsum(sw_ucb1_rewards_per_experiment, axis=1), axis=0
)

avg_reward_cd_ucb1 = np.mean(cd_ucb1_rewards_per_experiment, axis=0)
std_reward_cd_ucb1 = np.std(cd_ucb1_rewards_per_experiment, axis=0)
cum_avg_reward_cd_ucb1 = np.mean(
    np.cumsum(cd_ucb1_rewards_per_experiment, axis=1), axis=0
)
cum_std_rreward_cd_ucb1 = np.std(
    np.cumsum(cd_ucb1_rewards_per_experiment, axis=1), axis=0
)

avg_reward_exp3 = np.mean(exp3_rewards_per_experiment, axis=0)
std_reward_exp3 = np.std(exp3_rewards_per_experiment, axis=0)
cum_avg_reward_exp3 = np.mean(np.cumsum(exp3_rewards_per_experiment, axis=1), axis=0)
cum_std_rreward_exp3 = np.std(np.cumsum(exp3_rewards_per_experiment, axis=1), axis=0)


# %% Plot the cumulative reward
plt.figure(2, facecolor="white")
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(cum_avg_reward_ucb1, "b")
plt.plot(cum_avg_reward_sw_ucb1, "cyan")
plt.plot(cum_avg_reward_cd_ucb1, "r")
plt.plot(cum_avg_reward_exp3, "g")
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
    color="cyan",
)
plt.fill_between(
    range(len(cum_avg_reward_cd_ucb1)),
    cum_avg_reward_cd_ucb1 - cum_std_rreward_cd_ucb1,
    cum_avg_reward_cd_ucb1 + cum_std_rreward_cd_ucb1,
    alpha=0.2,
    color="r",
)
plt.fill_between(
    range(len(cum_avg_reward_exp3)),
    cum_avg_reward_exp3 - cum_std_rreward_exp3,
    cum_avg_reward_exp3 + cum_std_rreward_exp3,
    alpha=0.2,
    color="g",
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1", "EXP3"])
plt.title("Cumulative Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S6_cumulative_reward.png")

# %% Plot the instantaneous reward
plt.figure(3, facecolor="white")
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(avg_reward_ucb1, "b")
plt.plot(avg_reward_sw_ucb1, "cyan")
plt.plot(avg_reward_cd_ucb1, "r")
plt.plot(avg_reward_exp3, "g")
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
    color="cyan",
)
plt.fill_between(
    range(len(avg_reward_cd_ucb1)),
    avg_reward_cd_ucb1 - std_reward_cd_ucb1,
    avg_reward_cd_ucb1 + std_reward_cd_ucb1,
    alpha=0.2,
    color="r",
)
plt.fill_between(
    range(len(avg_reward_exp3)),
    avg_reward_exp3 - std_reward_exp3,
    avg_reward_exp3 + std_reward_exp3,
    alpha=0.2,
    color="g",
)
plt.legend(["UCB1", "SW_UCB1", "CUSUM_UCB1", "EXP3"])
plt.title("Instantaneous Reward")
fig = plt.gcf()
plt.show()

fig.savefig("results/S6_instantaneous_reward.png")

# %%
