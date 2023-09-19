"""
In this script, we will describe the environment and the clients in the project.
"""
# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt

from Environments.Users import  UserC1, UserC2, UserC3

from param import production_cost

# %% Create the three classes of users
Collector = UserC1()
Parent = UserC2()
Young = UserC3()

# %% Plot the demand curves for the three classes of users
fig = plt.figure(facecolor='white')
Collector.plot_demand_curve()
Parent.plot_demand_curve()
Young.plot_demand_curve()
plt.legend(['Collector', 'Parent', 'Young'])
plt.title("Conversion Rate Curves")
fig = plt.gcf()
plt.show()

fig.savefig("results/S0_conversion_rates.png")

# %% Plot the expected rewards for the three classes of users
fig = plt.figure(facecolor='white')
Collector.plot_expected_pricing_gain()
Parent.plot_expected_pricing_gain()
Young.plot_expected_pricing_gain()
plt.legend(['Collector', 'Parent', 'Young'])
plt.title("Expected Rewards Curves")
fig = plt.gcf()
plt.show()

fig.savefig("results/S0_expected_gain.png")

# %% Plot the click vs bid curves for the three classes of users
fig = plt.figure(facecolor='white')
Collector.plot_click_vs_bid()
Parent.plot_click_vs_bid()
Young.plot_click_vs_bid()
plt.legend(['Collector', 'Parent', 'Young'])
plt.title("Click vs Bid Curves")
fig = plt.gcf()
plt.show()

fig.savefig("results/S0_click_vs_bid.png")

# %% Plot the cumulative cost vs bid curves for the three classes of users
fig = plt.figure(facecolor='white')
Collector.plot_cost_vs_bid()
Parent.plot_cost_vs_bid()
Young.plot_cost_vs_bid()
# plt.plot([0, 1], [0, 1], 'k--')  # Check that the cumulative cost is always lower or equal than the bid (second price auction)
plt.legend(['Collector', 'Parent', 'Young'])
plt.title("Cost vs Bid Curves")
fig = plt.gcf()
plt.show()

fig.savefig("results/S0_cost_vs_bid.png")

# %% Plot the cumulative cost vs bid curves for the three classes of users
fig = plt.figure(facecolor='white')
Collector.plot_avg_cumulative_daily_cost_click_bid()
Parent.plot_avg_cumulative_daily_cost_click_bid()
Young.plot_avg_cumulative_daily_cost_click_bid()
plt.legend(['Collector', 'Parent', 'Young'])
plt.title("Average Cumulative daily Cost of Clicks vs Bid Curves")
fig = plt.gcf()
plt.show()

fig.savefig("results/S0_avg_cumulative_cost_vs_bid.png")


# %% Clairvoyant reward
Collector.plot_general_reward()
fig = plt.gcf()
fig.savefig("results/S0_clairvoyant_reward_Collector.png")

# %%
Parent.plot_general_reward()
fig = plt.gcf()
fig.savefig("results/S0_clairvoyant_reward_Parent.png")

# %%
fig = Young.plot_general_reward()
fig = plt.gcf()
fig.savefig("results/S0_clairvoyant_reward_Young.png")

# %%
