"""
In this script, we will describe the environment and the clients in the project.
"""
# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt

from Environments.Users import  UserC1, UserC2, UserC3

# %% Create the three classes of users
Collector = UserC1()
Parent = UserC2()
Young = UserC3()

# %% Plot the demand curves for the three classes of users
fig = plt.figure()
Collector.plot_demand_curve()
Parent.plot_demand_curve()
Young.plot_demand_curve()
plt.legend(['Collector', 'Parent', 'Young'])
plt.title("Conversion Rate Curves")
fig = plt.gcf()
plt.show()

fig.savefig("results/S0_conversion_rates.png")

# %% Plot the expected rewards for the three classes of users
fig = plt.figure()
Collector.plot_expected_reward()
Parent.plot_expected_reward()
Young.plot_expected_reward()
plt.legend(['Collector', 'Parent', 'Young'])
plt.title("Expected Rewards Curves")
fig = plt.gcf()
plt.show()

fig.savefig("results/S0_expected_gain.png")

# %% Plot the click vs bid curves for the three classes of users
fig = plt.figure()
Collector.plot_click_vs_bid()
Parent.plot_click_vs_bid()
Young.plot_click_vs_bid()
plt.legend(['Collector', 'Parent', 'Young'])
plt.title("Click vs Bid Curves")
fig = plt.gcf()
plt.show()

fig.savefig("results/S0_click_vs_bid.png")

# %% Plot the cumulative cost vs bid curves for the three classes of users
fig = plt.figure()
Collector.plot_cost_vs_bid()
Parent.plot_cost_vs_bid()
Young.plot_cost_vs_bid()
# plt.plot([0, 1], [0, 1], 'k--')  # Check that the cumulative cost is always lower or equal than the bid (second price auction)
plt.legend(['Collector', 'Parent', 'Young'])
plt.title("Cost vs Bid Curves")
fig = plt.gcf()
plt.show()

fig.savefig("results/S0_cost_vs_bid.png")

# %%
