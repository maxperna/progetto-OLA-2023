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
plt.show()

# %% Plot the expected rewards for the three classes of users
fig = plt.figure()
Collector.plot_expected_reward()
Parent.plot_expected_reward()
Young.plot_expected_reward()
plt.legend(['Collector', 'Parent', 'Young'])
plt.title("Expected Rewards Curves")
plt.show()

# %%
