import numpy as np

##################################################
#                Pricing parameters              #
##################################################

min_price = 150
max_price = 250
std_noise_pricing = 10

n_arms_pricing = 5

##################################################
#             Advertising parameters             #
##################################################

min_bid = 0.01
max_bid = 5.0
std_noise_cost = (max_bid - min_bid) * 0.05
std_noise_click = 10

n_arms_bidding = 100

##################################################
#              Experiment parameters             #
##################################################

T = 365
n_experiments_S1 = 10
n_experiments_S2 = 5
n_experiments_S3 = 5

production_cost = 100

std_noise_general = 10
sigma_gp = 5

##################################################
#                  Users parameters              #
##################################################

p1_stationary = np.array([0.05, 0.1, 0.1, 0.025, 0.0125])
p2_stationary = np.array([0.15, 0.15, 0.1, 0.075, 0.05])
p3_stationary = np.array([0.14, 0.12, 0.10, 0.03, 0.02])
max_conversion_rate = 0.4       # for plots