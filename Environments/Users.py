from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from param import min_bid, max_bid, min_price, max_price, std_noise_pricing, std_noise_cost, std_noise_click, production_cost
class User(ABC):
    """
    Abstract class used to define User
    :f1: value of binary feature F1 for specific class
    :f2: value of binary feature F2 for specific class
    :probabilities: set of probabilities for each price for the specific class of user
    """
    def __init__(self, f1_value, f2_value, probabilities):
        self._f1_value = f1_value
        self._f2_value = f2_value
        self.production_cost = production_cost
        # Pricing parameters
        self._probabilities = probabilities
        self._curve_params = None
        self._min_price = min_price
        self._max_price = max_price
        self._prices = np.linspace(self._min_price, self._max_price, len(self._probabilities))
        self._std_noise_pricing = std_noise_pricing  # std of the gaussian noise
        self._reward_of_prices = self._prices * self._probabilities
        # Advertising parameters
        self._min_bid = min_bid
        self._max_bid = max_bid
        self._std_noise_cost = std_noise_cost       
        self._std_noise_click = std_noise_click
        
    @property
    def probabilities(self):
        return self._probabilities

    @property
    def prices(self):
        return self._prices
    
    @property
    def max_price(self):
        return self._max_price
    
    @property
    def min_price(self):
        return self._min_price
    
    @property
    def reward_of_prices(self):
        return self._reward_of_prices
    
    ############################
    #      Demand curve        #
    ############################

    def base_function(self, x, a, b, c, d):
        """
        Function used to fit the demand curve
        """
        return a + b * x + c * x**2 + d * x**3 # + np.exp(d * x)

    def fit_demand_curve(self):
        """
        Method used to fit the demand curve for the specific class of user
        """
        x_val = np.linspace(self._min_price, self._max_price, len(self._probabilities)).astype(np.float32)
        y_val = self._probabilities
        params, _ = curve_fit(self.base_function, x_val, y_val)
        self._curve_params = params
        return

    def demand_curve(self, price):
        """
        Method used to evaluate the demand curve for the specific class of user
        """
        if self._curve_params is None:
            self.fit_demand_curve()
        return self.base_function(price, *self._curve_params)
    
    def generate_conversion_rate(self, price):
        """
        Method used to generate the noisy conversion rate for the specific class of user
        """
        return self.demand_curve(price) + np.random.normal(0, self._std_noise_pricing, size = self.demand_curve(price).shape)

    def plot_demand_curve(self):
        """
        Method used to plot the demand curve for the specific class of user
        """
        prices = np.linspace(self._min_price, self._max_price, 100)
        demand = self.demand_curve(prices)
        plt.xlabel('Price')
        plt.ylabel('Demand')
        plt.xlim(self._min_price, self._max_price)
        plt.ylim(0, 1)
        plt.title('Conversion Rate Curve for user class')
        return plt.plot(prices, demand)  # lable=self.__class__.__name__

    def plot_expected_pricing_gain(self):
        """
        Method used to plot the expected reward for the specific class of user
        """
        prices = np.linspace(self._min_price, self._max_price, 100)
        demand = self.demand_curve(prices)
        expected_reward = prices * demand
        plt.xlabel('Price')
        plt.ylabel('Expected reward')
        plt.xlim(self._min_price, self._max_price)
        plt.ylim(0, self._max_price)
        plt.title('Expected reward for prices by user class')
        return plt.plot(prices, expected_reward)
    
    ############################
    #    Advertising curves    # 
    ############################

    @abstractmethod
    def click_vs_bid(self, bid):
        """
        Method used to evaluate the click probability for the specific class of user
        """
        pass 

    @abstractmethod
    def cost_vs_bid(self, bid):
        """
        Method used to evaluate the cost for the specific class of user
        """
        pass

    def generate_click_bid_observations(self, bid):
        """
        Method used to generate noisy observations of click vs bid for the specific class of user
        """
        return self.click_vs_bid(bid) + np.random.normal(0, self._std_noise_click, size = self.click_vs_bid(bid).shape)

    def generate_cost_bid_observations(self, bid):
        """
        Method used to generate noisy observations of click vs bid for the specific class of user
        """
        return self.cost_vs_bid(bid) + np.random.normal(0, self._std_noise_cost, size = self.cost_vs_bid(bid).shape)

    def plot_click_vs_bid(self):
        """
        Method used to plot the click vs bid curve for the specific class of user
        """
        bids = np.linspace(self._min_bid, self._max_bid, 100)
        clicks = self.click_vs_bid(bids)
        plt.xlabel('Bid')
        plt.ylabel('Click')
        plt.xlim(self._min_bid, self._max_bid)
        plt.title('Click vs Bid Curve for user class')
        return plt.plot(bids, clicks)
    
    def plot_cost_vs_bid(self):
        """
        Method used to plot the cost vs bid curve for the specific class of user
        """
        bids = np.linspace(self._min_bid, self._max_bid, 100)
        cost = self.cost_vs_bid(bids)
        plt.xlabel('Bid')
        plt.ylabel('Cost')
        plt.xlim(0, 1)
        # plt.ylim(0, 1)
        plt.title('Cost vs Bid Curve for user class')
        return plt.plot(bids, cost)
    
    def plot_noisy_click_vs_bid(self):
        """
        Method used to plot the click vs bid curve for the specific class of user with noise
        """
        bids = np.linspace(self._min_bid, self._max_bid, 100)
        noisy_clicks = self.generate_click_bid_observations(bids)
        plt.xlabel('Bid')
        plt.ylabel('Click')
        plt.xlim(0, 1)
        # plt.ylim(0, 1)
        plt.title('Noisy Click vs Bid Curve for user class')
        return plt.plot(bids, noisy_clicks, "o")
    
    def plot_noisy_cost_vs_bid(self):
        """
        Method used to plot the cost vs bid curve for the specific class of user with noise
        """
        bids = np.linspace(self._min_bid, self._max_bid, 100)
        noisy_cost = self.generate_cost_bid_observations(bids)
        plt.xlabel('Bid')
        plt.ylabel('Cost')
        plt.xlim(0, 1)
        # plt.ylim(0, 1)
        plt.title('Noisy Cost vs Bid Curve for user class')
        return plt.plot(bids, noisy_cost, "o")
        
    def plot_avg_cumulative_daily_cost_click_bid(self):
        """
        Method used to plot the average cumulative daily cost vs bid curve for the specific class of user
        """
        bids = np.linspace(self._min_bid, self._max_bid, 100)
        avg_cumulative_daily_cost = self.cost_vs_bid(bids) * self.click_vs_bid(bids)
        plt.xlabel('Bid')
        plt.ylabel('Average cumulative daily cost of clicks')
        plt.xlim(0, 1)
        # plt.ylim(0, 1)
        plt.title('Average cumulative daily cost vs Bid Curve for user class')
        return plt.plot(bids, avg_cumulative_daily_cost)

    ############################
    #    Clairvoyant method    #
    ############################
    
    def clairvoyant(self, production_cost):
        best_bid = 0
        best_price = 0
        opt = 0
        bids = np.linspace(self._min_bid, self._max_bid, 100)
        for i in bids:
            for id_j, j in enumerate(self._prices):
                aux = self.click_vs_bid(i) * (self.probabilities[id_j]*(j-production_cost) - self.cost_vs_bid(i))
                if (aux > opt):
                    best_bid = i
                    best_price = j
                    opt = aux
        return [best_price, best_bid, opt]
    
    def general_reward(self, price, bid, production_cost):
        return self.click_vs_bid(bid) * (self.demand_curve(price)*(price-production_cost) - self.cost_vs_bid(bid))
    
    def plot_general_reward(self):
        fig = plt.figure(facecolor='white')
        prices = np.linspace(self._min_price, self._max_price, 200)
        bids = np.linspace(self._min_bid, self._max_bid, 200)
        reward = np.zeros((len(prices), len(bids)))
        for id_i, i in enumerate(prices):
            for id_j, j in enumerate(bids):
                reward[id_i, id_j] = self.general_reward(i, j, self.production_cost)
        plt.xlabel('Price')
        plt.ylabel('Bid')
        plt.xlim(self._min_price, self._max_price)
        plt.ylim(self._min_bid, self._max_bid)
        plt.title('General reward for user class')
        plt.contourf(prices, bids, reward, 100)
        plt.colorbar()
        return fig

class UserC1(User):
    """
    User class C1
    Collector User
    """

    def __init__(self):
        super().__init__(True, True, np.array([0.2, 0.4, 0.4, 0.1, 0.05]))

    def click_vs_bid(self, bid):
        bid = (bid - self._min_bid) / (self._max_bid - self._min_bid)
        return (1 - np.exp(- 8.0 * bid - 0.5 * bid**2)) * 70 # return number of clicks
        
    def cost_vs_bid(self, bid):
        bid = (bid - self._min_bid) / (self._max_bid - self._min_bid)
        return 1.4*np.log(1+bid/1.7) * (self._max_bid - self._min_bid) + self._min_bid  # return cost of the click

    

class UserC2(User):
    """
    User class C2
    Parent User
    """
    def __init__(self):
        probabilities = np.array([0.4, 0.35, 0.3, 0.15, 0.1])
        super().__init__(True, False, probabilities)

    def click_vs_bid(self, bid):
        bid = (bid - self._min_bid) / (self._max_bid - self._min_bid)
        return (1 - np.exp(-1.5 * bid + -1 * bid**2)) * 100

    def cost_vs_bid(self, bid):
        bid = (bid - self._min_bid) / (self._max_bid - self._min_bid)
        return 1.9*np.log(1+bid/1.9) * (self._max_bid - self._min_bid) + self._min_bid 
    

class UserC3(User):
    """
    User class C3
    Young User
    """
    def __init__(self):
        feature1 = False
        feature2 = np.random.choice([True, False])
        probabilities = np.array([0.4, 0.35, 0.3, 0.1, 0.05])
        super().__init__(feature1, feature2, probabilities)
    
    def click_vs_bid(self, bid):
        bid = (bid - self._min_bid) / (self._max_bid - self._min_bid)
        return (1 - np.exp(-1 * bid -0.9 * bid**2)) * 90 

    def cost_vs_bid(self, bid):
        bid = (bid - self._min_bid) / (self._max_bid - self._min_bid)
        return 1.7*np.log(1+bid/1.8) * (self._max_bid - self._min_bid) + self._min_bid 
    