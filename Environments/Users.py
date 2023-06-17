from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
        self._probabilities = probabilities
        self._curve_params = None
        self._min_price = 150
        self._max_price = 350
        self._prices = np.linspace(self._min_price, self._max_price, len(self._probabilities))

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

    def concave_function(self, x, a, b, c, d):
        """
        Function used to fit the demand curve
        """
        return a + b * x + c * x**2

    def fit_demand_curve(self):
        """
        Method used to fit the demand curve for the specific class of user
        """
        x_val = np.linspace(self._min_price, self._max_price, len(self._probabilities)).astype(np.float32)
        y_val = self._probabilities
        params, _ = curve_fit(self.concave_function, x_val, y_val)
        self._curve_params = params
        return


    def demand_curve(self, price):
        """
        Method used to evaluate the demand curve for the specific class of user
        """
        if self._curve_params is None:
            self.fit_demand_curve()
        return self.concave_function(price, *self._curve_params)

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

    # def best_price(self):
    #     """
    #     Return the best price fot the specific class of user
    #     """
    #     prices = np.linspace(self._min_price, self._max_price, self._max_price - self._min_price + 1)
    #     demand = self.demand_curve(prices)
    #     return prices[np.argmax(demand)]

    def plot_expected_reward(self):
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
        plt.title('Expected reward for user class')
        return plt.plot(prices, expected_reward)



class UserC1(User):
    """
    User class C1
    Collcetor User
    """

    def __init__(self):
        probabilities = np.array([0.3, 0.5, 0.85, 0.8, 0.7])
        super().__init__(True, True, probabilities)


class UserC2(User):
    """
    User class C2
    Parent User
    """
    def __init__(self):
        probabilities = np.array([0.3, 0.5, 0.7, 0.85, 0.8])
        super().__init__(True, False, probabilities)


class UserC3(User):
    """
    User class C3
    Young User
    """
    def __init__(self):
        feature1 = False
        feature2 = np.random.choice([True, False])
        probabilities = np.array([0.7, 0.8, 0.7, 0.5, 0.3])
        super().__init__(feature1, feature2, probabilities)
