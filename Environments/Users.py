from abc import ABC
import numpy as np


class User(ABC):
    """
    Abstract class used to define User
    :f1: value of binary feature F1 for specific class
    :f2: value of binary feature F2 for specific class
    :prices: set of available prices for class of user
    """
    def __init__(self, f1_value, f2_value, probabilities):
        self.f1_value = f1_value
        self.f2_value = f2_value
        self.probabilities = probabilities


class UserC1(User):
    """
    User class C1
    Frequent User
    """

    def __init__(self):
        probabilities = np.random.rand(5)
        probabilities = np.sort(probabilities)[::-1]
        super().__init__(True, True, probabilities)


class UserC2(User):
    """
    User class C2
    Non Member User
    """
    def __init__(self):
        super().__init__(False, False)


class UserC3(User):
    """
    User class C3
    Occasional User
    """
    def __init__(self):
        feature1 = np.random.choice([True, False])
        feature2 = not feature1
        super().__init__(feature1, feature2)
