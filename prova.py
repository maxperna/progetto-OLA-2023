import numpy as np

p = np.array([[0.9, 0.8, 0.5, 0.3, 0.2], [0.2, 0.3, 0.5, 0.8, 0.9], [0.3, 0.6, 0.9, 0.6, 0.3]])
p1 = np.array([0.05, 0.1, 0.1, 0.025, 0.0125])
prices = [150, 160, 170, 180, 190]

print(prices * p)
print(p * prices)

print(p.ndim)
print(p1.ndim)