import numpy as np

class Product():
    """
    Class defining product sold
    :name: name of the product
    :price: price of the product
    :number: quantity of product in stock
    """
    def __init__(self,name,price,number):
        self.price = price
        self.name = name
        self.number = number