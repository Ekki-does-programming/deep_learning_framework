import numpy as np
import matplotlib.pyplot as plt

# Base Layer
class BaseLayer:
    # this class is a parent class for all layers

    def __init__(self):
        self.trainable = False
        self.weights = None
        self.testing_phase = None
