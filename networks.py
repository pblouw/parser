import numpy as np
from helpers import *


class Hopfield(object):
    """
    A network that encodes the constraints for a local tree described by a set
    of symbolic rewrite rules. Eventually, this will be a real Hopfield 
    network that automatically seeks out an optimal state given a starting 
    state. Right now, the network simply acts to evaluate the optimality of 
    some state relative to some set of constraints.   
    """
    def __init__(self, label, bias):
        self.label = label
        self.constraints = None
        self.weights = None 
        self.clean_in = None
        self.clean_out = None
        self.state = bias
        self.vocab = []

    def threshold(self, x, size):
        """"Non-linearity for implementing the network's cleanup memory"""
        if size == 3:
            if x[1] > 0.3:
                x[1] = 1
            else: 
                x[1] = 0
        x[x > 0.15] = 1
        x[x <= 0.15] = 0
        return x

    def compute_energy(self):
        """Evaluates to the degree to which constraints are satisfied """ 
        Wa = np.dot(self.weights, self.state)
        C = self.threshold(np.dot(self.clean_in, Wa), len(self.vocab))
        CWa = np.dot(self.clean_out, C)
        aCWa = np.dot(np.transpose(self.state), normalize(CWa))
        return aCWa

    def update(self):
        """Todo - currently no implementation of energy minimization"""
        pass