import copy
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
    def __init__(self):
        self.weights = None 
        self.clean_in = None
        self.clean_out = None
        self.bindings = []

    def threshold(self, x, size):
        """"Non-linearity for implementing the network's cleanup memory"""
        # if size >= 4:
        #     if x[1] > 0.7:
        #         x[1] = 1
        #     else: 
        #         x[1] = 0
        #     if x[2] > 0.35:
        #         x[2] = 1
        #     else:
        #         x[2] = 0
        # elif size >= 3:
        #     if x[1] > 0.35:
        #         x[1] = 1
        #     else: 
        #         x[1] = 0
        x[x > 0.2] = 1
        x[x <= 0.2] = 0
        return x

    def compute_energy(self):
        """Evaluates to the degree to which constraints are satisfied """
        # print ''
        # print self.label 
        Wa = np.dot(self.weights, self.state)
        # print 'Raw values: ', np.dot(self.clean_in, Wa)
        C = self.threshold(np.dot(self.clean_in, Wa), len(self.bindings))
        # print 'Thresholding: ', C
        CWa = np.dot(self.clean_out, C)
        aCWa = np.dot(np.transpose(self.state), normalize(CWa))
        # print 'Local Energy: ', aCWa
        return aCWa

    def update(self):
        """Todo - currently no implementation of energy minimization"""
        pass

def energy_surface(lang, bindings, vocab, tree_set):
    """
    Evaluate the surface of the energy landscape
    """
    lang.networks = {}
    test_tree = None
    surface = np.zeros(len(bindings))

    for bs in lang.bindings.values():
        for b in bs:
            b.get_vectors(vocab)

    for tree in lang.trees:
        for b in tree.bindings:
            b.get_vectors(vocab)

    lang.build_constraints()
    lang.build_networks(vocab)

    for tree in lang.trees:
        if set(tree_set) == tree.binding_set():
            test_tree = tree

    for i in range(len(bindings)):
        subset = bindings[i]
        
        struct = copy.deepcopy(test_tree)
        
        for binding in subset:
            struct.remove(binding)
        surface[i] = lang.evaluate(struct)
    return surface