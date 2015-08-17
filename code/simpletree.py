import random
import sys
import copy
import numpy as np 
import matplotlib.pyplot as plt

from vsa import *
from helpers import *
from grammar import *
from networks import *

rules = {'S':[['NP','VP']],
         'NP':[['john'],['sarah'],['adam'],['michelle']],
         'VP':[['V']],
         'V':[['ran'],['ate'],['lied'],['stole']]}

dims = [512]
N = 100

tally_1 = np.zeros((N,9))


tree_set = ['S','l*NP','r*VP','rm*V','lm*john','rmm*ran']

removals = [['S','lm*john','rm*V'],
            ['lm*john','r*VP','rmm*ran'], 
            ['rm*V','lm*john','rmm*ran','l*NP'],       
            ['rmm*ran'],
            [],
            ['lm*john'],
            ['l*NP','rm*V','lm*john','rmm*ran'],
            ['S','l*NP','r*VP','rmm*ran'],
            ['l*NP','rm*V','r*VP']] 

def energy_surface(lang, bindings, vocab):
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
        print 'Subset is: ', subset, 'Energy = ', surface[i]
    print ''
    return surface


for dim in dims:
    lang = Language(rules)
    lang.build_trees(16)

    for i in range(N):
        lang.networks = {}

        vocab = Vocabulary(dimensions=dim, unitary=True)
        for item in get_fillers(rules):
            vocab.add(item)   
        for item in get_roles('', depth=4):
            vocab.add(item)
        vocab.add('BIAS')

        tally_1[i,:] = energy_surface(lang, removals, vocab)

    data_1 = -1*np.mean(tally_1, axis=0)
    error_1 = 1.96*np.std(tally_1, axis=0)
    np.save('Tree1-Data-'+str(dim), data_1)
    np.save('Tree1-Error-'+str(dim), error_1)

    # data_2 = -1*np.mean(tally_2, axis=0)
    # error_2 = 1.96*np.std(tally_2, axis=0)
    # np.save('Tree2-Data-'+str(dim), data_2)
    # np.save('Tree2-Error-'+str(dim), error_2)