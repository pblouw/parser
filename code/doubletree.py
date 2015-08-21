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
         'NP':[['john'],['sarah'],['adam'],['michelle'],['DET','N']],
         'VP':[['V'],['V','NP']],
         'V':[['ran'],['ate'],['lied'],['stole']],
         'DET':[['a'],['the']],
         'N':[['hamburger'],['dollar'],['thief'],['astronaut']]}


dims = 512
N = 100

tally = np.zeros((N,14))

tree_set1 = ['S','l*NP','r*VP','rm*V','ll*DET','lr*N',
             'llm*the','lrm*thief','rmm*stole']

tree_set2 = ['S','l*NP','r*VP','rl*V','rr*NP','ll*DET','lr*N','rrl*DET','rrr*N',
             'llm*the','lrm*thief','rlm*stole','rrlm*a','rrrm*dollar']

removals = [['S','llm*the','lrm*thief','rm*V'],
            ['rm*V','llm*the','rmm*stole','l*NP'],
            ['llm*the','r*VP','rmm*stole'],              
            ['rmm*stole'],
            [],
            ['lrm*thief'],
            ['lrm*thief','rm*V','lr*N'],
            ['rrr*N','rr*NP','rrl*DET','rl*V'],
            ['rrl*DET','rr*NP'],
            ['rrlm*a'],
            [],
            ['rrrm*dollar'],
            ['rrr*N','rr*NP','rrl*DET'],
            ['llm*the','lrm*thief','l*NP','rrl*DET','rl*V']]

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

    for i in range(len(bindings)):
        
        for tree in lang.trees:
            if i < 7:
                if set(tree_set1) == tree.binding_set():
                    test_tree = tree
            else:
                if set(tree_set2) == tree.binding_set():
                    test_tree = tree      

        subset = bindings[i]
        struct = copy.deepcopy(test_tree)
        
        for binding in subset:
            struct.remove(binding)
        surface[i] = lang.evaluate(struct)
    return surface

for dim in dims:
    lang = Language(rules)
    lang.build_trees(624)

    for i in range(N):
        lang.networks = {}

        vocab = Vocabulary(dimensions=dim, unitary=True)
        for item in get_fillers(rules):
            vocab.add(item)   
        for item in get_roles('', depth=4):
            vocab.add(item)
        vocab.add('BIAS')

        tally_2[i,:] = energy_surface(lang, removals, vocab)

    data_1 = -1*np.mean(tally_2, axis=0)
    error_1 = 1.96*np.std(tally_2, axis=0)
    np.save('Tree2-Data-'+str(dim), data_1)
    np.save('Tree2-Error-'+str(dim), error_1)
