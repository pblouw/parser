import random
import sys
import copy
import numpy as np 
import matplotlib.pyplot as plt

from vsa import *
from helpers import *
from grammar import *
from networks import *

sys.setrecursionlimit(10000)

rules = {'S':[['NP','VP']],
         'NP':[['DET','N']],#,['DET','ADJ','N']],
         'VP':[['V'],['V','NP']],
         'V':[['ate'],['chased'],['wanted'],['followed']],
         # 'ADJ':[['sullen'],['wild']],
         'DET':[['a'],['the'],['my'],['some']],
         'N':[['wizard'],['cowboy'],['musketeer'],['castle'],['squirrel'],['wizard'],['astronaut']]}

dims = [128, 256, 512]
N = 50

# tally_1 = np.zeros((N,16))
tally_2 = np.zeros((N,14))

# set_1 = ['S','l*NP','r*VP','ll*DET','lr*N', 'rl*V','rr*NP','rrl*DET','rrr*N',
#          'llm*the','lmm*sullen','lrm*cowboy','rlm*chased','rrlm*a','rrrm*wizard']
set_2 = ['S','l*NP','r*VP','ll*DET','lr*N','rl*V','rr*NP','rrl*DET','rrr*N',
         'llm*a','lrm*cowboy','rlm*chased','rrlm*the','rrrm*wizard']

# set_1 = set_1[::-1]
set_2 = set_2[::-1]


def energy_surface(lang, bindings, vocab):
    lang.networks = {}
    test_tree = None
    surface = np.zeros(len(bindings))

    for bs in lang.bindings.values():
        for b in bs:
            b.get_vectors(vocab)

    for tree in lang.trees:
        # print tree.display()
        for b in tree.bindings:
            b.get_vectors(vocab)

    lang.build_constraints()
    lang.build_networks(vocab)

    # acc = []
    # print ''
    # for nets in lang.networks.values():
    #     acc += nets
    #     for n in nets:
    #         print n.label


    # print 'Number of networks: ', len(acc)

    for tree in lang.trees:
        if set(bindings) == tree.binding_set():
            test_tree = tree

    for i in range(len(bindings)):
        subset = bindings[:i]

        struct = copy.deepcopy(test_tree)
        
        for binding in subset:
            struct.remove(binding)

        surface[i] = lang.evaluate(struct)
    return surface

for dim in dims:

    lang = Language(rules)
    lang.build_trees(2400)

    for i in range(N):
        print dim, i
        lang.networks = {}

        vocab = Vocabulary(dimensions=dim, unitary=True)

        for item in get_fillers(rules):
            vocab.add(item)   
        for item in get_roles('', depth=4):
            vocab.add(item)
        vocab.add('BIAS')

        # tally_1[i,:] = energy_surface(lang, set_1, vocab)
        tally_2[i,:] = energy_surface(lang, set_2, vocab)

        set_1 = ['S','l*NP','r*VP','ll*DET','lr*N','lm*ADJ', 'rl*V','rr*NP','rrl*DET','rrr*N',
                 'llm*the','lmm*sullen','lrm*cowboy','rlm*chased','rrlm*a','rrrm*wizard']

        set_2 = ['S','l*NP','r*VP','ll*DET','lr*N','rl*V','rr*NP','rrl*DET','rrr*N',
                 'llm*a','lrm*cowboy','rlm*chased','rrlm*the','rrrm*wizard']

        set_1 = set_1[::-1]
        set_2 = set_2[::-1]

        for binding_set in lang.bindings.values():
            for b in binding_set:
                b.get_vectors(vocab)

        for tree in lang.trees:
            for b in tree.bindings:
                b.get_vectors(vocab)

        lang.build_constraints()
        lang.build_networks(vocab)

        tree1 = None
        tree2 = None

        for tree in lang.trees:
            if tree.display() == ['the','sullen','cowboy','chased','a','wizard']:
                tree1 = tree
            if tree.display() == ['a','cowboy','chased','the','wizard']:
                tree2 = tree

        for j in range(16):
            subset = set_1[:j]
            print subset   
            test = copy.deepcopy(tree1)
            
            for binding in subset:
                test.remove(binding)
            print
            score = lang.evaluate(test)
            tally_1[i,j] = score

        for j in range(14):  
            subset = set_2[:j]
            test = copy.deepcopy(tree2)
        
            for binding in subset:
                test.remove(binding)    
            
            score = lang.evaluate(test)
            tally_2[i,j] = score

    # data_1 = -1*np.mean(tally_1, axis=0)
    # error_1 = 1.96*np.std(tally_1, axis=0)
    # np.save('Tree1-Data-'+str(dim), data_1)
    # np.save('Tree1-Error-'+str(dim), error_1)

    data_2 = -1*np.mean(tally_2, axis=0)
    error_2 = 1.96*np.std(tally_2, axis=0)
    np.save('Tree2-Data-'+str(dim), data_2)
    np.save('Tree2-Error-'+str(dim), error_2)