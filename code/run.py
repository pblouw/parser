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
         'NP':[['DET','N'],['DET','ADJ','N']],
         'ADJ':[['angry']],
         'VP':[['V'],['V','NP']],
         'V':[['ate'],['chased'],['wanted'],['followed']],
         'DET':[['a'],['the'],['my'],['some']],
         'N':[['apple'],['cowboy'],['musketeer'],['castle'],['squirrel'],['wizard'],['astronaut']]}
         
dims = [128, 256, 512]
N = 25

tally_1 = np.zeros((N,14))
tally_2 = np.zeros((N,16))

for dim in dims:
    for i in range(N):
        vocab = Vocabulary(dimensions=dim, unitary=True)

        for item in get_fillers(rules):
            vocab.add(item)   
        for item in get_roles('', depth=4):
            vocab.add(item)
        vocab.add('BIAS')

        set_1 = ['S','l*NP','r*VP','ll*DET','lr*N','rl*V','rr*NP','rrl*DET',
                 'rrr*N','rrrm*squirrel','rrlm*a','lrm*castle','llm*the','rlm*ate']

        set_2 = ['S','l*NP','r*VP','ll*DET','lr*N','rl*V','rr*NP','rrl*DET',
                 'rrm*ADJ','rrr*N','lrm*cowboy','llm*a','rmm*chased',
                 'rrlm*the','rrmm*angry','rrrm*wizard']

        set_1 = set_1[::-1]
        set_2 = set_2[::-1]


        lang = Language(rules, vocab)
        lang.build_trees(12700)
        lang.build_constraints()
        lang.build_networks()

        tree1 = None
        tree2 = None

        for tree in lang.trees:
            if tree.display() == ['the', 'castle','ate','a','squirrel']:
                tree1 = tree
            if tree.display() == ['a','cowboy','chased','the','angry','wizard']:
                tree2 = tree

        for j in range(14):
            subset = set_1[:j]
            print subset   
            test = copy.deepcopy(tree1)
            
            for binding in subset:
                test.remove(binding)

            score = lang.evaluate(test)
            tally_1[i,j] = score

        for j in range(16):  
            subset = set_2[:j]
            test = copy.deepcopy(tree2)
        
            for binding in subset:
                test.remove(binding)    
            
            score = lang.evaluate(test)
            tally_2[i,j] = score

    data_1 = -1*np.mean(tally_1, axis=0)
    error_1 = 1.96*np.std(tally_1, axis=0)

    np.save('Tree1-Data-'+str(dim), data_1)
    np.save('Tree1-Error-'+str(dim), error_1)

    data_2 = -1*np.mean(tally_2, axis=0)
    error_2 = 1.96*np.std(tally_2, axis=0)
    np.save('Tree2-Data-'+str(dim), data_2)
    np.save('Tree2-Error-'+str(dim), error_2)