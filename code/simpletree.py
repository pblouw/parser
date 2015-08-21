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

dim = 512
N = 100

tally = np.zeros((N,9))

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


lang = Language(rules)
lang.build_trees(16)

for i in range(N):
    vocab = Vocabulary(dimensions=dim, unitary=True)
    vocab.add('BIAS')

    for item in get_fillers(rules):
        vocab.add(item)   
    for item in get_roles('', depth=4):
        vocab.add(item)
    
    tally[i,:] = energy_surface(lang, removals, vocab, tree_set)

data = -1*np.mean(tally, axis=0)
error = 1.96*np.std(tally, axis=0)
np.save('SimpleTree-Data-'+str(dim), data)
np.save('SimpleTree-Error-'+str(dim), error)