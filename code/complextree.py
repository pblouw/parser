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


dim = 512
N = 100

tally = np.zeros((N,9))

tree_set = ['S','l*NP','r*VP','rm*V','ll*DET','lr*N',
             'llm*the','lrm*thief','rmm*stole']

removals = [['S','llm*the','lrm*thief','rm*V'],
            ['rm*V','llm*the','rmm*stole','l*NP'],
            ['llm*the','r*VP','rmm*stole'],              
            ['rmm*stole'],
            [],
            ['lrm*thief'],
            ['lrm*thief','rm*V','lr*N'],
            ['rm*V','llm*the','rmm*stole','l*NP'],
            ['l*NP','rm*V','r*VP','ll*DET','lr*N']]

lang = Language(rules)
lang.build_trees(624)

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
np.save('ComplexTree-Data-'+str(dim), data)
np.save('ComplexTree-Error-'+str(dim), error)
