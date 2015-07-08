import random
import sys
import numpy as np 
import matplotlib.pyplot as plt

from vsa import *
from helpers import *
from grammar import *
from networks import *

rules = {'S':[['NP','VP']],
         'NP':[['DET','N']],
         'VP':[['V']],#['V','NP']],
         'V':[['ate']],#['chased'],['wanted'],['followed']],
         'DET':[['an']],#['an'],['the']],#['my'],['some'],['every'],['one']],
         'N':[['apple'],['cowboy']]}#['musketeer'], ['castle'],['squirrel'],['wizard'],['astronaut']]}
         
dim = 512
N = 250

tally_2 = np.zeros((N,9))
tally_1 = np.zeros((N,9))

for i in range(N):

    vocab = Vocabulary(dimensions=dim, unitary=True)

    for item in get_fillers(rules):
        vocab.add(item)   
    for item in get_roles('', depth=4):
        vocab.add(item)
    vocab.add('BIAS')

    set_1 = ['S','l*NP','r*VP','ll*DET','lr*N',
             'lrm*apple','llm*an','rm*V','rmm*ate']
    set_2 = ['S','l*NP','r*VP','ll*DET','lr*N',
             'lrm*cowboy','llm*an','rm*V','rmm*ate']
    # set_2 = ['S','l*NP','r*VP','ll*DET','lr*N',
    #          'lrm*cowboy','llm*the','rm*V','rmm*wrote']

    lang = Language(rules, vocab)
    lang.build_trees(2)
    lang.build_constraints()
    lang.build_networks()

    for j in range(9):
        subset = set_1[:j+1]
        tree = Tree(rules, vocab, empty=True)
        for binding in subset:
            if binding == 'S':
                tree.bindings.append(Binding('','S', vocab))
            else:        
                role = binding.split('*')[0]
                filler = binding.split('*')[1]
                tree.bindings.append(Binding(role, filler, vocab))

        score = lang.evaluate(tree)
        tally_1[i,j] = score


    for j in range(9):
        subset = set_2[:j+1]
        tree = Tree(rules, vocab, empty=True)
        for binding in subset:
            if binding == 'S':
                tree.bindings.append(Binding('','S', vocab))
            else:            
                role = binding.split('*')[0]
                filler = binding.split('*')[1]
                tree.bindings.append(Binding(role, filler, vocab))

        score = lang.evaluate(tree)
        tally_2[i,j] = score

data_1 = -1*np.mean(tally_1, axis=0)
error_1 = 1.96*np.std(tally_1, axis=0)

np.save('Tree1-Data-'+str(dim), data_1)
np.save('Tree1-Error-'+str(dim), error_1)

data_2 = -1*np.mean(tally_2, axis=0)
error_2 = 1.96*np.std(tally_2, axis=0)
np.save('Tree2-Data-'+str(dim), data_2)
np.save('Tree2-Error-'+str(dim), error_2)