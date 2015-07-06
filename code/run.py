import random
import numpy as np 
import matplotlib.pyplot as plt

from vsa import *
from helpers import *
from grammar import *
from networks import *

rules = {'S':[['A','B']],
         'A':[['C','S'],['a']],
         'B':[['b']],
         'C':[['c']]}

dims = [512, 256, 128]
N = 250

tally_2 = np.zeros((N,11))
tally_1 = np.zeros((N,10))


for dim in dims:
    for i in range(N):

        vocab = Vocabulary(dimensions=dim, unitary=True)

        for item in get_fillers(rules):
            vocab.add(item)   
        for item in get_roles('', depth=4):
            vocab.add(item)
        vocab.add('BIAS')

        set_1 = ['S','l*A','r*B','lm*a','rm*b','ll*C',
                 'lr*S','lrl*A','lrr*B','lrlm*a']
        set_2 = ['S','l*A','r*B','rm*b','lr*S','ll*C',
                 'lrl*A','lrr*B','llm*c','lrlm*a','lrrm*b']

        lang = Language(rules, vocab)
        lang.build_trees(2)
        lang.build_constraints()
        lang.build_networks()

        for j in range(10):
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


        for j in range(11):
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