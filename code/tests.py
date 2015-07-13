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
         'NP':[['DET','N']],
         'VP':[['V']],#['V','NP']],
         'V':[['wrote'],['chased'],['wanted']],
         'DET':[['a']],#['the'],['my'],['one']],
         'N':[['musketeer']]}# ['castle'],['squirrel'],['wizard'],['astronaut']]}
         
dim = 512

vocab = Vocabulary(dimensions=dim, unitary=True)

for item in get_fillers(rules):
	vocab.add(item)   
for item in get_roles('', depth=4):
	vocab.add(item)
vocab.add('BIAS')


lang = Language(rules)
lang.build_trees(3)

for binding_set in lang.bindings.values():
    for b in binding_set:
        b.get_vectors(vocab)

for tree in lang.trees:
    for b in tree.bindings:
        b.get_vectors(vocab)

lang.build_constraints()
lang.build_networks(vocab)

for tree in lang.trees:
    if tree.display() == ['a','musketeer','chased']:
        tree1 = tree

set_1 = ['S','l*NP','r*VP','ll*DET','lr*N','rm*V',
         'lrm*musketeer','llm*a','rmm*chased']

set_1 = set_1[::-1]

for j in range(9):
    subset = set_1[:j]
    print subset   
    test = copy.deepcopy(tree1)
    
    for binding in subset:
        test.remove(binding)

    print test.display()
    score = lang.evaluate(test)
    print 'Score: ', score
    print ''
# for binding in tree.bindings:
# 	print binding.label
# 	for child in binding.children:
# 		print child.label
# 	print ''

# tree.remove('l*NP')

# print '------'
# print '------'


# for binding in tree.bindings:
# 	print binding.label
# 	for child in binding.children:
# 		print child.label
# 	print ''

# sys.exit()

# score = lang.evaluate(tree)

# print score
# print ''
# for net in lang.networks:
# 	print net

# print ''
# for binding in tree.bindings:
# 	print binding.label



# sys.exit()

for b in lang.bindings:
	print b, lang.bindings[b]
# print lang.tree_cache

print ''
for net in lang.networks:
	print net, lang.networks[net]


