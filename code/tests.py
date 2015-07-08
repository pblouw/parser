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
         'VP':[['V'],['V','NP']],
         'V':[['wrote'],['ate'],['chased'],['wanted']],
         'DET':[['a'],['the'],['my'],['one']],
         'N':[['cowboy'],['musketeer'], ['castle'],['squirrel'],['wizard'],['astronaut']]}
         
dim = 128

vocab = Vocabulary(dimensions=dim, unitary=True)

for item in get_fillers(rules):
	vocab.add(item)   
for item in get_roles('', depth=4):
	vocab.add(item)
vocab.add('BIAS')

lang = Language(rules, vocab)
lang.build_trees(35)
lang.build_constraints()
lang.build_networks()

# for b in lang.bindings:
# 	print b, lang.bindings[b]

# print ''
# for s in lang.tree_cache:
# 	print s
# for tree in lang.trees:
# 	print [b.label for b in tree.bindings]
# 	print ''

# print len(lang.bindings)
# print len(lang.networks)

# for net in lang.networks.values():
# 	print len(net)
