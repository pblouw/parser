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
         'V':[['wrote'],['ate']],#['chased'],['wanted']],
         'DET':[['a']],#['the'],['my'],['one']],
         'N':[['cowboy'],['musketeer']]}# ['castle'],['squirrel'],['wizard'],['astronaut']]}
         
dim = 512

vocab = Vocabulary(dimensions=dim, unitary=True)

for item in get_fillers(rules):
	vocab.add(item)   
for item in get_roles('', depth=4):
	vocab.add(item)
vocab.add('BIAS')

lang = Language(rules, vocab)
lang.build_trees(2)
lang.build_constraints()
lang.build_networks()

tree = lang.trees[0]

for binding in tree.bindings:
	print binding.label
	for child in binding.children:
		print child.label
	print ''

tree.remove('l*NP')

print '------'
print '------'


for binding in tree.bindings:
	print binding.label
	for child in binding.children:
		print child.label
	print ''

sys.exit()

score = lang.evaluate(tree)

print score
print ''
for net in lang.networks:
	print net

print ''
for binding in tree.bindings:
	print binding.label



sys.exit()

for b in lang.bindings:
	print b, lang.bindings[b]
print lang.tree_cache

print ''
for nets in lang.networks.values():
	for n in nets:
		print n.vocab
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
