import random
import sys
import copy
import numpy as np 
import matplotlib.pyplot as plt

from vsa import *
from helpers import *
from grammar import *
from networks import *


# New function definitions for dealing with HRRs
def disturb(v, std=0.001):
    """Add mean-zero gaussian noise to a vector"""
    return v + np.random.normal(loc=0, scale=std, size=len(v)) 

def norm_of_vector(v):
    """Return the length of a vector"""
    return np.linalg.norm(v)

def normalize(v):
    """Normalize a vector"""
    return v / norm_of_vector(v)


rules = {'S':[['NP','VP']],
         'NP':[['DET','N']],
         'VP':[['V']],#['V','NP']],
         'V':[['wrote']],
         'DET':[['a']],#['the'],['my'],['one']],
         'N':[['musketeer']]}# ['castle'],['squirrel'],['wizard'],['astronaut']]}

dim = 512
# Use randomly chosen HRRs from unit hypersphere

# Create Vocab
vocab = Vocabulary(dimensions=dim, unitary=True)

for item in get_fillers(rules):
    vocab.add(item)   
for item in get_roles('', depth=4):
    vocab.add(item)
vocab.add('BIAS')


lang = Language(rules)
lang.build_trees(1)

for binding_set in lang.bindings.values():
    for b in binding_set:
        b.get_vectors(vocab)

for tree in lang.trees:
    for b in tree.bindings:
        b.get_vectors(vocab)

lang.build_constraints()
tree = lang.trees[0]

weights = np.zeros((dim,dim))

con_list = ['S -> l*NP','S -> r*VP','S -> BIAS','r*VP -> S', 'l*NP -> S']

for b in tree.bindings:
	for c in b.constraints.keys():
		if c in con_list:
			weights += b.constraints[c]


NP = convolve(vocab['NP'].v, vocab['l'].v)
VP = convolve(vocab['VP'].v, vocab['r'].v)


weights_in = np.row_stack([vocab['S'].v, NP,
						   VP, vocab['BIAS'].v])
weights_out = np.column_stack([vocab['S'].v, VP,
 						  	   VP, vocab['BIAS'].v*-3])


weights = np.dot(weights, weights_out)
weights = np.dot(weights_in, weights)

from scipy import integrate
from scipy.special import logit

T = 1
dt = 0.001
k = 25
tau = 1
alpha = 10
bias = -5*np.ones(4)

# Use sigmoid as non-linearity
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))

# Clipping for sigmoid non-linearity
def clip(v):
    return np.clip(v, a_min=0.001, a_max=0.999)

def integral(activation, dx=0.0001):
    """Compute the integral of arctanh from 0 to x=activation"""
    return integrate.quad(np.arctanh, 0.0, activation)

# Revised to modify the pattern over a single timestep      
def update(W, pattern):  
    pattern = pattern.flatten()
    inp = (logit(pattern) / alpha) - bias / alpha
    dx = (-(inp / tau) + np.dot(W,pattern)) * dt
    da = k * (sigmoid(inp + dx) - sigmoid(inp))
    pattern += da
    pattern = clip(pattern)
    return pattern

def energy(W, pattern):
    """Get the energy of the current network state"""
    core = -1*np.dot(pattern.T, np.dot(W,pattern))
    # unit = ((1/tau)/alpha) * sum([integral(activation)[0] for activation in pattern])
    # base = -1*sum(np.multiply(pattern, bias))
    return core 

inp = normalize(disturb(np.dot(weights_in, 0.35*NP+0.35*VP+0.35*vocab['BIAS'].v), 0.0001))

state = sigmoid(alpha*inp+bias)
state_log = np.zeros((4, int(T/dt)))
ens_log = np.zeros(int(T/dt))
# print inp
# print state
# print ''

# Run the network
for step in xrange(int(T/dt)):  
    state = update(weights, state)
    ens = energy(weights, state)
    # print ens
    state_log[:, step] = state
    ens_log[step] = ens

keys = ['S','l*NP','r*VP','BIAS','Normalized Energy']
#Plot the results
plt.figure(figsize=(12,8))
for i in range(state_log.shape[0]):
    plt.plot(range(int(T/dt)), state_log[i,:])
plt.plot(range(int(T/dt)), ens_log/-min(ens_log))
plt.ylim([-1.1,1.1])
plt.legend(keys, bbox_to_anchor=(1, 0.26))
plt.title('Parsing a Local Subtree', fontsize=16)
plt.xlabel('Time (ms)')
plt.ylabel('Cosine Similarity')
plt.show()

print state

