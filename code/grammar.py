import random
import sys
import numpy as np
from vsa import *
from helpers import *
from networks import *

class Language(object):
    """A language, or set of strings, defined by a formal grammar.
    """
    def __init__(self, rules):
        self.rules = rules
        self.networks = {}
        self.trees = []
        self.tree_cache = []
        self.bindings = {}
        self.binding_cache = []

    def evaluate(self, tree):
        """Evaluate the wellformedness of a linguistic structure"""
        score = 0
        for binding in tree.bindings:
            # Run only those networks applicable to the tree
            if binding.label in self.networks.keys():
                nets = self.networks[binding.label]

                # Choose right network if option available
                if len(nets) < 2:
                    network = nets[0]
                else:
                    for net in nets:
                        # Check net vocab against binding labels
                        n_voc = set(net.bindings)
                        b_voc = set([c.label for c in binding.children])
                        b_voc.add(binding.label)
                        if binding.parent != None:
                            b_voc.add(binding.parent.label)

                        if n_voc == b_voc:
                            network = net
                            break
                    else:
                        network = random.choice(nets)

                # Initiliaze network state from tree bindings
                state_accumulator = [b.v for b in tree.bindings 
                                     if b.label in network.bindings]
                
                state = sum(state_accumulator) + self.vocab['BIAS'].v
                network.state = normalize(state)

                # Compute the energy of the network and reset
                score += network.compute_energy()


        return score

    def build_trees(self, n):
        """Add n new randomly generated trees to the language"""
        while n > 0:
            new_tree = Tree(self.rules)

            # Check the tree cache for tree duplication
            if new_tree.label not in self.tree_cache:
                self.trees.append(new_tree)
                self.tree_cache.append(new_tree.label)

                # Add a unique identifier to each binding for caching purposes
                for b in new_tree.bindings:
                    b.id = ''
                    if b.parent != None:
                        b.id += '('+b.parent.label +') <- '
                    b.id += b.label + ' -> (' 
                    b.id = b.id + ' '.join([c.label for c in b.children]) +')'

                    if b.id not in self.binding_cache: 
                        self.bindings[b.label] = self.bindings.get(b.label,[])
                        self.bindings[b.label].append(b)
                        self.binding_cache.append(b.id)
                n -= 1
                print n

    def build_networks(self, vocab):
        self.vocab = vocab

        """Add required networks to compute the language's constraints"""
        for binding_set in self.bindings.values():
            for b in binding_set:
                binding_voc = [b.label] + [c.label for c in b.children]   
                if b.parent != None:
                    binding_voc += [b.parent.label] 
                    mag = len(b.children) + 1   
                else: 
                    mag = len(b.children)

                # Only add a network for local trees that include children
                # if len(b.children) > 0:
                weights = sum([b.constraints[x] for x in b.constraints])
                bval = -1 * mag

                # Define the network
                net = Hopfield()
                net.bindings = binding_voc
                net.weights = weights

                # Define the input to the cleanup memory
                p_vec = [b.parent.v if b.parent != None else np.zeros(self.vocab.dimensions)]
                c_vecs = [child.v for child in b.children]
                
                # pi_vecs = [b.v / float(x+1) for x in range(len(c_vecs))] 
                in_vecs = p_vec + c_vecs + [self.vocab['BIAS'].v]

                # Define the output of the cleanup memory
                # po_vecs = [b.v for x in range(len(c_vecs))] 
                out_vecs = p_vec + c_vecs + [self.vocab['BIAS'].v*bval]

                net.clean_in = np.row_stack(in_vecs)
                net.clean_out = np.column_stack(out_vecs)

                if b.label in self.networks.keys():
                    self.networks[b.label] += [net]
                else:
                     self.networks[b.label] = [net]

    def build_constraints(self):
        """Generate constraints for each binding in the language"""
        for binding_set in self.bindings.values():
            for b in binding_set:
                b.get_constraints()


class Tree(object):
    """A collection of role-filler bindings that constitute a wellformed
    tree according to a supplied set of symbolic rewrite rules. Bindings 
    can be removed from a tree (rendering it unwellformed) for the 
    purposes of testing how constraints can be used to evaluate the 
    degree of wellformedness of a linguistic structure.

    Parameters
    ----------
    rules : dict
        Each key is symbol corresponding to the left hand side of a 
        rewrite rule, and each value is a list of symbol lists 
        corresponding to the right hand side of a rewrite rule.
        (each dict entry corresponds to one or more rewrite rules)
    empty : bool, optional
        If true, initiliaze an empty tree that can be built manually.
    """
    def __init__(self, rules, empty=False):
        self.bindings = []
        self.rules = rules
        if empty:
            pass
        else:
            self.build(rules, root_binding=Binding('', 'S'))

    def __getitem__(self, key):
        """Return the binding with the requested label"""
        for binding in self.bindings:
            if binding.label == key:
                return binding
        raise KeyError('Tree does not contain the provided binding')

    def remove(self, label):
        for b in self.bindings[:]:
            for child in b.children[:]:
                if child.label == label:
                    b.children.remove(child)
            if label == b.label:
                self.bindings.remove(b)
            if b.parent != None:
                if b.parent.label == label:
                    b.parent = None

    def build(self, rules, root_binding):
        """Build a tree using a provided set of grammatical rules"""
        self.bindings.append(root_binding)
        self.extend(root_binding, None)
        self.label = ''.join(self.display())

    def extend(self, binding, above):
        """Grow a tree from a binding that acts as the tree's root"""
        if above != None:
            binding.parent = Binding(above.role, above.filler)

        if binding.filler in self.rules:
            num_dict = {0:(''), 1:('m'), 2:('l','r'), 3:('l','m','r')}
            subfillers = random.choice(self.rules[binding.filler])   
            subroles = [binding.role+x for x in num_dict[len(subfillers)]]

            for i in xrange(len(subroles)):
                role = subroles[i]
                filler = subfillers[i]
                new = Binding(role, filler)
                binding.children.append(new)
                self.bindings.append(new)
                self.extend(new, binding)

    def display(self):
        """Get the terminal symbols of the tree in correct order"""
        bindings = [b for b in self.bindings if b.filler.islower()]
        bindings.sort(key = lambda x: x.role)
        return [b.filler for b in bindings]

    def binding_set(self):
        """Get the set of bindings that constitute the tree"""
        return set([b.label for b in self.bindings])

class Binding(object):
    """
    A tree-constituent that binds a particular filler to a particular role or 
    tree node. Bindings contain references to their children (i.e. the 
    bindings that follow immediately in the tree) and can be used to generate 
	the constraints that relate these children to their parent (& vice versa).
    Trees consist of a set of bindings that are consistent with a particular
    formal grammar.

    Parameters
    ----------
    role : str
        A sequence of letters identifying a pathway from the root of a tree to
        a particular node in the tree. Each letter labels a branching path at
        each depth in the tree.
    filler : str
        A symbol occupying a particular node in a tree. Symbols can correspond
        to syntactic categories (e.g. NP) or to words (e.g. "many").
    """
    def __init__(self, role, filler):
        self.label = label(role, filler)
        self.role = role
        self.filler = filler
        self.children = []
        self.parent = None
        self.constraints = {}

    def role_to_vec(self, role):
        """Build a vector encoding for a role used in tree bindings"""
        if role == '':
            return HRR.identity(self.vocab.dimensions)
        else:
            return self.vocab[role].v

    def get_vectors(self, vocab):
        self.vocab = vocab
        self.v = convolve(self.role_to_vec(self.role), self.vocab[self.filler].v)
        for c in self.children:
            c.v = convolve(self.role_to_vec(c.role), self.vocab[c.filler].v)
        if self.parent != None:
            p = self.parent
            vec = convolve(self.role_to_vec(p.role), self.vocab[p.filler].v)
            self.parent.v = vec

    def get_constraints(self):
        """Adds a set of constraints to the binding in the form of a
        dictionary attribute. Each key of the dictionary is a constraint 
        label and each value is a matrix that implements the constraint.
        The constraint label identifies the bindings that the constraint
        is defined over. 
        """
       
        # Bias constraint
        bias_hrr = convolve(inverse(self.v), self.vocab['BIAS'].v)
        bias_transform = get_convolution_matrix(bias_hrr)

        # Bias label
        bias_label = self.role + '*' + self.filler + ' -> BIAS'
        bias_label = bias_label[1:] if self.role == '' else bias_label
        self.constraints[bias_label] = bias_transform

        # Parent Constraint
        if self.parent != None:
            p = self.parent

             # Main to parent constraint
            mp_hrr = convolve(inverse(self.v), p.v)
            mp_transform = get_convolution_matrix(mp_hrr)

            # Main to parent label
            mp_left = self.role + '*' + self.filler
            mp_right = p.role + '*' + p.filler
            mp_label = mp_left + ' -> ' + mp_right

            # Check for presence of tree root and strip convolution symbol
            if self.role == '':
                mp_label = self.role + self.filler + ' -> ' + mp_right
            if p.role == '':
                mp_label = mp_left + ' -> ' + p.role + p.filler

            self.constraints[mp_label] = mp_transform

        # Child Constraints
        for child in self.children:
            # Main to child constraint
            mc_hrr = convolve(inverse(self.v), child.v)  
            mc_transform = get_convolution_matrix(mc_hrr)

            # Main to child label
            mc_left = self.role + '*' + self.filler
            mc_right = child.role + '*' + child.filler
            mc_label = mc_left + ' -> ' + mc_right

            # Check for presence of tree root and strip convolution symbol
            if self.role == '':
                mc_label = self.role + self.filler + ' -> ' + mc_right
            self.constraints[mc_label] = mc_transform
