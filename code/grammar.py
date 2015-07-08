import random
import numpy as np
from vsa import *
from helpers import *
from networks import *

class Language(object):
    """A language, or set of strings, defined by a formal grammar.
    """
    def __init__(self, rules, vocab):
        self.rules = rules
        self.vocab = vocab
        self.networks = {}
        self.trees = []
        self.tree_cache = []
        self.bindings = {}
        self.binding_cache = []

    def evaluate(self, tree):
        """Evaluate the wellformedness of a linguistic structure"""
        tree_vocab = [b.label for b in tree.bindings]
        score = 0

        for binding in tree.bindings:
            # Run only those networks applicable to the tree
            if binding.label in self.networks:
                nets = self.networks[binding.label]
                network = None

                # Choose best network if option available
                # This is not robust - fix asap
                if len(nets) < 2:
                    network = nets[0]
                else:
                    for net in nets:
                        voc = set(net.vocab)
                        if voc < set(tree_vocab) and len(voc) > 1:
                            network = net
                        else:
                            pass
                    if network == None:
                        network = random.choice(nets)

                # Initiliaze network state from tree bindings
                state_accumulator = [b.v for b in tree.bindings 
                                     if b.label in network.vocab]
                
                state = sum(state_accumulator) + self.vocab['BIAS'].v
                network.state = normalize(state)

                # Compute the energy of the network and reset
                score += network.compute_energy()
        return score

    def build_trees(self, n):
        """Add n new randomly generated trees to the language"""
        if n == 0:
            pass
        else:
            new_tree = Tree(self.rules, self.vocab)

            # Check the tree cache for tree duplication
            if new_tree.label not in self.tree_cache:
                self.trees.append(new_tree)
                self.tree_cache.append(new_tree.label)

                # Add a unique identifier to each binding for caching purposes
                for b in new_tree.bindings:
                    b.id = b.label + ' -> (' 
                    b.id = b.id + ' '.join([c.label for c in b.children]) +')'

                    if b.id not in self.binding_cache: 
                        self.bindings[b.label] = self.bindings.get(b.label,[])
                        self.bindings[b.label].append(b)
                        self.binding_cache.append(b.id)
                
                # Recursive call
                self.build_trees(n-1)
            else:
                self.build_trees(n)

    def build_networks(self):
        """Add required networks to compute the language's constraints"""
        for lst in self.bindings.values():
            for b in lst:
                vocab = [b.label] + [c.label for c in b.children]

                # Only add a network for local trees that include children
                if len(b.children) > 0:
                    constraints = b.constraints.keys()
                    weights = sum([b.constraints[x] for x in b.constraints])
                    damping = -2 * len(b.children)
                    bias = self.vocab['BIAS'].v

                    # Define the network
                    net = Hopfield(b.label, bias)
                    net.vocab = vocab
                    net.weights = weights
                    net.constraints = constraints

                    # Define the input to the cleanup memory
                    c_vecs = [child.v for child in b.children]
                    pi_vecs = [b.v / float(x+1) for x in range(len(c_vecs))] 
                    in_vecs = pi_vecs + c_vecs + [bias]

                    # Define the output of the cleanup memory
                    po_vecs = [b.v for x in range(len(c_vecs))] 
                    out_vecs = po_vecs + c_vecs + [bias*damping]

                    net.clean_in = np.row_stack(in_vecs)
                    net.clean_out = np.column_stack(out_vecs)

                    if b.label in self.networks.keys():
                        self.networks[b.label] += [net]
                    else:
                         self.networks[b.label] = [net]

    def build_constraints(self):
        """Generate constraints for each binding in the language"""
        for lst in self.bindings.values():
            for binding in lst:
                binding.get_constraints()


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
    def __init__(self, rules, vocab, empty=False):
        self.bindings = []
        self.vocab = vocab
        self.rules = rules
        if empty:
            pass
        else:
            self.build(rules, root_binding=Binding('', 'S', vocab))

    def __getitem__(self, key):
        """Return the binding with the requested label"""
        for binding in self.bindings:
            if binding.label == key:
                return binding
        raise KeyError('Tree does not contain the provided binding')

    def build(self, rules, root_binding):
        """Build a tree using a provided set of grammatical rules"""
        self.bindings.append(root_binding)
        self.extend(root_binding)
        self.label = ''.join(self.display())

    def extend(self, binding):
        """Grow a tree from a binding that acts as the tree's root"""
        if binding.filler in self.rules:

            num_dict = {1:('m'), 2:('l','r'), 3:('l','m','r')}
            subfillers = random.choice(self.rules[binding.filler])   
            subroles = [binding.role+x for x in num_dict[len(subfillers)]]

            for i in xrange(len(subroles)):
                role = subroles[i]
                filler = subfillers[i]
                new = Binding(role, filler, self.vocab)
                binding.children.append(new)
                self.bindings.append(new)
                self.extend(new)

    def display(self):
        """Get the terminal symbols of the tree in correct order"""
        bindings = [b for b in self.bindings if b.filler.islower()]
        bindings.sort(key = lambda x: x.role)
        return [b.filler for b in bindings]


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
    def __init__(self, role, filler, vocab):
        self.label = label(role, filler)
        self.role = role
        self.filler = filler
        self.vocab = vocab
        self.children = []
        self.constraints = {}
        
        self.v = convolve(self.role_to_vec(role), self.vocab[filler].v)

    def role_to_vec(self, role):
        """Build a vector encoding for a role used in tree bindings"""
        if role == '':
            return HRR.identity(self.vocab.dimensions)
        else:
            return self.vocab[role].v

    def get_constraints(self):
        """Adds a set of constraints to the binding in the form of a
        dictionary attribute. Each key of the dictionary is a constraint 
        label and each value is a matrix that implements the constraint.
        The constraint label identifies the bindings that the constraint
        is defined over. 
        """
        for child in self.children:
            depth = len(child.role[:-1])

            # Bias constraint
            bias_hrr = convolve(inverse(self.v), self.vocab['BIAS'].v)
            bias_transform = get_convolution_matrix(bias_hrr)

            # Bias label
            bias_label = self.role + '*' + self.filler + ' -> BIAS'
            bias_label = bias_label[1:] if self.role == '' else bias_label

            # Parent to child constraint
            pc_hrr = convolve(inverse(self.v), child.v)  
            pc_transform = get_convolution_matrix(pc_hrr)

            # Parent to child label
            pc_left = self.role + '*' + self.filler
            pc_right = child.role + '*' + child.filler
            pc_label = pc_left + ' -> ' + pc_right

            # Child to parent constraint
            cp_hrr = convolve(inverse(child.v), self.v)
            cp_transform = get_convolution_matrix(cp_hrr)

            # Child to parent label
            cp_left = child.role + '*' + child.filler
            cp_right = self.role + '*' + self.filler
            cp_label = cp_left + ' -> ' + cp_right

            # Check for presence of tree root and strip convolution symbol
            if self.role == '':
                pc_label = self.role + self.filler + ' -> ' + pc_right
                cp_label = cp_left + ' -> ' + self.role + self.filler

            self.constraints[bias_label] = bias_transform
            self.constraints[pc_label] = pc_transform
            self.constraints[cp_label] = cp_transform
