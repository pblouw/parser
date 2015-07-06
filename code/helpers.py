import numpy as np
import sys

# Symbolic functions
def label(role, filler):
    if role == '':
        return role+filler
    else:
        return role+'*'+filler


def get_roles(tip, depth=5):
    """Recursive generator for building role identifying strings"""
    if depth == 0:
        return
    paths = [tip+x for x in ('l','m','r')] # Change for binary branching trees
    for path in paths:
        yield path
        for subpath in get_roles(path, depth=depth-1):
            yield subpath


def get_fillers(rules):
    keys = rules.keys()
    values = flatten(rules.values())
    return keys+values

def flatten(lst):
    """Flatten an arbitrarily nested list"""
    acc = []
    for item in lst:
        if type(item) == type([]):
            acc += flatten(item)
        else:
            acc.append(item)
    return acc

# Vector functions
def inverse(vec):
    return np.roll(vec[::-1], 1)

def convolve(a, b):
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def get_convolution_matrix(vec):
    """Return the matrix that does a circular convolution by this vector.

    This should be such that A*B == dot(A.get_convolution_matrix, B.v)
    """
    D = len(vec)
    T = []
    for i in range(D):
        T.append([vec[(i - j) % D] for j in range(D)])
    return np.array(T)

def norm_of_vector(v):
        return np.linalg.norm(v)

def normalize(v):
    if norm_of_vector(v) > 0:
        return v / norm_of_vector(v)
    else:
        return np.zeros(len(v))







