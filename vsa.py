import numpy as np

class Vocabulary(object):
    """A collection of vectors that are used to define tree constituents and
    grammatical constraints. The vocab acts as a dictionary, with each vector
    label acting as a key, and each vector acting as a value.

    Parameters
    -----------
    dimensions : int
        Number of dimensions for each vector in the vocabulary.
    unitary : bool, optional
        If True, all generated vectors are unitary vectors.
    max_similarity : float, optional
        When randomly generating vectors, ensure that the cosine of the
        angle between the new vector and all existing vectors is less
        than this amount.  If the system is unable to find such a vector
        after 10000 tries, a warning message is printed.
    """

    def __init__(self, dimensions, max_similarity=0.2, unitary=False):
        self.dimensions = dimensions
        self.max_similarity = max_similarity
        self.unitary = unitary
        self.vectors = np.zeros((0, dimensions), dtype=float)
        self.items = {}

    def __getitem__(self, key):
        return self.items[key]

    def add(self, key, data=None, attempts=10000):
        """Add a new vector to the Vocabulary, tagged with the a text label"""
        count = 0
        if data != None:
            hrr = HRR(data, self.unitary)
        else:
            hrr = HRR(self.dimensions, self.unitary)

        if len(self.items) > 0:
            while count < attempts:
                similarity = np.dot(self.vectors, hrr.v)
                if max(similarity) < self.max_similarity:
                    break
                hrr = HRR(self.dimensions, self.unitary)
                count += 1
            else:
                print 'Could not generate a good vector after ' \
                      '%s attempts' % (attempts)

        self.items[key] = hrr
        self.vectors = np.vstack([self.vectors, hrr.v])


class HRR(object):
    """ A holographic reduced representation, based on the work of Tony Plate
    (2003). HRRs constitute a vector symbolic architecture for encoding 
    symbolic structures in high-dimensional vector spaces. HRRs utilize 
    circular convolution for binding and vector addition for superposition. 
    Binding and unbinding are approximate, not exact, so HRRs are best thought of 
    as providing a lossy compression of a symbol structure into a vector space.

    Parameters
    -----------
    data : int or np.array
        An int specifies the dimensionality of a randomly generated HRR. An
        np.array can provided instead to specify the value on each dimension of
        the HRR. (note that these values must be statistically distributed in 
        a particular way in order for the HRR to act as expected)
    unitary : bool, optional
        If True, the generated HRR will be unitary, i.e. its exact and 
        approximate inverses are equivalent.
    """

    def __init__(self, data, unitary=False):
        if type(data) == int:
            self.v = np.random.randn(data)
        elif isinstance(data, np.ndarray):
            self.v = data
        else:
            raise Exception('Data must a numpy array or an integer')
        
        self.normalize()
        if unitary:
            self.make_unitary()

    def __add__(self, other):
        """Operator overloading for performing vector addition"""
        if isinstance(other, HRR):
            return HRR(self.v + other.v)
        else:
            raise Exception('Both objects must by HRRs')    

    def __mul__(self, other):
        """Operator overloading for performing circular convolution"""
        if isinstance(other, HRR):
            return self.convolve(other.v)
        else:
            raise Exception('Both objects must by HRRs')

    def __invert__(self):
    	"""Operator overloading for performing pseudo-inversion"""
        return HRR(data=self.v[-np.arange(len(self.v))])

    def length(self):
        """Return the L2 norm of the vector"""
        return np.linalg.norm(self.v)

    def normalize(self):
        """Modify the vector to have an L2 norm of 1"""
        nrm = np.linalg.norm(self.v)
        if nrm > 0:
            self.v /= nrm 
        else:
            self.v = np.zeros(len(self.v))

    def convolve(self, other):
        """Return the circular convolution of two HRRs"""
        x = np.fft.ifft(np.fft.fft(self.v) * np.fft.fft(other.v)).real
        return SemanticPointer(data=x)

    def make_unitary(self):
        """Make the HRR unitary"""
        fft_val = np.fft.fft(self.v)
        fft_imag = fft_val.imag
        fft_real = fft_val.real
        fft_norms = np.sqrt(fft_imag ** 2 + fft_real ** 2)
        fft_unit = fft_val / fft_norms
        self.v = np.array((np.fft.ifft(fft_unit)).real)

    def get_circulant_matrix(self):
        """Generate the matrix that performs a linear transformation 
        equivalent to convolution by this HRR. 
        (i.e. A*B == np.dot(A.get_convolution_matrix(), B.v))"""
        D = len(self.v)
        T = []
        for i in range(D):
            T.append([self.v[(i - j) % D] for j in range(D)])
        return np.array(T)

    @staticmethod  
    def identity(dim):
    	"""Generate the identity vector for convolution"""
        v = np.zeros(dim)
        v[0] = 1
        return v






