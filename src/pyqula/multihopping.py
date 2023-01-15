import numpy as np

from .algebra import isnumber,issparse
from . import algebra

is_number = isnumber

class MultiHopping():
    """Class for a multihopping"""
    def __init__(self,a):
        if type(a)==dict: # dictionary type
            self.dict = a # dictionary
        elif type(a)==MultiHopping: # multihopping type
            self.dict = a.dict # dictionary
        elif type(a)==np.ndarray or issparse(a) or type(a)==np.matrix:
            dd = dict() ; dd[(0,0,0)] = a
            self.dict = dd
        else: raise
    def __add__(self,a):
        if type(a)!=MultiHopping: return NotImplemented
        out = add_hopping_dict(self.dict,a.dict)
        out = MultiHopping(out) # create a new object
        return out
    def __mul__(self,a):
        if type(a)==MultiHopping:
            out = multiply_hopping_dict(self.dict,a.dict)
            out = MultiHopping(out) # create a new object
            return out
        elif isnumber(a):
            out = dict()
            for key in self.dict:
                out[key] = a*self.dict[key]
            out = MultiHopping(out) # create a new object
            return out
        else: return NotImplemented
    def __rmul__(self,a): 
        if isnumber(a): return self*a
        else: return NotImplemented
    def __neg__(self): return (-1)*self
    def __sub__(self,a): return self + (-a)
    def get_dict(self):
        return self.dict # dictionary
    def dot(self,a):
        """Compute the dot product"""
        a = MultiHopping(a) # as Multihopping
        return dot_hopping_dict(self.dict,a.dict) # dictionary
    def norm(self):
        """Norm of the MultiHopping"""
        return np.sqrt((self.dot(self)).real)
    def is_hermitian(self):
        dd = self - self.get_dagger()
        if dd.norm()>1e-7: return False
        else: return True
    def get_dagger(self):
        out = dict()
        for key in self.dict:
            key2 = tuple(-np.array(key,dtype=int))
            out[key] = np.conjugate(self.dict[key2].T)
        out = MultiHopping(out) # create a new object
        return out
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
    def is_zero(self):
        for key in self.dict:
            m = self.dict[key]
            if not algebra.is_zero(m): return False
        return True






def add_hopping_dict(hop1i,hop2i):
    """Multiply two hopping dictionaries"""
    # convert all the keys to tuples
    hop1,hop2 = dict(),dict()
    for key in hop1i: hop1[tuple(key)] = hop1i[key]
    for key in hop2i: hop2[tuple(key)] = hop2i[key]
    # now collect all the matrices
    out = dict() # create dictionary
    keys = [key1 for key1 in hop1]
    for key2 in hop2:
        if key2 not in keys: keys.append(key2) # store
    for key in keys:
        m = 0 # initialize
        if key in hop1: 
            m = (m + hop1[key]).copy()
        if key in hop2: 
            m = (m + hop2[key]).copy()
        out[key] = m.copy() # store
    return out


def dot_hopping_dict(hop1,hop2):
    """Multiply two hopping dictionaries"""
    out = dict() # create dictionary
    out = 0. # initialize
    keys = [key1 for key1 in hop1]
    for key2 in hop2:
        if key2 not in keys: keys.append(key2) # store
    for key in keys:
        m1,m2 = 0.,0.
        if key in hop1: 
            m1 = np.array(algebra.todense(hop1[key]))
        if key in hop2: 
            m2 = np.array(algebra.todense(hop2[key]))
        m = np.sum(np.conjugate(m1)*m2)
        out = out + m # add this contribution
    return out





def multiply_hopping_dict(hop1,hop2):
    """Multiply two hopping dictionaries"""
    out = dict() # create dictionary
    for key1 in hop1:
        for key2 in hop2:
            key = tuple(np.array(key1) + np.array(key2))
            m = hop1[key1]@hop2[key2] # multiply
            if np.max(np.abs(m))>1e-6: # if non-zero
                if key in out: out[key] = out[key] + m # add
                else: out[key] = m # store
    return out # return output




def direct_sum_hopping_dict(hop1,hop2):
    """Perform the direct sum of two hopping dictionaries"""
    from .algebra import direct_sum
    out = dict() # create dictionary
    keys = [key1 for key1 in hop1]
    keys2 = [key1 for key1 in hop2]
    zero1 = 0.*hop1[keys[0]] # initialize as zero
    zero2 = 0.*hop2[keys2[0]] # initialize as zero
    for key2 in hop2:
        if key2 not in keys: keys.append(key2) # store
    for key in keys:
        if key in hop1: m1 = hop1[key]
        else: m1 = zero1
        if key in hop2: m2 = hop2[key]
        else: m2 = zero2
        out[key] = direct_sum([m1,m2]) # store
    return out
