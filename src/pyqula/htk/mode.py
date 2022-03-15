# library to verify the type of Hamiltonian
import numpy as np

def check_mode(h,n):
    if n=="spinless_nambu":
        if (not h.has_spin) and h.has_eh: return True
        else: return False
    elif n=="spinful_nambu":
        if h.has_spin and h.has_eh: return True
        else: return False
    elif n=="spinful":
        if h.has_spin and (not h.has_eh): return True
        else: return False
    elif n=="spinless":
        if (not h.has_spin) and (not h.has_eh): return True
        else: return False
    else: raise



def reduce_hamiltonian(self):
    """Try to reduce the dimensionality of the Hamiltonian"""
    if self.check_mode("spinless"): return self # nothing to do
    elif self.check_mode("spinful"): # spinful Hamiltonian
        h = self.copy() # copy Hamiltonian
        h.remove_spin() # remove the spin degree of freedom
        h0 = h.copy() # copy Hamiltonian
        h.turn_spinful() # add spin degree of freedom
        if self.same_hamiltonian(h): 
#            print("Hamiltonian has become spinless")
            return h0 # return the spinless one
        else: return self # return the spinful one
        return self # nothing to do
    else: return self # nothing to do


def same_hamiltonian(self,h,ntries=10):
    """Check if two hamiltonians are the same"""
    if h.dimensionality==0: ntries = 1
    hk1 = self.get_hk_gen()
    hk2 = h.get_hk_gen()
    for i in range(ntries):
      k = np.random.random(3)
      m = hk1(k) - hk2(k)
      if np.max(np.abs(m))>0.000001: return False
    return True


def turn_spinful(self,enforce_tr=False):
    """Turn the hamiltonian spinful"""
    if self.has_spin: return # already spinful
    if self.is_sparse: # sparse Hamiltonian
      self.turn_dense() # dense Hamiltonian
      self.turn_spinful(enforce_tr=enforce_tr) # spinful
      self.turn_sparse()
    else: # dense Hamiltonian
      from ..increase_hilbert import spinful
      def fun(m):
          if enforce_tr: return spinful(m,np.conjugate(m))
          else: return spinful(m)
      self.modify_hamiltonian_matrices(fun) # modify the matrices
      self.has_spin = True # set spinful


def make_compatible(h1,h2):
    """Make a Hamiltonian h1 compatible with hamiltonian h2"""
    ho = h1.copy() # copy Hamiltonian
    if h1.has_spin: ho.turn_spinful() # make spinful
    if h2.has_eh: ho.turn_nambu() # add e-h symmetry
    return ho


