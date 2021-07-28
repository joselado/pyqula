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



