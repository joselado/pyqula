from __future__ import print_function
import numpy as np


def anderson(h,w=0.0,p=1.0):
  """Return a Hamiltonian with Anderson disorder"""
  def fdis(r):
      if np.random.random()<p: # probability of an impurity
        return (np.random.random() - .5)*2*w # disorder
      return 0.0
  h.add_onsite(fdis)
  return h


def phase(h,w=0.0):
  """Random phase disorder"""
  if h.has_eh: raise # not yet
  if h.has_spin: raise # not yet
  ho = h.copy() # copy the Hamiltonian
  n = h.intra.shape[0] # dimension
  cs = np.array(np.random.random((n,n)) - .5)*2*w # disorder
  cs = (cs - cs.T)/2. # Hermitian
  cs = np.exp(1j*np.pi*cs) # phases
  ho.intra = np.matrix(np.array(ho.intra)*np.array(cs))
  return ho


