import numpy as np


def kchain(h,k):
  """ Return the kchain Hamiltonian """
  if h.dimensionality != 2: raise
  if h.is_multicell: h = h.get_no_multicell() # redefine
  tky = h.ty*np.exp(1j*np.pi*2.*k)
  tkxy = h.txy*np.exp(1j*np.pi*2.*k)
  tkxmy = h.txmy*np.exp(-1j*np.pi*2.*k)  # notice the minus sign !!!!
  # chain in the x direction
  ons = h.intra + tky + np.conjugate(tky).T  # intra of k dependent chain
  hop = h.tx + tkxy + tkxmy  # hopping of k-dependent chain
  return (ons,hop)


