import numpy as np


def merge_channels(h1,h2):
    """Merge the spin up and down channels"""
    if h1.has_spin or h2.has_spin: raise
    if h1.has_eh or h2.has_eh: raise
    h1 = h1.get_multicell() # multicell Hamiltonian
    h2 = h2.get_multicell() # multicell Hamiltonian
    if len(h1.hopping)!=len(h2.hopping): raise
    h = h1.copy() # copy Hamiltonian
    h.turn_spinful() # make spinful
    h.turn_dense()
    h1.turn_dense()
    h2.turn_dense()
    h.clean()
    if h.has_eh: raise # not implemented
    n = len(h.geometry.r) # number of sites
    for i in range(n): # loop over positions
      for j in range(n): # loop over positions
        h.intra[2*i,2*j] = h2.intra[i,j] # set that spin channel
        h.intra[2*i+1,2*j+1] = h1.intra[i,j] # set that spin channel
    for k in range(len(h.hopping)): # loop
        d = np.array(h.hopping[k].dir)
        d1 = np.array(h1.hopping[k].dir)
        d2 = np.array(h2.hopping[k].dir)
        dd1 = d-d1
        dd2 = d-d1
        if dd1.dot(dd1)>0.001 or dd2.dot(dd2)>0.001: raise
        for i in range(n): # loop over positions
          for j in range(n): # loop over positions
            h.hopping[k].m[2*i,2*j] = h2.hopping[k].m[i,j]
            h.hopping[k].m[2*i+1,2*j+1] = h1.hopping[k].m[i,j]
    return h




