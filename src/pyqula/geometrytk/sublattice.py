import numpy as np

info = False

def get_sublattice(rs):
  """Return indexes of the sublattice, assuming that there is sublattice"""
  n = len(rs) # loop over positions
  sublattice = [0 for i in range(n)] # list for the sublattice
  ii = np.random.randint(n)
  sublattice[ii] = -1 # start with this atom
  if info: print("Looking for a sublattice")
  while True: # infinite loop
    for i in range(n): # look for a neighbor for site i
      if sublattice[i]!=0: continue # already assigned
      for j in range(n): # loop over the rest of the atoms
        if sublattice[j]==0: continue # next one
        dr = rs[i] - rs[j] # distance to site i
        if 0.9<dr.dot(dr)<1.01: # if NN and sublattice 
          sublattice[i] = -sublattice[j] + 0 # opposite
          continue # next one
    if np.min(np.abs(sublattice))!=0: break
  return sublattice


