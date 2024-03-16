import numpy as np


def bulk2ribbon(hin,n=10):
  """ Create a ribbon hamiltonian object"""
  if hin.dimensionality!=2: raise
  hin = hin.get_multicell() # into multicell form
  hr = hin.get_supercell(nsuper=[1,n,1]) # Hamiltonian of the ribbon
  hr.geometry.dimensionality = 1 # one dimensional
  hr.dimensionality = 1 # one dimensional
  out = [] # empty list
  for i in range(len(hr.hopping)): # loop over hoppings
      d = hr.hopping[i].dir # direction
      d = np.array(d) # as array
      if np.max(np.abs(d[1:]))==0: # hopping in x direction
          out.append(hr.hopping[i].copy())
  hr.hopping = out # overwrite the hoppings
  return hr



def bulk2film(hin,n=10):
  """ Create a ribbon hamiltonian object"""
  if hin.dimensionality!=3: raise
  hin = hin.get_multicell() # into multicell form
  hr = hin.get_supercell(nsuper=[1,1,n]) # Hamiltonian of the ribbon
  hr.geometry.dimensionality = 2 # two dimensional
  hr.dimensionality = 2 # two dimensional
  out = [] # empty list
  for i in range(len(hr.hopping)): # loop over hoppings
      d = hr.hopping[i].dir # direction
      d = np.array(d) # as array
      if np.max(np.abs(d[2:]))==0: # hopping in x,y direction
          out.append(hr.hopping[i].copy())
  hr.hopping = out # overwrite the hoppings
  return hr




