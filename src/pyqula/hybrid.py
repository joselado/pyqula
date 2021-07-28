# library with specific function for hybrid ribbons
from __future__ import print_function
import numpy as np




def create_hybrid(h1,h2,coupling=1.0):
  """ Creates a hybrid hamiltonian from the two inputs"""
  from copy import deepcopy
  hy = deepcopy(h1) # copy hamiltonian
  # check that thw two hamiltonians are mixable
  if not h1.has_spin == h2.has_spin:  # spinpol
    raise
  if not h1.has_eh == h2.has_eh:  # electron hole
    raise
  if not len(h1.intra) == len(h2.intra):  # same dimension
    print("Wrong dimensions", len(h1.intra),len(h2.intra))
    raise
  if not h1.dimensionality == h2.dimensionality:  # dimension
    raise
  dd = h1.intra.shape[0] # dimension of the hamiltonian
  # we wnt a matrix like
  # ( h1_1   t12  )
  # ( t21   h2_2  )
  # substitute the block h2_2 by the second block in input h2
  for i in range(dd/2,dd):
    for j in range(dd/2,dd):
      hy.intra[i,j] = h2.intra[i,j] # substitute element
      hy.inter[i,j] = h2.inter[i,j] # substitute element
  # and now create t12 by mixing 
  for i in range(dd/2):
    for j in range(dd/2,dd):
      hy.intra[i,j] = coupling*(h1.intra[i,j]+h2.intra[i,j])/2.0 # substitute element
      hy.inter[i,j] = coupling*(h1.inter[i,j]+h2.inter[i,j])/2.0 # substitute element
  for i in range(dd/2,dd):
    for j in range(dd/2):
      hy.intra[i,j] = coupling*(h1.intra[i,j]+h2.intra[i,j])/2.0 # substitute element
      hy.inter[i,j] = coupling*(h1.inter[i,j]+h2.inter[i,j])/2.0 # substitute element
  hy.check()  # check that everything is fine
  return hy  # return the hamiltonian




def half_and_half(h1,h2,fun=None,tlen=0.001,direction=1):
  """Create a Hamiltonian using half one matrix and half of the
  other"""
#  if h1.dimensionality ==0: raise # only for 1d
  if fun is None: # no function provided
    def fun(r):
      if r[direction]<-tlen: return 0.0
      elif r[direction]>tlen: return 1.0
      else: return (r[1]+tlen)/(2*tlen) # interpolate
  hout = h1.copy() # copy Hamiltonians
  d = 1
  if h1.has_spin: d *= 2 # double
  if h1.has_eh: d *= 2 # double
  for i in range(h1.intra.shape[0]):
    for j in range(h1.intra.shape[0]):
      fac = fun((h1.geometry.r[i//d]+h2.geometry.r[j//d])/2) # factor
      hout.intra[i,j] = fac*(h1.intra[i,j]) + (1.-fac)*h2.intra[i,j]
      if h1.is_multicell: # multicell Hamiltonian
        for ii in range(len(h1.hopping)):
          hout.hopping[ii].m[i,j] = fac*(h1.hopping[ii].m[i,j]) + (1.-fac)*h2.hopping[ii].m[i,j]
        
      else:
        if h1.dimensionality==1: # one dimensional
          hout.inter[i,j] = fac*(h1.inter[i,j]) + (1.-fac)*h2.inter[i,j]
        else: raise
  return hout




def heterostructure(h1,h2):
  """Build a heterostructure, assuming that the only difference
  are onsite matrices"""
  h = h1.copy() # copy Hamiltonian
  n = h.intra.shape[0]//2 # retain these elements
  for i in range(n):
    for j in range(n):
      h.intra[i,j] = h2.intra[i,j]
  return h # return Hamiltonian









