from __future__ import print_function

import numpy as np
from . import geometry
from .hamiltonians import sx,sy,sz
from scipy.sparse import csc_matrix

def hamiltonian(mw=1.0):
  """Return the Hamiltonian for the surface of a 3d TI"""
  g = geometry.single_square_lattice() # get the geometry
  return geometry2hamiltonian(g,mw=mw)

  h = g.get_hamiltonian() # initialize the Hamiltonian
  h.tx = 1j*sy.todense() # update hopping 
  h.ty = -1j*sx.todense() # update hopping 
  # Now add the Wilson mass term
  h.intra = 2*mw*sz.todense() # update hopping 
  h.tx -= mw/2.*sz.todense() # update hopping 
  h.ty -= mw/2.*sz.todense() # update hopping 
  return h # return the Hamiltonian


def geometry2hamiltonian(g,mw=0.6,is_sparse=False):
  """Given an input geometry, generate the Hamiltonian"""


  ## Only valid for square lattice ###
  tol = 0.001 # tolerancy
  ons = csc_matrix(mw*np.matrix(np.identity(2,dtype=np.complex))) # onsite
  ons = csc_matrix(2.*sz*mw) # onsite
  tx = csc_matrix(1j*sy-mw/2.*sz) # hopping in x direction
  ty = csc_matrix(-1j*sx-mw/2.*sz) # hopping in y direction
  dx = np.array([1.,0.,0.])
  dy = np.array([0.,1.,0.])
  def f(r1,r2): # function the returns the hopping
    dr = r2-r1 # distance
    if dr.dot(dr)<tol: return ons # onsite
    elif (dr-dx).dot(dr-dx)<tol: return tx # onsite
    elif (dr-dy).dot(dr-dy)<tol: return ty # onsite
    elif (dr+dx).dot(dr+dx)<tol: return tx.H # onsite
    elif (dr+dy).dot(dr+dy)<tol: return ty.H # onsite
    else: return ons*0.

  h = g.get_hamiltonian(fun=f,spinful_generator=True) # generate 
  return h
  


