from __future__ import print_function
from copy import deepcopy
from scipy.sparse import bmat
from scipy.sparse import csc_matrix as csc
import numpy as np
from . import sculpt

def honeycomb2square(h):
  """Transforms a honeycomb lattice into a square lattice"""
  ho = deepcopy(h) # output geometry
  g = h.geometry # geometry
  go = deepcopy(g) # output geometry
  go.a1 = - g.a1 - g.a2 
  go.a2 = g.a1 - g.a2
  # perform a check to see if the supercell is well built
  if np.abs(go.a1)[1]>0.001 or np.abs(go.a2[0])>0.001: raise
  go.x = np.concatenate([g.x,g.x-g.a1[0]])
  go.y = np.concatenate([g.y,g.y-g.a1[1]])
  go.z = np.concatenate([g.z,g.z])
  go.xyz2r() # update r atribbute
  zero = csc(h.tx*0.)
  # define sparse
  intra = csc(h.intra)
  tx = csc(h.tx)
  ty = csc(h.ty)
  txy = csc(h.txy)
  txmy = csc(h.txmy)
  # define new hoppings
  ho.intra = bmat([[intra,tx.H],[tx,intra]]).todense()
  ho.tx = bmat([[txy.H,zero],[ty.H,txy.H]]).todense()
  ho.ty = bmat([[txmy,ty.H],[txmy,zero]]).todense()
  ho.txy = bmat([[zero,None],[None,zero]]).todense()
  ho.txmy = bmat([[zero,zero],[tx.H,zero]]).todense()
  ho.geometry = go
  return ho



def invert_axis2(h):
  """Changes one axis in the hamiltonian"""
  ho = h.copy()
  ho.geometry.a2 = -h.geometry.a2
  ho.ty = h.ty.H
  ho.txy = h.txmy
  ho.txmy = h.txy
  return ho


def turn_square(h):
  """Turn the hamiltonian square no matter what!!!"""
  try: 
    return honeycomb2squareMoS2(h)
  except:
    ho = invert_axis2(h)
    return honeycomb2squareMoS2(ho)



def honeycomb2squareMoS2(h,check=True):
  """Transforms a honeycomb lattice into a square lattice"""
  ho = deepcopy(h) # output geometry
  g = h.geometry # geometry
  go = deepcopy(g) # output geometry
  go.a1 = g.a1 + g.a2 
  go.a2 = g.a1 - g.a2
  # now put the first vector along the x axis
  go.r = np.concatenate([g.r,g.r + g.a1])
  go.r2xyz()
  go = sculpt.rotate_a2b(go,go.a1,np.array([1.,0.,0.]))
  # perform a check to see if the supercell is well built  
  a1a2 = go.a1.dot(go.a2)
#  if a1a2>0.001: 
  if np.abs(go.a1)[1]>0.001 or np.abs(go.a2[0])>0.001: 
    ang = go.a1.dot(go.a2)
    print("The projection of lattice vectors is",ang)
    if check: raise
  go.r2xyz() # update r atribbute
  zero = csc(h.tx*0.)
  # define sparse
  intra = csc(h.intra)
  tx = csc(h.tx)
  ty = csc(h.ty)
  txy = csc(h.txy)
  txmy = csc(h.txmy)
  # define new hoppings
  ho.intra = bmat([[intra,tx],[tx.H,intra]]).todense()
  ho.tx = bmat([[txy,zero],[ty,txy]]).todense()
  ho.ty = bmat([[txmy,zero],[ty.H,txmy]]).todense()
  ho.txy = bmat([[zero,zero],[tx,zero]]).todense()
  ho.txmy = bmat([[zero,zero],[zero,zero]]).todense()
  ho.geometry = go
  return ho


















def bulk2ribbon(h,n=10):
  """Converts a hexagonal bulk hamiltonian into a ribbon hamiltonian"""
  from ribbonizate import hamiltonian_bulk2ribbon as b2r
  return b2r(h,n=n)  # workaround so that old script work














def bulk2ribbon_zz(h,n=10):
  """Converts a hexagonal bulk hamiltonian into a ribbon hamiltonian"""
  h = honeycomb2square(h) # create a square 2d geometry
  go = h.geometry.copy() # copy geometry
  ho = h.copy() # copy hamiltonian
  ho.dimensionality = 1 # reduce dimensionality
  go.dimensionality = 1 # reduce dimensionality
  intra = [[None for i in range(n)] for j in range(n)]
  inter = [[None for i in range(n)] for j in range(n)]
  for i in range(n): # to the the sam index
    intra[i][i] = csc(h.intra) 
    inter[i][i] = csc(h.tx) 
  for i in range(n-1): # one more or less
    intra[i][i+1] = csc(h.ty)  
    intra[i+1][i] = csc(h.ty.H)  
    inter[i+1][i] = csc(h.txmy) 
    inter[i][i+1] = csc(h.txy) 
  ho.intra = bmat(intra).todense()
  ho.inter = bmat(inter).todense()
  return ho




def orthogonalize_geometry(g):
  go = deepcopy(g) # output geometry
  go.a1 = g.a1 + g.a2
  go.a2 = g.a1 - g.a2
  # now put the first vector along the x axis
  go.r = np.concatenate([g.r,g.r + g.a1])
  go.r2xyz()
  go = sculpt.rotate_a2b(go,go.a1,np.array([1.,0.,0.]))
  # perform a check to see if the supercell is well built  
  a1a2 = go.a1.dot(go.a2)
#  if a1a2>0.001: 
  if np.abs(go.a1)[1]>0.001 or np.abs(go.a2[0])>0.001:
    ang = go.a1.dot(go.a2)
    print("The projection of lattice vectors is",ang)
    if check: raise
  go.has_sublattice = False
  go.r2xyz() # update r atribbute
  return go

