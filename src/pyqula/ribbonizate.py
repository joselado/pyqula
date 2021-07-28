from __future__ import print_function
import numpy as np

def bulk2ribbon(g, n=5):
  """ Transformas a 2D geometry into a ribbon"""
  if not g.dimensionality == 2: raise # has to be two dimensional
  if not np.abs(g.a1.dot(g.a2)) < 0.0001: raise # has to be orthogonal
  go = g.copy() # create new geometry
  go.dimensionality = 1 # ribbon
  go.x = []
  go.y = []
  go.z = []
  for i in range(n):
    go.x += (g.x).tolist() # append x
    go.z += (g.z).tolist() # append x
    go.y += (g.y+i*g.a2[1]).tolist() # append x
  go.x = np.array(go.x)
  go.y = np.array(go.y)
  go.z = np.array(go.z)
  go.xyz2r()
  go.celldis = g.a1[0]
  go.center()
  return go


def reflect(g):
  """ Reflects a certain geometry with respect to the origin"""
  g.y = g.y - min(g.y) # move to the zero
  go = g.copy()
  go.x = (g.x).tolist() + (g.x).tolist()
  go.y = (g.y).tolist() + (-g.y).tolist()
  go.z = (g.z).tolist() + (g.z).tolist()
  go.x = np.array(go.x)
  go.y = np.array(go.y)
  go.z = np.array(go.z)
  go.xyz2r()
  go = remove_repeated(go) # remove the repeted atoms
  return go  




def remove_repeated(g):
  """ Remove repeated coordinates"""
  go = g.copy() # copy geometry
  go.r = [] # empty list
  for i in range(len(g.r)): # only accept
    unique = True
    for j in range(len(go.r)):
      if i!=j:
        dr = g.r[i] -g.r[j]
        dr = dr.dot(dr) # distance
        if dr<0.2:
          unique=False
    if unique:
      go.r.append(g.r[i])
    else:
      print("Repeated",i)
  go.r = np.array(go.r)
  go.r2xyz()
  return go


 

def hamiltonian_bulk2ribbon(h,n=10,sparse=False,check=True):
  """Converts a hexagonal bulk hamiltonian into a ribbon hamiltonian"""
  from scipy.sparse import csc_matrix as csc
  from scipy.sparse import bmat
  go = h.geometry.copy() # copy geometry
  ho = h.copy() # copy hamiltonian
#  if np.abs(go.a1.dot(go.a2))>0.01: 
#    if check: raise # if axis non orthogonal
  ho.dimensionality = 1 # reduce dimensionality
  ho.geometry.dimensionality = 1 # reduce dimensionality
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
  if sparse:
    ho.intra = bmat(intra)
    ho.inter = bmat(inter)
    ho.is_sparse = True # hamiltonian is sparse
  else:
    ho.intra = bmat(intra).todense()
    ho.inter = bmat(inter).todense()
  # calculate the angle 
  import sculpt
  ho.geometry = sculpt.rotate_a2b(ho.geometry,ho.geometry.a1,np.array([1.,0.,0.]))
  ho.geometry = h.geometry.supercell((1,n)) # create supercell
  ho.geometry.dimensionality = 1
  ho.geometry.a1 = h.geometry.a1 # add the unit cell vector
  ho.dimensionality = 1
  # for geometries with names
  if ho.geometry.atoms_have_names:
    ho.geometry.atoms_names = ho.geometry.atoms_names*n
  return ho


