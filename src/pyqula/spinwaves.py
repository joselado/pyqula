
# routines for dealing with spin models, using the Holfstein Primakov
# transformation

from . import neighbor
import numpy as np
from scipy.sparse import csc_matrix,coo_matrix
from scipy.sparse import identity as sparseiden

class SpinModel():
  def __init__(self,g):
    self.geometry = g # store the geometry
    self.dimensionality = g.dimensionality
    self.spins = np.array([1. for r in g.r]) # ones
    self.magnetization = np.array([[0.,0.,1.] for r in g.r]) # ones
  def hp_heisenberg(self,fun=None,d=None,k=None):
    return hp_heisenberg(self,fun=fun,d=d,k=k)
  def szsz(self,fun=None):
    return szsz(self,fun=fun)
  def setup_exchange(self):
    h = self.get_hamiltonian(has_spin=False) # generate a hamiltonian
    class Exchange(): pass
    self.exchange = Exchange() 
    self.exchange.intra = h.intra
    if self.dimensionality==0: pass
    elif self.dimensionality==1:
      self.exchange.inter = h.inter
    elif self.dimensionality==2:
      self.exchange.tx = h.tx
      self.exchange.ty = h.ty
      self.exchange.txy = h.txy
      self.exchange.txmy = h.txmy
    else: raise

 

def set_couplings(g,f=None):
  """Set the exchange couplings"""
  js = []
  xs = []
  ys = []
  if callable(f): # callable function
    m = neighbor.parametric_hopping(g.r,g.r)
  else: raise
  return m # return pairs

def xyj(m):
  from scipy.sparse import coo_matrix
  m2 = coo_matrix(m)
  return (m.rows,m.cols,m.data)


def hp_heisenberg(sm,fun=None,d=None,k=None):
  """Perform the Holfstein-Primakov for this spin model"""
  sm.hamiltonian = sm.geometry.get_hamiltonian(has_spin=False,fun=fun) # initialize
  spins = sm.spins # spins
  ns = len(spins) # number of sites
  inds = [i for i in range(ns)]
  def c2h(m0):
    def genij(i,j): return np.identity(3)*m0[i,j]
    return sites2coupling(genij,spins)
  # intra term
  for i in range(sm.hamiltonian.intra.shape[0]): 
    sm.hamiltonian.intra[i,i] = 0.0 # set coupling to zero
  (mons,mhop) = c2h(sm.hamiltonian.intra)
  if d is not None: # add the uniaxial anisotropy
    try: # for an array
      d[0]
      ds = d
    except: # for a number
      ds = [d for i in range(ns)] # loop over spins
    mani =  csc_matrix((ds,(inds,inds)),shape=(ns,ns)).todense()
    mons = mons + mani # add anisotropy 
  sm.hamiltonian.intra = mons + mhop
  # now add the anisotropic exchange
  if k is not None:
    hk = sm.geometry.get_hamiltonian(has_spin=False,fun=k) # initialize
    (mons,mhop) = c2h(hk.intra) # anisotropic part
    sm.hamiltonian.intra = sm.hamiltonian.intra + mons

  if sm.dimensionality == 0: pass # one dimensional
  elif sm.dimensionality == 1: # one dimensional
    (mons,mhop) = c2h(sm.hamiltonian.inter)
    (mons2,mhop2) = c2h(sm.hamiltonian.inter.H)
    sm.hamiltonian.inter = mhop
    sm.hamiltonian.intra += mons + mons2
  elif sm.dimensionality == 2: # one dimensional
    for name in ["tx","ty","txy","txmy"]: # loop over attributes
      (mons,mhop) = c2h(getattr(sm.hamiltonian,name))
      (mons2,mhop2) = c2h(getattr(sm.hamiltonian,name).H)
      setattr(sm.hamiltonian,name,mhop) # set hopping
      sm.hamiltonian.intra += mons + mons2 # add to onsite
  else: raise
  return sm
    



def sites2coupling(genij,spins):
  """Create the couplings of the HP-hamiltonian"""
  # SzSz terms
  # There are two matrices, one that renormalizes onsite energies
  # and another one that creates hoppings
  n = len(spins)
  ons = np.zeros((n,n),dtype=np.complex) # initial
  hop = np.zeros((n,n),dtype=np.complex) # initial
  iden = np.identity(n,dtype=np.complex) # identity
  for i in range(n): # loop over spins
    for j in range(n): # loop over spins
      cij = genij(i,j) # matrix with the couplings SiSj
      # the quantization axis is z
      ons[i,i] += cij[2,2] # Sz Sz component
      # there are missing terms!!!!!!
  # S+ S- terms
      sij = np.sqrt(spins[i]*spins[j]) # denominator
      hop[i,j] = -(cij[0,0] + cij[1,1])*sij/2.
  return (np.matrix(ons),np.matrix(hop)) # return matrices










def szsz(sm,fun=None):
  """Perform the Holfstein-Primakov for this spin model"""
  sm.hamiltonian = sm.geometry.get_hamiltonian(has_spin=False,fun=fun) # initialize
  spins = sm.spins # spins
  ns = len(spins) # number of sites
  inds = [i for i in range(ns)]
  def c2h(m0):
    def genij(i,j): return np.identity(3)*m0[i,j]
    return sites2coupling(genij,spins)
  # intra term
  for i in range(sm.hamiltonian.intra.shape[0]): 
    sm.hamiltonian.intra[i,i] = 0.0 # set coupling to zero
  (mons,mhop) = c2h(sm.hamiltonian.intra)
  mout = mons.copy() # copy matrix
  if sm.dimensionality == 0: pass # one dimensional
  elif sm.dimensionality == 1: # one dimensional
    (mons,mhop) = c2h(sm.hamiltonian.inter)
    (mons2,mhop2) = c2h(sm.hamiltonian.inter.H)
    mout += mons + mons2
  elif sm.dimensionality == 2: # one dimensional
    for name in ["tx","ty","txy","txmy"]: # loop over attributes
      (mons,mhop) = c2h(getattr(sm.hamiltonian,name))
      (mons2,mhop2) = c2h(getattr(sm.hamiltonian,name).H)
      mout += mons + mons2
  else: raise
  return mout  # return onsite matrix
    





def sites2coupling_sparse(mij,spins):
  """Create the couplings of the HP-hamiltonian"""
  # SzSz terms
  # There are two matrices, one that renormalizes onsite energies
  # and another one that creates hoppings
  n = len(spins)
  ons = coo_matrix(([],([],[])),shape=(n,n),dtype=np.complex)
  hop = coo_matrix(([],([],[])),shape=(n,n),dtype=np.complex)
  iden = sparseiden(n,dtype=np.complex) # identity matrix
  raise # not finished
  
  for i in range(n): # loop over spins
    for j in range(n): # loop over spins
      cij = genij(i,j) # matrix with the couplings SiSj
      # the quantization axis is z
      ons[i,i] += cij[2,2] # Sz Sz component
      # there are missing terms!!!!!!
  # S+ S- terms
      sij = np.sqrt(spins[i]*spins[j]) # denominator
      hop[i,j] = -(cij[0,0] + cij[1,1])*sij/2.
  return (np.matrix(ons),np.matrix(hop)) # return matrices









