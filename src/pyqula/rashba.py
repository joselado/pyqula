import numpy as np
from scipy.sparse import coo_matrix
from .rotate_spin import sx,sy,sz
from . import neighbor

# dense 2x2 Pauli matrices, converted once at import time: rashba_matrix
# below needs one 2x2 spin block per bond (potentially hundreds of
# thousands of them for a large flake), and building each one out of
# scipy-sparse sx/sy/sz arithmetic pays scipy's per-call sparse-matrix
# bookkeeping (format checks, index-dtype inference, prune...) every
# single bond; a plain numpy array has none of that overhead and lets all
# bonds be combined into a handful of vectorized array operations instead
# of a Python loop of scipy calls.
_sx = np.asarray(sx.todense())
_sy = np.asarray(sy.todense())
_sz = np.asarray(sz.todense())

def add_rashba(self,c):
    """
    Add rashba to a Hamiltonian
    """
    from .operators import isnumber
    if isnumber(c):
        if c==0.0: return
    if not self.has_spin: # no spin degree of freedom
        self.turn_spinful() # spinful hamiltonian
    def rashba(*args,**kwargs):
        return self.spinful2full(rashba_matrix(*args,**kwargs))
    g = self.geometry
    is_sparse = self.is_sparse # saprse Hamiltonian
    self.intra = self.intra + rashba(g.r,c=c,is_sparse=is_sparse)
    if self.dimensionality==0: return
    if self.is_multicell: # multicell hamiltonians
      for i in range(len(self.hopping)): # loop over hoppings
        d = self.hopping[i].dir # direction
        Rd = g.a1*d[0] + g.a2*d[1] + g.a3*d[2]
        r2 = [ir + Rd for ir in g.r] # new coordinates
        self.hopping[i].m = self.hopping[i].m + rashba(g.r,r2=r2,c=c,is_sparse=is_sparse)
    else: # conventional Hamiltonians
      if g.dimensionality==1:  # one dimensional
        r2 = [ir + g.a1 for ir in g.r]
        self.inter = self.inter + rashba(g.r,r2=r2,c=c,is_sparse=is_sparse)
      elif g.dimensionality==2:  # two dimensional
        r2 = [ir + g.a1 for ir in g.r]
        self.tx = self.tx + rashba(g.r,r2=r2,c=c,is_sparse=is_sparse)
        r2 = [ir + g.a2 for ir in g.r]
        self.ty = self.ty + rashba(g.r,r2=r2,c=c,is_sparse=is_sparse)
        r2 = [ir + g.a1+g.a2 for ir in g.r]
        self.txy = self.txy + rashba(g.r,r2=r2,c=c,is_sparse=is_sparse)
        r2 = [ir + g.a1-g.a2 for ir in g.r]
        self.txmy = self.txmy + rashba(g.r,r2=r2,c=c,is_sparse=is_sparse)
      else: raise








def rashba_matrix(r1,r2=None,c=0.,d=[0.,0.,1.],is_sparse=False):
  """
  Add Rashba coupling, returns a spin polarized matrix
  This will assume only Rashba between first neighbors

  Only first-neighbor bonds ever contribute (Rashba is a nearest-neighbor
  term), so this builds the (2*nat1,2*nat2) sparse matrix directly from
  that (typically O(nat)) list of bonds, instead of allocating an
  nat1 x nat2 Python list of 2x2 blocks and handing it to scipy.sparse.bmat
  regardless of how few of those blocks are actually nonzero -- the old
  approach needed nat1*nat2 Python object slots up front (~1e10 for
  nat~1e5), infeasible in both time and memory well before bmat even ran.
  """
  if r2 is None:
    r2 = r1
  r1 = np.array(r1)
  r2 = np.array(r2)
  nat1 = len(r1)
  nat2 = len(r2)
  pairs = neighbor.find_first_neighbor(r1,r2) # (i,j) first-neighbor bonds
  if len(pairs)==0:
    m = coo_matrix(([],([],[])),shape=(2*nat1,2*nat2),dtype=np.complex128)
    if not is_sparse: m = m.todense()
    return m
  ii,jj = pairs[:,0],pairs[:,1]
  rij = r2[jj] - r1[ii] # (nbonds,3), one bond vector per row
  dx,dy,dz = rij[:,0],rij[:,1],rij[:,2] # (nbonds,)
  # cross product with the Pauli vector, for every bond at once:
  # (nbonds,2,2)
  rxs0 = dy[:,None,None]*_sz - dz[:,None,None]*_sy
  rxs1 = dz[:,None,None]*_sx - dx[:,None,None]*_sz
  rxs2 = dx[:,None,None]*_sy - dy[:,None,None]*_sx
  ms = 1j*(d[0]*rxs0 + d[1]*rxs1 + d[2]*rxs2) # (nbonds,2,2)
  if callable(c): # position-dependent strength: one python call per bond
    mid = (r1[ii]+r2[jj])/2.
    cvals = np.array([c(p) for p in mid],dtype=np.complex128)
    s = ms*cvals[:,None,None]
  else: s = ms*c # constant strength: one vectorized multiply
  a_idx = np.array([0,0,1,1])
  b_idx = np.array([0,1,0,1])
  rows = (2*ii[:,None] + a_idx[None,:]).reshape(-1)
  cols = (2*jj[:,None] + b_idx[None,:]).reshape(-1)
  data = s[:,a_idx,b_idx].reshape(-1)
  keep = data!=0.0
  m = coo_matrix((data[keep],(rows[keep],cols[keep])),
          shape=(2*nat1,2*nat2),dtype=np.complex128)
  if not is_sparse: m = m.todense() # to normal matrix
  return m


