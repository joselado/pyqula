import numpy as np
from scipy.sparse import coo_matrix,bmat
from .rotate_spin import sx,sy,sz
from . import neighbor

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
  """
  zero = coo_matrix([[0.,0.],[0.,0.]])
  if r2 is None:
    r2 = r1
  nat = len(r1) # number of atoms
  m = [[None for i in range(nat)] for j in range(nat)] # create matrix
  for i in range(nat): m[i][i] = zero.copy() # initilize
  neighs = neighbor.connections(r1,r2) # neighbor connections
  for i in range(nat): # loop over first atoms
#    for j in range(nat):  # loop over second atoms
    for j in neighs[i]:  # loop over second atoms
      rij = r2[j] - r1[i]   # x component
      dx,dy,dz = rij[0],rij[1],rij[2]  # different components
      rxs = [dy*sz-dz*sy,dz*sx-dx*sz,dx*sy-dy*sx]  # cross product
      ms = 1j*(d[0]*rxs[0] + d[1]*rxs[1] + d[2]*rxs[2]) # E dot r times s
      s = 0.0*ms
      if 0.9<(rij.dot(rij))<1.1: # if nearest neighbor
        if callable(c): s = ms*c((r1[i]+r2[j])/2.) # function
        else: s = ms*c # multiply
      m[i][j] = s # rashba term
  if not is_sparse: m = bmat(m).todense()  # to normal matrix
  else: m = bmat(m) # sparse matrix
  return m


