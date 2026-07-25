from __future__ import print_function
from scipy.sparse import coo_matrix
from .rotate_spin import sx,sy,sz
import numpy as np
from numba import jit
from . import parallel

from .algebra import isnumber

# dense 2x2 Pauli matrices, converted once at import time: building each
# bond's 2x2 spin block via scipy-sparse sx/sy/sz arithmetic (as this used
# to be written) pays scipy's per-call sparse bookkeeping (format checks,
# index-dtype inference, prune...) on every single bond -- see rashba.py
# for the same fix and the profiling that motivated it.
_sx = np.asarray(sx.todense())
_sy = np.asarray(sy.todense())
_sz = np.asarray(sz.todense())


def generalized_kane_mele(r1,r2,rm,fun=0.0,tol=1e-5):
    """Return the Kane-Mele generalized Hamiltonian.

    Only second-neighbor bonds mediated by a real first-neighbor path
    contribute, so this builds the (2*nsites,2*nsites) sparse matrix
    directly from that (typically O(nsites)) bond list, found via a
    KD-tree batch query (find_close_neighbors_batch, O(N log N) total)
    instead of one O(len(rm))/O(len(r2)) scan per site (O(N^2) total) --
    and never allocates an nsites x nsites Python list of blocks for
    scipy.sparse.bmat regardless of sparsity, which used to be infeasible
    in time and memory for nsites beyond a few thousand.
    """
    if fun==0.0: return 0
    if isnumber(fun): kmfun = lambda r: fun # function that always returns fun
    elif callable(fun): kmfun = fun # callable function
    else: raise # no idea
    nsites = len(r1) # number of sites
    rm = np.array(rm)
    r1 = np.array(r1)
    r2 = np.array(r2)
    from .neighbor import find_close_neighbors_batch
    indsim_all = find_close_neighbors_batch(r1,rm,d=1.1) # candidate intermediate sites, per i
    indsj_all = find_close_neighbors_batch(r1,r2,d=4.1) # candidate final sites, per i
    ii,jj,urx,ury,urz,cvals = [],[],[],[],[],[]
    for i in range(nsites): # loop over initial site
        indsim = indsim_all[i]
        if len(indsim)==0: continue
        rmi = rm[indsim] # retain only those sites for the loop
        for j in indsj_all[i]: # loop over final site
            dr = r1[i]-r2[j] # difference
            if dr.dot(dr)>=4.1: continue # not close enough
            ur = km_vector(r1[i],r2[j],rmi,tol=tol) # kane mele vector
            if ur[0]==0.0 and ur[1]==0.0 and ur[2]==0.0: continue # no bond path found
            r3 = (r1[i] + r2[j])/2.0
            ii.append(i); jj.append(j)
            urx.append(ur[0]); ury.append(ur[1]); urz.append(ur[2])
            cvals.append(kmfun(r3))
    if len(ii)==0: return coo_matrix((2*nsites,2*nsites),dtype=np.complex128)
    ii = np.array(ii); jj = np.array(jj)
    urx = np.array(urx); ury = np.array(ury); urz = np.array(urz)
    cvals = np.array(cvals,dtype=np.complex128)
    # cross-product-with-sigma * coupling strength, for every bond at once
    sm = 1j*(urx[:,None,None]*_sx + ury[:,None,None]*_sy
            + urz[:,None,None]*_sz)*cvals[:,None,None] # (nbonds,2,2)
    a_idx = np.array([0,0,1,1])
    b_idx = np.array([0,1,0,1])
    rows = (2*ii[:,None] + a_idx[None,:]).reshape(-1)
    cols = (2*jj[:,None] + b_idx[None,:]).reshape(-1)
    data = sm[:,a_idx,b_idx].reshape(-1)
    keep = data!=0.0
    return coo_matrix((data[keep],(rows[keep],cols[keep])),
            shape=(2*nsites,2*nsites),dtype=np.complex128)


def km_vector(ri,rj,rm,tol=1e-5):
  """Return the Kane Mele vector"""
  v = np.array([0.,0.,0.])
  return km_vector_jit(np.array(ri).real,np.array(rj).real,v,np.array(rm).real,tol=tol)


@jit(nopython=True)
def km_vector_jit(ri,rj,v,rm,tol=1e-5):
    for k in range(rm.shape[0]): # look for an intermediate site
      dr1 = rm[k]-ri # difference
      dr2 = rj-rm[k] # difference
      dr1m = np.sum(dr1*dr1)
      dr2m = np.sum(dr2*dr2)
      if abs(dr1m-1.)<tol and abs(dr2m-1.)<tol: # if connected
         v = np.cross(dr1,dr2) # Kane Mele vector
         return v
    v = np.array([0.,0.,0.])
    return v





def haldane(r1,r2,rm,fun=0.0,sublattice=None):
    """Return the Haldane coupling.

    Builds the (nsites,nsites) sparse matrix from the (typically
    O(nsites)) bond list directly, instead of allocating a fully dense
    nsites x nsites complex array (160GB+ for nsites~1e5) -- and finds
    the candidate final sites for every site with one shared KD-tree
    query (find_close_neighbors_batch) instead of one O(len(r2)) scan per
    site."""
    r1 = np.array(r1)
    if sublattice is None: sublattice = np.zeros(len(r1)) + 1.0
    if isnumber(fun):
        if fun==0.0: return 0 # skip
        kmfun = lambda r: fun # function that always returns fun
    elif callable(fun): kmfun = fun # callable function
    else: # anything else
        from .potentials import array2potential
        kmfun = array2potential(r1[:,0],r1[:,1],fun)
    nsites = len(r1) # number of sites
    r2 = np.array(r2)
    rm = np.array(rm)
    from . import neighbor
    neighs = neighbor.connections(r1,rm) # list with neighbors of each site
    # radius must be >= sqrt(4.1) to match the dr.dot(dr)<4.1 acceptance
    # test below -- this pre-filter used to be a tighter d=1.9, which
    # silently dropped genuine bonds with distance in (1.9, sqrt(4.1))
    indsj_all = neighbor.find_close_neighbors_batch(r1,r2,d=np.sqrt(4.1)) # candidate final sites, per i
    rows,cols,data = [],[],[]
    for i in range(nsites): # loop over initial site
      rijs_idx = neighs[i] # first neighbors of site i
      if len(rijs_idx)==0: continue
      rijs = rm[rijs_idx]
      for j in indsj_all[i]: # loop over final site
        dr = r1[i]-r2[j] # difference
        if dr.dot(dr)>=4.1: continue
        ur = km_vector(r1[i],r2[j],rijs) # kane mele vector
        if ur[2]==0.0: continue
        r3 = (r1[i] + r2[j])/2.0
        v = 1j*ur[2]*kmfun(r3)*(sublattice[i]+sublattice[j])/2. # clockwise or anticlockwise
        if v!=0.0:
            rows.append(i); cols.append(j); data.append(v)
    return coo_matrix((data,(rows,cols)),shape=(nsites,nsites),
            dtype=np.complex128) # return matrix



def get_haldane_function(g,stagger=False):
    """Return a function that computes the Haldane coupling"""
    rm = g.multireplicas(3)
    def f(r1,r2):
      m = haldane([r1],[r2],rm,fun=1.0).todense()[0,0]
      if not stagger: return m # conventional Haldane
      i = g.get_index(r1,replicas=True)
      j = g.get_index(r2,replicas=True)
      i = g.sublattice[i]
      j = g.sublattice[j]
      m = m*(i+j)/2
      return m
    return f # return function











def add_haldane(h,t):
  """Add Haldane to a Hamiltonian"""
#  if h.has_spin: raise # not for spinful
#  if h.has_eh: raise # not for spinful
  add_haldane_like(h,t,haldane,sublattice=None) # add Haldane to a Hamiltonian


def add_modified_haldane(h,t):
  """Add Haldane to a Hamiltonian"""
#  if h.has_eh: raise # not for spinful
  if not h.geometry.has_sublattice: return # if it does not have sublattice
  add_haldane_like(h,t,haldane,sublattice=h.geometry.sublattice) 


def add_anti_kane_mele(h,t):
  """Add Haldane to a Hamiltonian"""
  if not h.geometry.has_sublattice: return # if it does not have sublattice
  if not h.has_spin: h.turn_spinful()
  add_haldane_like(h,t,haldane,sublattice=h.geometry.sublattice,
          time_reversal = True) 


def add_kane_mele(self,t,**kwargs):
  """Add to a Hamiltonian a Haldane-like hopping"""
  if not self.has_spin: self.turn_spinful() # spilful Hamiltonian
  from .multicell import close_enough # check if two rs are close
  g = self.geometry
  if not self.has_spin: raise  # only for spinfull
  if self.is_multicell:   # multicell Hamiltonian
    ncells = 2 # number of neighboring cells to check
    if self.dimensionality==0: rs = g.r # fix for zero dimensional
    elif self.dimensionality==1:  # three dimensional
      rs = [] # all the cells
      for i in range(-ncells,ncells+1): # loop over neighbouring cells
        rtmp = [ri + i*g.a1 for ri in g.r] # new positions
        if close_enough(g.r,rtmp,rcut=2.1): # if this positions are not too far
          rs += rtmp
    elif self.dimensionality==2:  # three dimensional
      rs = [] # all the cells
      for i in range(-ncells,ncells+1): # loop over neighbouring cells
        for j in range(-ncells,ncells+1):
          rtmp = [ri + i*g.a1 +j*g.a2 for ri in g.r] # new positions
          if close_enough(g.r,rtmp,rcut=2.1): # if this positions are not too far
            rs += rtmp
    elif self.dimensionality==3:  # three dimensional
      rs = [] # all the cells
      for i in range(-ncells,ncells+1): # loop over neighbouring cells
        for j in range(-ncells,ncells+1):
          for k in range(-ncells,ncells+1):
            rtmp = [ri + i*g.a1 +j*g.a2 + k*g.a3 for ri in g.r] # new positions
            if close_enough(g.r,rtmp,rcut=2.1): # if this positions are not too far
              rs += rtmp
    else: raise

    m = generalized_kane_mele(g.r,g.r,rs,fun=t,**kwargs) # kane mele coupling
    m = self.spinful2full(m) # convert the matrix
    self.intra = self.intra + m # add matrix
    for i in range(len(self.hopping)): # loop over hoppings
      d = self.hopping[i].dir
#      print("Generating Haldane-like",d,end="\r")
      r2 = [ri + d[0]*g.a1 + d[1]*g.a2 +d[2]*g.a3 for ri in g.r] # second vectors
      m = generalized_kane_mele(g.r,r2,rs,fun=t) # kane mele coupling
      m = self.spinful2full(m) # convert the matrix
      self.hopping[i].m += m
    return

  else:  # conventional Hamiltonian
    if self.dimensionality==0:  # zero dimensional
      rs = g.r # positions
    elif self.dimensionality==1:  # one dimensional
      rs = [] # all the cells
      for i in (-1,0,1):
        for ri in g.r:
          rs.append(ri + i*g.a1)
    elif self.dimensionality==2:  # two dimensional
      rs = [] # all the cells
      for i in (-1,0,1): # loop over neighbouring cells
        for j in (-1,0,1):
          for ri in g.r:
            rs.append(ri + i*g.a1 + j*g.a2)
    else: raise
    # now create the hamiltonian
    m = generalized_kane_mele(g.r,g.r,rs,fun=t,**kwargs) # kane mele coupling
    m = self.spinful2full(m) # convert the matrix
    self.intra = self.intra + m
    if self.dimensionality==0: pass  # zero dimensional
    elif self.dimensionality==1:  # zero dimensional
      r2 = [ri + g.a1 for ri in g.r] # new positions
      m = generalized_kane_mele(g.r,r2,rs,fun=t,**kwargs)
      m = self.spinful2full(m) # convert the matrix
      self.inter = self.inter + m 
    elif self.dimensionality==2:  # two dimensional
      r2 = [ri + 1*g.a1 + 0*g.a2 for ri in g.r] # second vectors
      m = generalized_kane_mele(g.r,r2,rs,fun=t,**kwargs)
      m = self.spinful2full(m) # convert the matrix
      self.tx = self.tx + m 
      ###############
      r2 = [ri + 0*g.a1 + 1*g.a2 for ri in g.r] # second vectors
      m = generalized_kane_mele(g.r,r2,rs,fun=t,**kwargs)
      m = self.spinful2full(m) # convert the matrix
      self.ty = self.ty + m 
      #################
      r2 = [ri + 1*g.a1 + 1*g.a2 for ri in g.r] # second vectors
      m = generalized_kane_mele(g.r,r2,rs,fun=t)
      m = self.spinful2full(m) # convert the matrix
      self.txy = self.txy + m 
      #################
      r2 = [ri + 1*g.a1  -1*g.a2 for ri in g.r] # second vectors
      m = generalized_kane_mele(g.r,r2,rs,fun=t)
      m = self.spinful2full(m) # convert the matrix
      self.txmy = self.txmy + m
    else: raise



from .increase_hilbert import spinful,spinful_sparse



def add_haldane_like(self,t,spinless_generator,
        sublattice=None,time_reversal=False):
  """Add to a Hamiltonian a Haldane-like hopping"""
  if isnumber(t): 
      if t==0.0: return # skip
  g = self.geometry
  if sublattice is None: sublattice = np.zeros(len(g.r)) + 1.0 # ones
  def generator(r1,r2,rs,fun=t,sublattice=sublattice): # define function
     m = spinless_generator(r1,r2,rs,fun=t,
             sublattice=sublattice)
     return self.spinless2full(m,time_reversal=time_reversal) # return matrix
#     if self.has_spin: return spinful_sparse(m) # spinful
#     else: return m # spinless
  from .multicell import close_enough # check if two rs are close
  if self.is_multicell:   # multicell Hamiltonians
    if self.dimensionality==0:  rs = g.r
    else: # higher dimensional
      ncells = self.geometry.get_ncells() # number of unit cells
      if self.dimensionality==1:  # three dimensional
        rs = [] # all the cells
        for i in range(-ncells,ncells+1): # loop over neighbouring cells
          rtmp = self.geometry.replicas(d=[i,0,0]) 
          if close_enough(g.r,rtmp,rcut=2.1): # if these positions are not too far
            rtmp = [ri for ri in rtmp]
            rs += rtmp
      elif self.dimensionality==2:  # three dimensional
        rs = [] # all the cells
        for i in range(-ncells,ncells+1): # loop over neighbouring cells
          for j in range(-ncells,ncells+1):
            rtmp = self.geometry.replicas(d=[i,j,0]) 
            if close_enough(g.r,rtmp,rcut=2.1): # if this positions are not too far
              rtmp = [ri for ri in rtmp]
              rs += rtmp
      elif self.dimensionality==3:  # three dimensional
        rs = [] # all the cells
        for i in range(-ncells,ncells+1): # loop over neighbouring cells
          for j in range(-ncells,ncells+1):
            for k in range(-ncells,ncells+1):
              rtmp = self.geometry.replicas(d=[i,j,k]) 
              if close_enough(g.r,rtmp,rcut=2.1): # if not too far
                rtmp = [ri for ri in rtmp]
                rs += rtmp
      else: raise

#    self.intra += generator(g.r,g.r,rs,fun=t,sublattice=sublattice) # coupling
    dirs = [[0,0,0]] + [t.dir for t in self.hopping] # directions
    def pfun(d): # function to parallelize
#    for i in range(len(self.hopping)): # loop over hoppings
#      print("Generating Haldane-like",d)
      r2 = self.geometry.replicas(d=d)
      return generator(g.r,r2,rs,fun=t,sublattice=sublattice) 
    ms = parallel.pcall(pfun,dirs) # get matrices
    self.intra = self.intra + ms[0] # intracell matrix
    for i in range(len(self.hopping)):
        self.hopping[i].m += ms[i+1] # store
    return

  else:  # conventional Hamiltonian
    if self.dimensionality==0:  # zero dimensional
      rs = g.r # positions
    elif self.dimensionality==1:  # one dimensional
      rs = [] # all the cells
      for i in (-1,0,1):
        for ri in g.r:
          rs.append(ri + i*g.a1)
    elif self.dimensionality==2:  # two dimensional
      rs = [] # all the cells
      for i in (-1,0,1): # loop over neighbouring cells
        for j in (-1,0,1):
          for ri in g.r:
            rs.append(ri + i*g.a1 + j*g.a2)
    else: raise
    # now create the hamiltonian
    self.intra = self.intra + generator(g.r,g.r,rs,fun=t,sublattice=sublattice) # kane mele coupling
    if self.dimensionality==0: pass  # zero dimensional
    elif self.dimensionality==1:  # zero dimensional
      r2 = [ri + g.a1 for ri in g.r] # new positions
      self.inter += generator(g.r,r2,rs,fun=t,sublattice=sublattice) # kane mele coupling
    elif self.dimensionality==2:  # two dimensional
      r2 = [ri + 1*g.a1 + 0*g.a2 for ri in g.r] # second vectors
      self.tx = self.tx + generator(g.r,r2,rs,fun=t,sublattice=sublattice)
      r2 = [ri + 0*g.a1 + 1*g.a2 for ri in g.r] # second vectors
      self.ty = self.ty + generator(g.r,r2,rs,fun=t,sublattice=sublattice)
      r2 = [ri + 1*g.a1 + 1*g.a2 for ri in g.r] # second vectors
      self.txy = self.txy + generator(g.r,r2,rs,fun=t,sublattice=sublattice)
      r2 = [ri + 1*g.a1  -1*g.a2 for ri in g.r] # second vectors
      self.txmy = self.txmy + generator(g.r,r2,rs,fun=t,sublattice=sublattice)
    else: raise




def add_kane_mele_old(self,t):
  """Add to a Hamiltonian a Haldane-like hopping"""
#  print("This does not work with eh, kanemele")
  if not self.has_spin: self.turn_spinful() # spilful Hamiltonian
  from .multicell import close_enough # check if two rs are close
  g = self.geometry
  if not self.has_spin: raise  # only for spinfull
  if self.is_multicell:   # multicell Hamiltonians
    ncells = 4 # number of neighboring cells to check
    if self.dimensionality==1:  # three dimensional
      rs = [] # all the cells
      for i in range(-ncells,ncells+1): # loop over neighbouring cells
        rtmp = [ri + i*g.a1 for ri in g.r] # new positions
        if close_enough(g.r,rtmp,rcut=2.1): # if this positions are not too far
          rs += rtmp
    elif self.dimensionality==2:  # three dimensional
      rs = [] # all the cells
      for i in range(-ncells,ncells+1): # loop over neighbouring cells
        for j in range(-ncells,ncells+1):
          rtmp = [ri + i*g.a1 +j*g.a2 for ri in g.r] # new positions
          if close_enough(g.r,rtmp,rcut=2.1): # if this positions are not too far
            rs += rtmp
    elif self.dimensionality==3:  # three dimensional
      rs = [] # all the cells
      for i in range(-ncells,ncells+1): # loop over neighbouring cells
        for j in range(-ncells,ncells+1):
          for k in range(-ncells,ncells+1):
            rtmp = [ri + i*g.a1 +j*g.a2 + k*g.a3 for ri in g.r] # new positions
            if close_enough(g.r,rtmp,rcut=2.1): # if this positions are not too far
              rs += rtmp
    else: raise

    self.intra += generalized_kane_mele(g.r,g.r,rs,fun=t) # kane mele coupling
    for i in range(len(self.hopping)): # loop over hoppings
      d = self.hopping[i].dir
#      print("Adding Kane Mele in hopping",d,end="\r")
      r2 = [ri + d[0]*g.a1 + d[1]*g.a2 +d[2]*g.a3 for ri in g.r] # second vectors
      self.hopping[i].m += generalized_kane_mele(g.r,r2,rs,fun=t) # kane mele coupling
    return

  else:  # conventional Hamiltonian
    if self.dimensionality==0:  # zero dimensional
      rs = g.r # positions
    elif self.dimensionality==1:  # one dimensional
      rs = [] # all the cells
      for i in (-1,0,1):
        for ri in g.r:
          rs.append(ri + i*g.a1)
    elif self.dimensionality==2:  # two dimensional
      rs = [] # all the cells
      for i in (-1,0,1): # loop over neighbouring cells
        for j in (-1,0,1):
          for ri in g.r:
            rs.append(ri + i*g.a1 + j*g.a2)
    else: raise
    # now create the hamiltonian
    self.intra = self.intra + generalized_kane_mele(g.r,g.r,rs,fun=t) # kane mele coupling
    if self.dimensionality==0: pass  # zero dimensional
    elif self.dimensionality==1:  # zero dimensional
      r2 = [ri + g.a1 for ri in g.r] # new positions
      self.inter += generalized_kane_mele(g.r,r2,rs,fun=t) # kane mele coupling
    elif self.dimensionality==2:  # two dimensional
      r2 = [ri + 1*g.a1 + 0*g.a2 for ri in g.r] # second vectors
      self.tx = self.tx + generalized_kane_mele(g.r,r2,rs,fun=t)
      r2 = [ri + 0*g.a1 + 1*g.a2 for ri in g.r] # second vectors
      self.ty = self.ty + generalized_kane_mele(g.r,r2,rs,fun=t)
      r2 = [ri + 1*g.a1 + 1*g.a2 for ri in g.r] # second vectors
      self.txy = self.txy + generalized_kane_mele(g.r,r2,rs,fun=t)
      r2 = [ri + 1*g.a1  -1*g.a2 for ri in g.r] # second vectors
      self.txmy = self.txmy + generalized_kane_mele(g.r,r2,rs,fun=t)
    else: raise
