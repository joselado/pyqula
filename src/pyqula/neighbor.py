from __future__ import print_function
import numpy as np
from scipy.sparse import csc_matrix,bmat
from numba import jit
from . import algebra


minimum_hopping = 1e-3


try: 
  from . import first_neighborsf90
  def find_first_neighbor(r1,r2):
      """Calls the fortran routine"""
      from . import first_neighborsf90 as fn
      r1t = np.matrix(r1).T
      r2t = np.matrix(r2).T
      nn = fn.number_neighborsf90(r1t,r2t)
      if nn==0: return []  # if no neighbors found
      pairs = np.array(fn.first_neighborsf90(r1t,r2t,nn))
      return pairs.T # return the pairs


except:
  def find_first_neighbor(r1,r2):
     """Calls the fortran routine"""
     r1 = np.array(r1)
     r2 = np.array(r2)
     nn = number_neighbors_jit(r1,r2) # number of first neighbors
     out = np.zeros((nn,2),dtype=np.int) # generate indexes
     out = find_first_neighbor_jit(r1,r2,out) # generate all the pairs
     return out

@jit(nopython=True)
def number_neighbors_jit(r1,r2):
    """Number of neighbors"""
    out = 0
    for i in range(len(r1)):
      for j in range(len(r2)):
         ri = r1[i]
         rj = r2[j]
         dr = ri-rj
         dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
         if 0.99<dr2<1.01: out += 1 # increase
    return out # number of neighbors

@jit(nopython=True)
def find_first_neighbor_jit(r1,r2,pairs):
    """Find the first neighbors"""
    out = 0
    for i in range(len(r1)):
      for j in range(len(r2)):
         ri = r1[i]
         rj = r2[j]
         dr = ri-rj
         dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
         if 0.99<dr2<1.01:
             pairs[out,0] = i
             pairs[out,1] = j
             out += 1 # increase
    return pairs # number of neighbors






def connections(r1,r2):
  """Return a list with the connections of each atom"""
  pairs = find_first_neighbor(r1,r2) # get the pairs of first neighbors
  out = [[] for i in range(len(r1))] # output list
  for p in pairs:
    out[int(p[0])].append(int(p[1])) 
  return out # return list





def parametric_hopping(r1,r2,fc,is_sparse=False):
  """ Generates a parametric hopping based on a function"""
  if is_sparse: # sparse matrix
    print("Sparse parametric hopping")
    m = np.matrix([[0.0j for i in range(len(r2))] for j in range(len(r1))])
    rows,cols,data = [],[],[]
    for i in range(len(r1)):
      for j in range(len(r2)):
        val = fc(r1[i],r2[j]) # add hopping based on function
        if abs(val) > minimum_hopping: # retain this hopping
         data.append(val)
         rows.append(i)
         cols.append(j)
    n = len(r2)
    m = csc_matrix((data,(rows,cols)),shape=(n,n))
  #  if not is_sparse: m = m.todense() # dense matrix
    return m
  else:
    n = len(r2)
    m = np.matrix(np.zeros((n,n),dtype=np.complex)) # complex matrix
    for i in range(len(r1)):
      for j in range(len(r2)):
        m[i,j] = fc(r1[i],r2[j])
    return m

 





def parametric_hopping_spinful(r1,r2,fc,is_sparse=False):
  """ Generates a parametric hopping based on a function, that returns
  a 2x2 matrix"""
  m = [[None for i in range(len(r2))] for j in range(len(r1))]
  for i in range(len(r1)):
    for j in range(len(r2)):
      val = fc(r1[i],r2[j]) # add hopping based on function
      m[i][j] = val # store this result
  m = bmat(m) # convert to matrix
  if not is_sparse: m = m.todense() # dense matrix
  return m







def generate_parametric_hopping(h,f=None,mgenerator=None,
             spinful_generator=False):
  """ Adds a parametric hopping to the hamiltonian based on an input function"""
  rs = h.geometry.r # positions
  g = h.geometry # geometry
  has_spin = h.has_spin # check if it has spin
  is_sparse = h.is_sparse
  if mgenerator is None: # no matrix generator given on input
    if f is None: raise # no function given on input
    if spinful_generator:
      raise
      print("WARNING, I am not sure why I programmed this")
      h.has_spin = True
      generator = parametric_hopping_spinful
    else:
      h.has_spin = False
      generator = parametric_hopping
    def mgenerator(r1,r2):
      return generator(r1,r2,f,is_sparse=is_sparse)
  else:
    if h.dimensionality==3: raise
  h.intra = mgenerator(rs,rs)
  if h.dimensionality == 0: pass
  elif h.dimensionality == 1:
    dr = g.a1
    h.inter = mgenerator(rs,rs+dr)
  elif h.dimensionality == 2:
    h.tx = mgenerator(rs,rs+g.a1)
    h.ty = mgenerator(rs,rs+g.a2)
    h.txy = mgenerator(rs,rs+g.a1+g.a2)
    h.txmy = mgenerator(rs,rs+g.a1-g.a2)
  elif h.dimensionality == 3:
    if spinful_generator: raise # not implemented
    h.is_multicell = True # multicell Hamiltonian
    from . import multicell
    multicell.parametric_hopping_hamiltonian(h,fc=f)
  else: raise
  # check that the sparse mde is set ok
  if is_sparse and type(h.intra)==type(np.matrix([[]])):
    h.is_sparse = False
    h.turn_sparse() # turn the matrix sparse
  if not is_sparse and type(h.intra)!=type(np.matrix([[]])):
    h.is_sparse = True
    h.turn_dense() # turn the matrix sparse
  if has_spin: # Hamiltonian should be spinful
    h.has_spin = False
    h.turn_spinful()
  return h




def neighbor_distances(g,n=4):
    """Return distances to neighbors"""
    nsuper = max([n//len(g.r)+3,3])
    g = g.supercell(nsuper) # create supercell
    r = g.r # positions
    n = len(r)
    out = np.zeros(n*n) # empty array
    out = neighbor_distances_jit(r,out) # distances
    out = np.round(out,6) # unique distances
    out = np.unique(out) # unique distances
    return np.array([out[i+1] for i in range(len(out)-1)]) # return


@jit(nopython=True)
def neighbor_distances_jit(r,out):
    n = len(r) # number of sites
    k = 0
    for i in range(n):
        for j in range(n):
            dr = r[i]-r[j]
            dis = dr[0]*dr[0]+dr[1]*dr[1]+dr[2]*dr[2]
            dis = np.sqrt(dis) # square root
            out[k] = dis # store
            k+=1 # increase
    return out



def neighbor_cells(num,dim=3):
  """Return indexes of neighboring cells,
  ordered from closer to further"""
  cells = [] # empty list
  if dim==0: return cells
  elif dim==1:
    for i in range(-num,num+1): cells.append([i,0,0])
  elif dim==2:
    for i in range(-num,num+1):
      for j in range(-num,num+1):
        cells.append([i,j,0])
  elif dim==3:
    for i in range(-num,num+1):
      for j in range(-num,num+1):
        for k in range(-num,num+1):
          cells.append([i,j,k])
  # now order the cells
  dis = [np.array(a).dot(np.array(a)) for a in cells] # distances
  cells = [y for (x,y) in zip(dis,cells)] # sort
  return cells # return the indexes



def neighbor_directions(g,cutoff=3):
  """Return the vectors pointing to neighbors"""
  dirs = []
  if g.dimensionality==0: return [[0.,0.,0.]] # zero dimensional
  elif g.dimensionality==1: # one dimensional
    for i1 in range(-cutoff,cutoff+1): dirs.append([i1,0,0])
  elif g.dimensionality==2: # two dimensional
    for i1 in range(-cutoff,cutoff+1):
      for i2 in range(-cutoff,cutoff+1):
        dirs.append([i1,i2,0])
  elif g.dimensionality==3: # three dimensional
    for i1 in range(-cutoff,cutoff+1):
      for i2 in range(-cutoff,cutoff+1):
        for i3 in range(-cutoff,cutoff+1):
          dirs.append([i1,i2,i3])
  dirs = [np.array(d) for d in dirs]
  return dirs # return directions

