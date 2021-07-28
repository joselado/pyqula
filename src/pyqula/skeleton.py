# rountines to create hamiltonians ased on a certain skeleton,
# used to build ribbons

from scipy.sparse import csc_matrix
from scipy.sparse import bmat
from . import sculpt
from . import geometry
from . import multicell
import numpy as np

def build_ribbon(hin,g=None,n=20):
  """ Build a supercell using a certain geometry as skeleton"""
  if g is None: # if skeleton not provided
    h.geometry.get_lattice_name()
    if h.geometry.lattice_name=="square": # square lattice
      g = geometry.square_ribbon(n) 
    else: raise # not implemented
  # now build the hamiltonian
  h = hin.copy() # generate hamiltonian
  # if the hamiltonian is not multicell, turn it so
  if not h.is_multicell: h = multicell.turn_multicell(h)
  gin = h.geometry # geometry of the hamiltonian input
# use the same axis
  gh = sculpt.rotate_a2b(h.geometry,h.geometry.a1,np.array([0.,1.,0.])) 
#  if np.abs(h.geometry.a1.dot(h.geometry.a2)) > 0.01: raise # orthogonal
#  gh = h.geometry
  def normalize(v): # normalize a vector
    return v/np.sqrt(v.dot(v))
# get reciprocal vectors
  (w1,w2,w3) = sculpt.reciprocal(normalize(gh.a1),normalize(gh.a2)) 
#  exit()
  def get_rij(r):
    """Provide a vector r, return this vector expressed in the basis cell""" 
    i = r.dot(w1) # first projection
    j = r.dot(w2) # second projection
    i,j = round(i),round(j)
    return [i,j,0] # return indexes
  ho = h.copy() # generate hamiltonian
  intra = [[None for i in range(len(g.r))] for j in range(len(g.r))]
  hoppings = [] # empty list for the hoppings
  for i in range(len(g.r)): # loop over positions
    intra[i][i] = h.intra # intracell hopping
  for i in range(len(g.r)): # hopping up to third cells
    for j in range(len(g.r)): # hopping up to third cells
      rij = g.r[i] - g.r[j] # distance between replicas
      intra[i][j] = multicell.get_tij(h,rij=get_rij(rij)) 
  ho.intra = csc_matrix(bmat(intra)) # add the intracell matrix
  for nn in [-3,-2,-1,1,2,3]: # hopping up to third cells
    inter = [[None for i in range(len(g.r))] for j in range(len(g.r))]
    for i in range(len(g.r)): # hopping up to third cells
      for j in range(len(g.r)): # hopping up to third cells
        rij = g.r[i] - g.r[j] # distance between replicas
        rij += nn*h.geometry.a1 # add the displacement
        if i==j: # for diagonal, at least zeros
          mm = multicell.get_tij(h,rij=get_rij(rij)) 
          if mm is None: inter[i][j] = h.intra*0.0 # store zero
          else: inter[i][j] = mm # store matrix
        else: inter[i][j] = multicell.get_tij(h,rij=get_rij(rij)) 
    hopping = multicell.Hopping() # create object
    hopping.m = csc_matrix(bmat(inter)) # store matrix
    hopping.dir = np.array([nn,0.,0.]) # store vector
    hoppings.append(hopping) # store hopping
  gout = g.copy() # copy geometry for the hamiltonian
  rs = []
  for jr in g.r: # loop over skeleton geometry
    for ir in h.geometry.r: # loop over basis
      rs.append(ir + jr[0]*h.geometry.a1 + jr[1]*h.geometry.a2)
  gout.r = np.array(rs) # store
  gout.r2xyz() # update
  gout.celldis = h.geometry.a1[0] # this has to be well done
  ho.geometry = gout # assign geometry
  ho.hopping = hoppings # store the full hoppings list
  ho.dimensionality = 1 # one dimensional
  ho.is_multicell = True # multicell Hamiltonian
  ho.is_sparse = True # sparse Hamiltonian
  return ho # return ribbon hamiltonian






def build_island(h,n=5,angle=30,nedges=6):
  """ Build an island starting from a 2d geometry"""
  gin = geometry.triangular_lattice() # create lattice
  angle = sculpt.get_angle(h.geometry.a1,h.geometry.a2)/np.pi*180
  if np.abs(angle-60)<1.: gin.a2 = -gin.a2 # change the unit cell
  g = sculpt.build_island(gin,n=n,angle=angle,nedges=nedges,clear=False) # get the island
  angle2 = sculpt.get_angle(gin.a1,gin.a2)/np.pi*180
  if np.abs(angle-angle2)>1.: raise # error in the angles
  gh = sculpt.rotate_a2b(h.geometry,h.geometry.a1,gin.a1) # use the same axis
  # now define a function to select the correct hopping
  (w1,w2,w3) = sculpt.reciprocal(gin.a1,gin.a2) # get reciprocal vectors
  def get_rij(r):
    """Provide a vector r, return this vector expressed in the basis cell""" 
    i = r.dot(w1) # first projection
    j = r.dot(w2) # second projection
    return i,j # return indexes
  # turn into multicell
  if not h.is_multicell: h = multicell.turn_multicell(h) 
  h.turn_sparse() # convert into sparse
  intra = [[None for ri in g.r] for rj in g.r ]
  # loop over the skeleton
  for i in range(len(g.r)): # loop over i
    ri = g.r[i]
    for j in range(len(g.r)): # loop over j
      rj = g.r[j]
      vi,vj = get_rij(ri-rj) # get the vector
      if np.abs(vi)>2 or np.abs(vj)>2: continue
      m = multicell.get_tij(h,rij=np.array([vi,vj,0])) # return the matrix
      intra[i][j] = m # store in the matrix
  # fic the new hamiltonian
  ho = h.copy() # copy hamiltonian object
  from scipy.sparse import bmat
  ho.dimensionality = 0 # zero dimensional
  ho.intra = bmat(intra) # store hamiltonian
  # fix the new geometry
  go = h.geometry.copy() # copy the original geometry
  go.dimensionality = 0 # zero dimensional
  rs = [] # empty list
  for rd in g.r:
    for r in h.geometry.r:
      vi,vj = get_rij(rd) # get the vector
      ri = r +vi*h.geometry.a1 + vj*h.geometry.a2
      rs.append(ri) # append the vector
  rs = np.array(rs) # convert to array
  go.r = rs # store
  if go.atoms_have_names: # if the atoms have names, expand
    go.atoms_names = go.atoms_names*len(g.r) # enlarge the list
  go.r2xyz() # fill the xyz values
  ho.geometry = go # store in the hamiltonian
  go.write()
  return ho




def image2island(impath,h,s=20):
  """ Build an island starting from a 2d geometry"""
  gin = h.geometry # create lattice
  g = sculpt.image2island(impath,gin,s=s) # get the island
  angle = 0.0
  angle2 = sculpt.get_angle(gin.a1,gin.a2)/np.pi*180
  if np.abs(angle-angle2)>1.: raise # error in the angles
  gh = sculpt.rotate_a2b(h.geometry,h.geometry.a1,gin.a1) # use the same axis
  # now define a function to select the correct hopping
  (w1,w2,w3) = sculpt.reciprocal(gin.a1,gin.a2) # get reciprocal vectors
  def get_rij(r):
    """Provide a vector r, return this vector expressed in the basis cell""" 
    i = r.dot(w1) # first projection
    j = r.dot(w2) # second projection
    return i,j # return indexes
  # turn into multicell
  if not h.is_multicell: h = multicell.turn_multicell(h) 
  h.turn_sparse() # convert into sparse
  intra = [[None for ri in g.r] for rj in g.r ]
  # loop over the skeleton
  for i in range(len(g.r)): # loop over i
    ri = g.r[i]
    for j in range(len(g.r)): # loop over j
      rj = g.r[j]
      vi,vj = get_rij(ri-rj) # get the vector
      if np.abs(vi)>2 or np.abs(vj)>2: continue
      m = multicell.get_tij(h,rij=np.array([vi,vj,0])) # return the matrix
      intra[i][j] = m # store in the matrix
  # fic the new hamiltonian
  ho = h.copy() # copy hamiltonian object
  from scipy.sparse import bmat
  ho.dimensionality = 0 # zero dimensional
  ho.intra = bmat(intra) # store hamiltonian
  # fix the new geometry
  go = h.geometry.copy() # copy the original geometry
  go.dimensionality = 0 # zero dimensional
  rs = [] # empty list
  for rd in g.r:
    for r in h.geometry.r:
      vi,vj = get_rij(rd) # get the vector
      ri = r +vi*h.geometry.a1 + vj*h.geometry.a2
      rs.append(ri) # append the vector
  rs = np.array(rs) # convert to array
  go.r = rs # store
  if go.atoms_have_names: # if the atoms have names, expand
    go.atoms_names = go.atoms_names*len(g.r) # enlarge the list
  go.r2xyz() # fill the xyz values
  ho.geometry = go # store in the hamiltonian
  go.write()
  return ho



