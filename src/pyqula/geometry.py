from __future__ import print_function
import numpy as np
from copy import deepcopy
from scipy.sparse import bmat
from scipy.sparse import csc_matrix as csc
from . import sculpt
from . import klist
from .supercell import non_orthogonal_supercell
from . import supercell as supercelltk
from . import checkclass
import scipy.linalg as lg
from numba import jit
from .htk.g2h import get_hamiltonian
from .helptk import get_docstring

class Geometry:
    """ Class for a geometry in a system """
    def __init__(self):
        self.data = dict() # empty dictionary with different data
        self.has_sublattice = False # has sublattice index
        self.sublattice_number = 2 # two sublattices
        self.has_fractional = False
        self.dimensionality = 1 # dimension of the hamiltonian
        self.x = [] # positions in x
        self.y = [] # positions in y
        self.z = [] # positions in z
        self.r = [] # full positions 
        self.celldis = 1.0 # distance to the nearest cell (for 1d)
        self.a1 = np.array([100.0,0.0,0.])  # first vector to the nearest cell
        self.a2 = np.array([0.0,100.0,0.])  # first vector to the nearest cell
        self.a3 = np.array([0.0,0.0,100.])  # first vector to the nearest cell
        self.b1 = np.array([1.0,0.0,0.])  # first vector to the nearest cell
        self.b2 = np.array([0.0,1.0,0.])  # first vector to the nearest cell
        self.b3 = np.array([0.0,0.0,1.])  # first vector to the nearest cell
        self.shift_kspace = False # shift the klist when plotting
        self.name = "None"
        self.primal_geometry = None # store the primal geometry
        self.supercell_matrix = None # integer matrix M, if built via get_supercell(M)
        self.supercell_replica = None # per-atom replica vector n in Z^3, if built via get_supercell(M)
        self.supercell_primal_index = None # per-atom primal-cell atom index, if built via get_supercell(M)
        self.lattice_name = "" # lattice name
        self.atoms_names = [] # no name for the atoms
        self.atoms_have_names = False # atoms do not have names
        self.ncells = 2 # number of neighboring cells returned
    def neighbor_distances(self,**kwargs):
        return neighbor_distances(self,**kwargs)
    def get_neighbor_distances(self,**kwargs):
        return neighbor_distances(self,**kwargs)
    def normalize_nn_distance(self):
        """Set the NN istance equal to 1"""
        if self.dimensionality>0: raise
        d = self.neighbor_distances(n=1)[0]
        self.r = self.r/d
        self.r2xyz()
    def get_index(self,r,**kwargs):
        return get_index(self,r,**kwargs)
    def __add__(self,g1):
        from .geometrytk.galgebra import sum_geometries
        return sum_geometries(self,g1)
    def __sub__(self,a):
        return self + (-1)*a
    def __radd__(self,g1):
        return sum_geometries(self,g1)
    def get_kmesh(self,**kwargs):
        """Return the k-mesh"""
        return klist.kmesh(self.dimensionality,**kwargs)
    def get_default_kpath(self,**kwargs):
        from . import klist
        return klist.default(self,**kwargs)
    def set_finite(self,periodic=False):
      """ Transfrom the geometry into a finite system"""
      if periodic:
        f = self.periodic_vector() # get the function
        self.get_distance = f # store that function
      self.dimensionality = 0 # set as finite
    def get_orthogonal(self):
        return supercelltk.target_angle_volume(self,angle=0.5)
    def closest_index(self,r):
        return sculpt.get_closest(self,n=1,r0=r)[0]
    def get_closest_position(self,r,n=1):
        r = np.array(r)
        if n==0:
          ii = self.closest_index(r)
          return self.r[ii] # return this position
        else:
          iis = sculpt.get_closest(self,n=n,r0=r)
          return [self.r[ii] for ii in iis] # return positions
    def get_supercell(self,nsuper,**kwargs):
        return get_supercell(self,nsuper,**kwargs)
    supercell = get_supercell # backwards compatibility
    def xyz2r(self):
      """Updates r atributte according to xyz"""
      self.r = np.array([self.x,self.y,self.z]).transpose()
    def r2xyz(self):
      """Updates x,y,z atributtes according to r"""
      r = np.array(self.r).transpose()
      self.x = r[0]
      self.y = r[1]
      self.z = r[2]
    @get_docstring(get_hamiltonian) # inherint docstring
    def get_hamiltonian(self,**kwargs):
        return get_hamiltonian(self,**kwargs)
    def write(self,**kwargs):
        """ Writes the geometry in file"""
        write_positions(self,**kwargs)
        write_xyz(self)
        write_lattice(self)
        write_sublattice(self)
    def get_kpath(self,*args,**kwargs):
        return klist.get_kpath(self,*args,**kwargs)
    def write_positions(self,**kwargs):
        """Write the positions in a file"""
        write_positions(self,**kwargs)
    def copy(self):
        """Copy the geometry"""
        return deepcopy(self)
    def set_origin(self,r=None):
        if r is None: r = self.r[self.get_central()[0]]
        self.x = self.x - r[0]
        self.y = self.y - r[1]
        self.z = self.z - r[2]
        self.xyz2r() # update r
    def center(self):
        """ Centers the geometry in (0,0,0)"""
        self.x = self.x - np.sum(self.x)/len(self.x)
        self.y = self.y - np.sum(self.y)/len(self.y)
        self.z = self.z - np.sum(self.z)/len(self.z)
        self.xyz2r() # update r
    def get_lattice_name(self):
        if self.dimensionality==2:
            if np.abs(self.a1.dot(self.a2))<0.0001:        
              self.lattice_name = "square"
            else:
              self.lattice_name = "triangular"
    def get_k2K(self):
        from .kpointstk.mapping import get_k2K
        return get_k2K(self)
    def reciprocal2natural(self,v):
        """
        Return a natural vector in real reciprocal coordinates
        """
        return self.get_k2K_generator()(v)
    def get_fractional(self,center=False):
        """Fractional coordinates"""
        self.update_reciprocal() # update reciprocal lattice vectors
        get_fractional(self,center=center) # get fractional coordinates
    def rotate(self,angle):
      """Rotate the geometry"""
      return sculpt.rotate(self,angle*np.pi/180)
    def clean(self,iterative=False):
      return sculpt.remove_unibonded(self,iterative=iterative)
    def get_diameter(self):
      """Return the maximum distance between two atoms"""
      return get_diameter(self)  
    def periodic_vector(self):
      return periodic_vector(self)
    def get_sublattice(self,**kwargs):
      """Initialize the sublattice"""
      if self.has_sublattice: 
          self.sublattice = get_sublattice(self.r,**kwargs)
      else: 
          self.sublattice = get_sublattice(self.r,**kwargs)
          self.has_sublattice = True
    def shift(self,r0):
      """Shift all the positions by r0"""
      self.x[:] -= r0[0]
      self.y[:] -= r0[1]
      self.z[:] -= r0[2]
      self.xyz2r() # update
      if self.dimensionality>0:
        self.get_fractional(center=True)
        self.fractional2real()
    def write_function(self,fun,**kwargs):
        from .geometrytk.write import write_function
        return write_function(self,fun,**kwargs)
    def neighbor_directions(self,n=None):
      """Return directions linking to neighbors"""
      if n is None: n = self.ncells
      return neighbor_directions(self,n)
    def get_ncells(self):
        if self.dimensionality==0: return 0
        else:
            n = int(10/np.sqrt(self.a1.dot(self.a1)))
            return max([1,n])
    def write_profile(self,d,**kwargs):
        """Write a profile in a file"""
        write_profile(self,d,**kwargs)
    def replicas(self,**kwargs):
        from .geometrytk.replicas import replicas
        return replicas(self,**kwargs)
    def multireplicas(self,n):
        from .geometrytk.replicas import multireplicas
        return multireplicas(self,n)
    def bloch_phase(self,d,k):
        """Return the Bloch's phase for a specific k-vector"""
        from .geometrytk.bloch import bloch_phase
        return bloch_phase(self,d,k)
    def remove(self,i=0):
        """
        Remove one site
        """
        if callable(i): return sculpt.intersec(self,lambda r: not i(r))
        if type(i)==list: pass
        else: i = [i]
        return sculpt.remove(self,i)
    def center_in_atom(self,n0=None):
        """
        Center the geometry in an atom
        """
        if n0 is None: n0 = sculpt.get_central(self)[0] # get the index
        sculpt.shift(self,r=self.r[n0]) # shift the geometry
    def get_central(self,n=1):
        """
        Return a list of central atoms
        """
        return sculpt.get_central(self,n=n) # get the index
    def update_reciprocal(self):
        """
        Update reciprocal lattice vectors
        """
        self.b1,self.b2,self.b3 = get_reciprocal(self.a1,self.a2,self.a3)
    def get_k2K_generator(self,**kwargs):
        return get_k2K_generator(self,**kwargs)
    def k2K(self,k): return get_k2K_generator(self,toreal=False)(k)
    def K2k(self,k): return get_k2K_generator(self,toreal=True)(k)
    def fractional2real(self):
      """
      Convert fractional coordinates to real coordinates
      """
      fractional2real(self)
    def real2fractional(self):
      self.get_fractional() # same function
    def add_strain(self,*args,**kwargs):
        from .geometrytk import strain
        return strain.add_strain(self,*args,**kwargs)
    def get_connections(self):
      """
      Return the connections of each site
      """
      from . import neighbor
      self.connections = neighbor.connections(self.r,self.r)
      return self.connections # return list



from .geometrytk.lattices import *



def supercell1d(g,nsuper):
  """
  Creates a supercell of the system
  """
  # get the old geometry 
  y = g.y
  x = g.x
  z = g.z
  celldis = g.a1[0]
  if np.abs(g.a1.dot(g.a1) - g.a1[0]**2)>0.001:
    print("Something weird in supercell 1d")
    return supercell1d(sculpt.rotate_a2b(g,g.a1,np.array([1.,0.,0.])),nsuper) 
  # position of the supercell
  yout = []
  xout = []
  for i in range(nsuper):
    yout += y.tolist()
    xout += (x+i*celldis).tolist()
  # now modify the geometry
  go = deepcopy(g)
  go.x = np.array(xout)
  go.y = np.array(yout)
  # and shift to zero
  go.z = np.array(z.tolist()*nsuper)
  go.center() # center the unit cell
  go.celldis = celldis*nsuper
  go.a1 = g.a1*nsuper # supercell
  go.xyz2r() # update r
  if g.has_sublattice: # if has sublattice, keep the indexes
    go.sublattice = np.concatenate([g.sublattice for i in range(nsuper)])  # store the keeped atoms
#    print(nsuper)
  if g.atoms_have_names: # supercell sublattice
    go.atoms_names = g.atoms_names*nsuper
  go.get_fractional()
  return go






from .supercell import supercell2d
from .supercell import supercell3d




def read(input_file="POSITIONS.OUT"):
  """ Reads a geometry """
  m = np.genfromtxt(input_file).transpose()
  g = Geometry() # cretae geometry
  g.dimensionality = 0
  g.x = m[0]
  g.y = m[1]
  g.z = m[2]
  g.xyz2r() # create r coordinates
  # check whether if it has sublattice
  try:
    g.sublattice = np.genfromtxt("SUBLATTICE.OUT")
    g.has_sublattice = True
    if len(g.sublattice) != len(g.r): raise
    print("Read sublattice from SUBLATTICE.OUT")
  except: g.has_sublattice = False
  try:
    dim = int(open("DIMENSIONALITY.OUT").read())
  except: dim = 0
  g.dimensionality = dim # store
  if dim>0: # if it has lattice
    lat = np.genfromtxt("LATTICE.OUT")   # read lattice
    if dim==2: # two dimensional
      g.a1 = lat[0]
      g.a2 = lat[1]
    elif dim==3: # two dimensional
      g.a1 = lat[0]
      g.a2 = lat[1]
      g.a3 = lat[2]
    elif dim==1: # two dimensional
      g.celldis = np.sqrt(lat.dot(lat)) # distance between cells
      g.a1 = lat
  return g




from .ribbon import bulk2ribbon


def get_reciprocal2d(a1,a2):
  """Get reciprocal vectors"""
  b1 = np.array([a2[1],-a2[0],0.])
  b2 = np.array([-a1[1],a1[0],0.])
  b1 = b1 / np.sqrt(b1.dot(b1))
  b2 = b2 / np.sqrt(b2.dot(b2))
  return (b1,b2)


def apilate(g,drs=[np.array([0.,0.,0.])]):
  """ generate a geometry by appilating another geometry, displacements
  are given on input """
  nl = len(drs) # number of layers
  ro = np.concatenate([g.r + dr for dr in drs])
  go = g.copy()
  go.r = ro
  go.r2xyz()
  return go



from .geometrytk.write import write_xyz
from .geometrytk.write import write_lattice
from .geometrytk.write import write_sublattice
from .geometrytk.write import write_positions
write_geometry = write_positions


def remove_duplicated(g):
  """ Remove duplicated atoms"""
  if not g.atoms_have_names: raise
  go = g.copy() # copy geometry
  rs = remove_duplicated_positions(g.r)
  go.r = np.array(rs)
  go.r2xyz() # update the other coordinates
#  go.atoms_names = names
  return go


def remove_duplicated_positions(r):
  r = np.array(r) # as array
  if len(r)==0: return np.zeros((0,3))
  rs = np.empty(r.shape) # upper bound on the number of kept atoms
  nkept = 0 # number of atoms kept so far
  for ir in r: # loop over atoms
     if nkept==0: store = True # nothing stored yet
     else: # compare against every already-kept atom at once instead of
       # a per-pair python loop (this used to be O(natoms^2) with a
       # python-level distance computation on every pair)
       dr = rs[:nkept]-ir
       store = not np.any(np.sum(dr*dr,axis=1)<0.01)
     if store: # store this atom
       rs[nkept] = ir
       nkept += 1
  return rs[:nkept].copy() # return unrepeated atoms




def get_reciprocal(a1,a2,a3):
  """Return the reciprocal lattice vectors
  By definition, ai*bj = delta_ij"""
  (ux,uy,uz) = (a1,a2,a3)
#  ux = ux/np.sqrt(ux.dot(ux))
#  uy = uy/np.sqrt(uy.dot(uy))
#  uz = uz/np.sqrt(uz.dot(uz))
  a2kn = np.array([ux,uy,uz]) # matrix for the change of basis
  r2a = np.linalg.inv(np.array([ux,uy,uz]).T) # from real space to lattice vectors
  b1,b2,b3 = r2a[0,:],r2a[1,:],r2a[2,:]
  b1 = np.array(b1).reshape(3)
  b2 = np.array(b2).reshape(3)
  b3 = np.array(b3).reshape(3)
  return b1,b2,b3


from .geometrytk.fractional import get_fractional_function
from .geometrytk.fractional import get_fractional
from .geometrytk.fractional import fractional2real





def get_diameter(g):
  """Get the maximum distance between atoms"""
  from scipy.spatial import distance
  r = np.array(g.r) # positions
  dis = distance.cdist(r,r) # distances
  return np.max(dis)
  


def periodic_vector(g):
  """Returns a function that calculates the distance between
  two sites, using periodic boundary conditions"""
  if g.dimensionality != 2:
    print("WARNING, not 2d")
  a1 = g.a1 # first unit vector
  a2 = g.a2 # second unit vector
  shifts = [] # empty list
  for i in [-1,0,1]: # loop over closest cells
    for j in [-1,0,1]: # loop over closest cells
      shifts.append(i*a1+j*a2) # list of possible vectors
  def dis(r1,r2):
    dr = r1-r2
    rr = [dr + v for v in shifts] # loop over shifts
    rr2 = [r.dot(r) for r in rr] # distance
    mr = np.min(rr2)
    return rr[rr2.index(mr)] # return minimum
  return dis # return function
 



def periodic_zrotation(g,v=np.array([0.,0.])):
  """Returns a function that calculates the rotation between
  two sites, applying twisted boundary conditions"""
  if g.dimensionality != 2:
    print("WARNING, not 2d")
  a1 = g.a1 # first unit vector
  a2 = g.a2 # second unit vector
  shifts = [] # empty list
  index = [] # empty list
  for i in [-1,0,1]: # loop over closest cells
    for j in [-1,0,1]: # loop over closest cells
      shifts.append(i*a1+j*a2) # list of possible vectors
      index.append((i,j)) # list of indexes
  def rot(r1,r2):
    dr = r1-r2
    rr = [dr + v for v in shifts] # loop over shifts
    rr2 = [r.dot(r) for r in rr] # distance
    mr = np.min(rr2) # minimum distance
    (i,j) = index[rr2.index(mr)] # index of the joining
    return v[0]*i + v[1]*j # return rotation
  return rot







from .geometrytk.sublattice import get_sublattice


from .neighbor import neighbor_directions
from .neighbor import neighbor_cells


def replicate_array(g,v,nrep=1):
   """Replicate a certain array in a supercell"""
   if len(np.array(v).shape)>1: # not one dimensional
       return np.array([replicate_array(g,vi,nrep=nrep) for vi in v.T]).T
   else: 
       from .checkclass import number2array
       nrep = number2array(nrep,d=g.dimensionality) # as array
       nout = 1
       for n in nrep: nout *= n # multiply
       return np.array(v.tolist()*nout)


def write_profile(g,d,name="PROFILE.OUT",nrep=3,normal_order=False):
  """Write a certain profile in a file"""
  if g.dimensionality == 0: nrep = 1
  if callable(d): d = np.array([d(ri) for ri in g.r]) # call
  else: d = np.array(d) # assume it is an array
  go = g.copy() # copy geometry
  go = go.supercell(nrep) # create supercell
  d = replicate_array(g,d,nrep=nrep) # replicate
  if normal_order:
      m = np.array([go.x,go.y,go.z,d]).T
      header = "x        y       z        profile"
  else:
      m = np.array([go.x,go.y,d,go.z]).T
      header = "x        y     profile      z"
  np.savetxt(name,m,fmt='%.8f',delimiter="    ",header=header) # save in file



from .indexing import get_index



def same_site(r1,r2):
    """Check if it is the same site"""
    dr = r1-r2
    dr = dr.dot(dr)
    if dr<0.0001: return 1.0
    else: return 0.0





from .geometrytk.write import write_vasp


from .neighbor import neighbor_distances


def array2function(g,v):
    r = g.r # positions
    def f(ri):
        return array2function_jit(r,v,np.array(ri))
    return f # return function


@jit(nopython=True)
def array2function_jit(r,v,ir):
    n = len(r)
    for i in range(n):
        dr = r[i] - ir # vector difference
        dr2 = dr[0]**2 + dr[1]**2 + dr[2]**2
        if dr2<1e-3: return v[i]
    return 0.0


from .sculpt import image2island


from .geometrytk import readgeometry 
read_xyz = readgeometry.read_xyz



def get_supercell(self,nsuper,store_primal=False):
    """Creates a supercell"""
    from .checkclass import number2array
    if store_primal: # store the primal geometry
        self.primal_geometry = self.copy() 
    if self.dimensionality==0: return self # zero dimensional
    if np.array(nsuper).shape==(3,3): # if a matrix is given
        return supercelltk.non_orthogonal_supercell(self,nsuper)
    if self.dimensionality==1:
        if checkclass.is_iterable(nsuper): nsuper = nsuper[0]
        return supercell1d(self,nsuper)
    elif self.dimensionality==2:
        nsuper = number2array(nsuper,d=2) # get an array
        nsuper1 = nsuper[0] 
        nsuper2 = nsuper[1]
        if np.max(np.abs(nsuper-np.round(nsuper)))>1e-5:
            return supercelltk.target_angle_volume(self,angle=None,
                    volume=nsuper1*nsuper2)
        else: return supercell2d(self,n1=nsuper1,n2=nsuper2)
    elif self.dimensionality==3:
        nsuper = number2array(nsuper,d=3)
        if np.max(np.abs(nsuper-np.round(nsuper)))>1e-5: raise # not implementet
        nsuper1 = nsuper[0]
        nsuper2 = nsuper[1]
        nsuper3 = nsuper[2]
        s = supercell3d(self,n1=nsuper1,n2=nsuper2,n3=nsuper3)
    else: raise NotImplementedError
    s.center()
    s.get_fractional()
    return s








from .kpointstk.mapping import get_k2K_generator

