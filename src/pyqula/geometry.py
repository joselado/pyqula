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


try:
  from . import supercellf90
  use_fortran = True
except: 
  use_fortran = False
#  print("FORTRAN routines not present in geometry.py")
use_fortran = False

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
        self.x = self.x - sum(self.x)/len(self.x)
        self.y = self.y - sum(self.y)/len(self.y)
        self.z = self.z - sum(self.z)/len(self.z)
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
    def get_sublattice(self):
      """Initialize the sublattice"""
      if self.has_sublattice: self.sublattice = get_sublattice(self.r)
      else: 
          self.sublattice = get_sublattice(self.r)
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



def honeycomb_armchair_ribbon(ntetramers=10):
  """ Creates the positions of an armchair ribbon
  of width ntetramers, return a geometry class """
  from numpy import array, sqrt
  n = ntetramers
  x=array([0.0 for i in range(4*n)])
  y=array([0.0 for i in range(4*n)])
  s3=sqrt(3.0)/2.0 
  for ii in range(n):
    fi=float(ii)*s3*2.0
    i=4*ii
    x[i]=0.0
    x[i+1]=1.0
    x[i+2]=1.5
    x[i+3]=2.5 
    y[i]=fi
    y[i+1]=fi
    y[i+2]=fi+s3
    y[i+3]=fi+s3
  x=x-sum(x)/float(4*n)
  y=y-sum(y)/float(4*n)
  g = Geometry() # create geometry class
  g.x = x  # add to the x atribute
  g.y = y  # add to the y atribute
  g.z = y*0.0  # add to the y atribute
  g.celldis = 3.0 # add distance to the nearest cell
  g.a1 = np.array([3.0,0.,0.]) # add distance to the nearest cell
  g.shift_kspace = True # shift kpoint when plotting
  g.has_sublattice = True # has sublattice index
  g.sublattice = [(-1.)**i for i in range(len(x))] # subattice number
  g.name = "honeycomb_armchair_ribbon"  # name of the geometry
  g.xyz2r() # create r coordinates
  g.dimensionality = 1
  return g


honeycomb_ribbon = honeycomb_armchair_ribbon # alias



def square_ribbon(natoms):
  """ Creates the hamiltonian of a square ribbon lattice"""
  from numpy import array
  x=array([0.0 for i in range(natoms)]) # create x coordinates
  y=array([float(i) for i in range(natoms)])  # create y coordinates
  y=y-np.sum(y)/float(natoms) # shift to the center
  g = Geometry() # create geometry class
  g.x = x  # add to the x atribute
  g.y = y  # add to the y atribute
  g.z = y*0.0  # add to the y atribute
  g.celldis = 1.0 # add distance to the nearest cell
  g.a1 = np.array([1.0,0.,0.]) # add distance to the nearest cell
  g.xyz2r() # create r coordinates
  g.has_sublattice = False # does not have sublattice
  return g


def ladder(): return square_ribbon(2)

def bisquare_ribbon(ncells):
  g = square_lattice_bipartite()
  g = g.supercell((1,ncells))
  g.dimensionality = 1
  return g



def chain(n=1):
  """ Create a chain """
  g = square_ribbon(1) 
  g = g.get_supercell(n)
  g.has_sublattice = False
  g.get_fractional()
#  g.sublattice = [(-1)**i for i in range(len(g.x))]
  return g



def bichain(n=1):
    """ Create a chain """
    g = square_ribbon(1) 
    g = g.get_supercell(2)
    g.has_sublattice = True
    g.sublattice = [(-1)**i for i in range(len(g.x))]
    g = g.get_supercell(n)
    return g


def dimer():
    """Geomtry of a dimer"""
    g = bichain()
    g.dimensionality = 0
    return g




def square_tetramer_ribbon(ntetramers):
    """ Creates the hamiltonian of a square tetramer ribbon lattice"""
    from numpy import array
    natoms = ntetramers*4
    x=array([0.0 for i in range(natoms)]) # create x coordinates
    y=array([0.0 for i in range(natoms)])  # create y coordinates
    for i in range(ntetramers):
      x[4*i] = 0.0
      x[4*i+1] = 1.0
      x[4*i+2] = 1.0
      x[4*i+3] = 0.0
      y[4*i] = 2.*i
      y[4*i+1] = 2.*i
      y[4*i+2] = 2.*i +1.0
      y[4*i+3] = 2.*i +1.0
    y=y-sum(y)/float(natoms) # shift to the center
    x=x-sum(x)/float(natoms) # shift to the center
    g = Geometry() # create geometry class
    g.x = x  # add to the x atribute
    g.y = y  # add to the y atribute
    g.z = y*0.  # add to the z atribute
    g.celldis = 2.0 # add distance to the nearest cell
    g.a1 = np.array([2.0,0.,0.]) # add distance to the nearest cell
    g.shift_kspace = True # add distance to the nearest cell
    g.xyz2r() # create r coordinates
    g.has_sublattice = True # has sublattice index
    g.sublattice = [(-1.)**i for i in range(len(x))] # subattice number
    g.dimensionality = 1
    return g







def square_zigzag_ribbon(npairs):
  """ Creates the hamiltonian of a square zigzag (11) lattice"""
  from numpy import array,sqrt
  s2 = sqrt(2.) # square root of 2
  natoms = 2*npairs
  x=array([s2/4.*(-1)**i for i in range(natoms)]) # create x coordinates
  y=array([0.0 for i in range(natoms)])  # create y coordinates of pairs
  yp=array([s2*float(i) for i in range(npairs)])  # create y coordinates of pairs
  for i in range(npairs): # y position in each pair
    y[2*i] = yp[i]
    y[2*i+1] = yp[i] + s2/2.
  y=y-sum(y)/float(natoms) # shift to the center
  g = Geometry() # create geometry class
  g.x = x  # add to the x atribute
  g.y = y  # add to the y atribute
  g.celldis = s2 # add distance to the nearest cell
  g.xyz2r() # create r coordinates
  g.dimensionality = 1
  return g




def honeycomb_zigzag_ribbon(ntetramers=10):
  from numpy import array, sqrt
  n = ntetramers
  x=array([0.0 for i in range(4*n)])
  y=array([0.0 for i in range(4*n)])
  s3=sqrt(3.0)/2.0
  for ii in range(n):
    fi=-float(ii)*3.0
    i=4*ii
    x[i]=0.0
    x[i+1]=s3
    x[i+2]=s3
    x[i+3]=0.0
    y[i]=fi
    y[i+1]=fi-0.5
    y[i+2]=fi-1.5
    y[i+3]=fi-2.0
  x=x-sum(x)/float(4*n)
  y=y-sum(y)/float(4*n)
  g = Geometry() # create geometry class
  g.x = x  # add to the x atribute
  g.y = y  # add to the y atribute
  g.z = y*0.0  # add to the z atribute
  g.celldis = sqrt(3.0) # add distance to the neares cell
  g.a1 = np.array([sqrt(3.0),0.,0.]) # add distance to the nearest cell
  g.has_sublattice = True # has sublattice index
  g.sublattice = [(-1.)**i for i in range(len(x))] # subattice number
  g.name = "honeycomb_zigzag_ribbon"
  g.xyz2r() # create r coordinates
  g.dimensionality = 1
  return g



def honeycomb_lattice_zigzag():
  """ Return a honeyomb lattice with 4 atoms per unit cell"""
  from numpy import array, sqrt
  x=array([0.0 for i in range(4)])
  y=array([0.0 for i in range(4)])
  s3=sqrt(3.0)/2.0
  for ii in range(1):
    fi=-float(ii)*3.0
    i=4*ii
    x[i]=0.0
    x[i+1]=s3
    x[i+2]=s3
    x[i+3]=0.0
    y[i]=fi
    y[i+1]=fi-0.5
    y[i+2]=fi-1.5
    y[i+3]=fi-2.0
  g = Geometry() # create geometry class
  g.x = x  # add to the x atribute
  g.y = -y  # add to the y atribute
  g.z = y*0.0  # add to the z atribute
  g.a1 = np.array([sqrt(3.0),0.,0.]) # a1 distance
  g.a2 = np.array([0.,3.,0.]) # a1 distance
  g.has_sublattice = True # has sublattice index
  g.sublattice = [(-1.)**i for i in range(len(x))] # subattice number
  g.name = "honeycomb_zigzag_ribbon"
  g.xyz2r() # create r coordinates
  g.dimensionality = 2
  g.center()
  return g

def honeycomb_lattice_armchair():
    g = honeycomb_lattice_zigzag()
    g.a1,g.a2 = g.a2,-g.a1 # switch axis
    from . import sculpt
    g = sculpt.rotate_a2b(g,g.a1,np.array([1.0,0.0,0.0]))
    return g


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






################################################
########### begin 2d geometries ################
################################################

def honeycomb_lattice(n=1):
  """
  Create a honeycomb lattice
  """
  g = Geometry() # create geometry
  g.x = np.array([-0.5,0.5])
  g.y = np.array([0.0,0.0])
  g.z = np.array([0.0,0.0])
  g.a1 = np.array([3./2.,np.sqrt(3.)/2,0.]) # first lattice vector
  g.a2 = np.array([-3./2.,np.sqrt(3.)/2,0.]) # second lattice vector
  g.a3 = np.array([0.,0.,10.]) # second lattice vector
  g.dimensionality = 2 # two dimensional system
  g.xyz2r() # create r coordinates
  g.has_sublattice = True # has sublattice index
  g.sublattice = [(-1.)**i for i in range(len(g.x))] # subattice number
  g.update_reciprocal() # update reciprocal lattice vectors
  g.get_fractional()
  if n>1: return supercelltk.target_angle(g,angle=1./3.,volume=int(n),
          same_length=True) 
  return g


def buckled_honeycomb_lattice(n=1):
  """
  Return a buckled honeycomb lattice
  """
  from . import films
  g = diamond_lattice_minimal()
  g = films.geometry_film(g,nz=n)
  return g 



def triangular_lattice(n=1):
  """
  Creates a triangular lattice
  """
  g = Geometry() # create geometry
  g.x = np.array([0.0])
  g.y = np.array([0.0])
  g.z = np.array([0.0])
  g.a1 = np.array([np.sqrt(3.)/2.,1./2,0.]) # first lattice vector
  g.a2 = np.array([-np.sqrt(3.)/2.,1./2,0.]) # second lattice vector
  g.dimensionality = 2 # two dimensional system
  g.xyz2r() # create r coordinates
  g.has_sublattice = False # has sublattice index
  g.get_fractional() # update reciprocal lattice vectors
  if n>1: return supercelltk.target_angle_volume(g,angle=1./3.,volume=int(n),
          same_length=True) 
  g = sculpt.rotate_a2b(g,g.a1,np.array([1.,0.,0.]))
  return g




def triangular_lattice_tripartite():
  """
  Creates a triangular lattice with three sites per unit cell
  """
  g = triangular_lattice()
  return supercelltk.target_angle_volume(g,angle=1./3.,volume=3,
          same_length=True)



def triangular_lattice_pentapartite():
  """
  Creates a triangular lattice with five sites per unit cell
  """
  g = triangular_lattice()
  return supercelltk.target_angle(g,angle=1./3.,volume=5,same_length=True)





def triangular_ribbon(n):
  g = triangular_lattice() # create geometry
  go = g.copy() # copy geometry
  r0 = [] # empty list
  for ir in g.r:
    r0.append(ir) # supercell
    r0.append(ir+g.a1) # supercell
  rs = []
  dr = g.a1+g.a2 # displacement vector
  for i in range(n): # loop over replicas
    for ir in r0: # loop over unit cell
      rs.append(dr*i + ir) # append atom
  go.r = np.array(rs) # save coordinates
  go.r2xyz() # update
  go.a1 = g.a1 - g.a2 #
  go.center()
  go.dimensionality = 1
  # now rotate the geometry
  from . import sculpt
  go = sculpt.rotate_a2b(go,go.a1,np.array([1.0,0.0,0.0]))
  # setup the cell dis parameter (deprecated)
  go.celldis = go.a1[0]
  return go



def square_lattice_bipartite():
  """
  Creates a square lattice
  """
  g = Geometry() # create geometry
  g.x = np.array([-0.5,0.5,0.5,-0.5])
  g.y = np.array([-0.5,-0.5,0.5,0.5])
  g.z = g.x*0.
  g.a1 = np.array([2.,0.,0.]) # first lattice vector
  g.a2 = np.array([0.,2.,0.]) # second lattice vector
  g.dimensionality = 2 # two dimensional system
  g.xyz2r() # create r coordinates
  g.has_sublattice = True # has sublattice index
  g.sublattice = [(-1.)**i for i in range(len(g.x))] # subattice number
  g.update_reciprocal() # update reciprocal lattice vectors
  return g



def square_lattice():
  """
  Creates a square lattice
  """
  g = Geometry() # create geometry
  g.x = np.array([0.])
  g.y = np.array([0.])
  g.z = g.x*0.
  g.a1 = np.array([1.,0.,0.]) # first lattice vector
  g.a2 = np.array([0.,1.,0.]) # second lattice vector
  g.dimensionality = 2 # two dimensional system
  g.xyz2r() # create r coordinates
  g.has_sublattice = False # has sublattice index
  g.sublattice = [0. for i in range(len(g.r))] # subattice number
  return g


single_square_lattice = square_lattice # alias



def cubic_lattice():
  """
  Creates a cubic lattice
  """
  g = Geometry() # create geometry
  g.r = [np.array([0.,0.,0.])]
  g.x = [0.0]
  g.y = [0.0]
  g.z = [0.0]
  g.a1 = np.array([1.,0.,0.]) # first lattice vector
  g.a2 = np.array([0.,1.,0.]) # second lattice vector
  g.a3 = np.array([0.,0.,1.]) # second lattice vector
  g.dimensionality = 3 # three dimensional system
  g.has_sublattice = False # has sublattice index
  g.sublattice = [0. for i in range(len(g.r))] # subattice number
  return g


cubic_lattice_minimal = cubic_lattice


def cubic_lattice_bipartite():
  """
  Creates a cubic lattice
  """
  g = Geometry() # create geometry
  a1 = np.array([1.,0.,0.]) # first lattice vector
  a2 = np.array([0.,1.,0.]) # second lattice vector
  a3 = np.array([0.,0.,1.]) # second lattice vector
  rs = []
  ss = []
  for i in range(2):
    for j in range(2):
      for k in range(2):
        ss.append((-1)**(i+j+k)) # sublattice
        rs.append(i*a1 + j*a2 + k*a3) # position
  g.a1 = a1*2      
  g.a2 = a2*2      
  g.a3 = a3*2      
  g.sublattice = np.array(ss)
  g.r = np.array(rs)
  g.r2xyz()
  g.dimensionality = 3 # three dimensional system
  g.has_sublattice = True # has sublattice index
#  g.sublattice = [(-1.)**i for i in range(len(g.x))] # subattice number
  return g


def cubic_lieb_lattice():
  """
  Return a 3d Lieb lattice
  """
  g = cubic_lattice_bipartite()
  g = g.remove(0) # remove this atom
  return g


def lieb_lattice():
  """
  Create a 2d Lieb lattice
  """
  g = Geometry() # create geometry
  g.x = np.array([-0.5,0.5,0.5])
  g.y = np.array([-0.5,-0.5,0.5])
  g.z = g.x*0.
  g.a1 = np.array([2.,0.,0.]) # first lattice vector
  g.a2 = np.array([0.,2.,0.]) # second lattice vector
  g.dimensionality = 2 # two dimensional system
  g.xyz2r() # create r coordinates
  g.has_sublattice = True # has sublattice index
  g.sublattice = [(-1.)**i for i in range(len(g.x))] # subattice number
  return g




def kagome_lattice(n=1):
  """
  Creates a Kagome lattice
  """
  g = Geometry() # create geometry
  dx = 1./2.
  dy = np.sqrt(3)/2.
  g.x = np.array([-dx,dx,0.])
  g.y = np.array([-dy,-dy,0.0])
  g.z = np.array([0.0,0.0,0.])
  g.a1 = np.array([2.,0.,0.]) # first lattice vector
  g.a2 = np.array([1.,np.sqrt(3),0.]) # second lattice vector
  g.dimensionality = 2 # two dimensional system
  g.xyz2r() # create r coordinates
  g.center()
  g.has_sublattice = True # does not have sublattice index
  g.sublattice_number = 3 # three sublattices
  g.sublattice = [0,1,2] # the three sublattices
  if n>1: return supercelltk.target_angle(g,angle=1./3.,volume=int(n),
          same_length=True) 
  g.update_reciprocal()
  return g


def rectangular_kagome_lattice():
  """
  Creates a square kagome lattice
  """
  g = kagome_lattice()
  g = g.supercell(2) # create a supercell
  go = g.copy()
  go.a2 = np.array([0.0,np.sqrt(3)*2.0,0.0]) 
  return go







def honeycomb_lattice_square_cell():
  """
  Creates a honeycomb lattice
  """
  from .supercell import target_angle_volume
  g = honeycomb_lattice() # create geometry
  return target_angle_volume(g,volume=2,angle=.5)



def honeycomb_lattice_C6():
  """
  Geometry for a honeycomb lattice, taking a unit cell
  with C6 rotational symmetry
  """
  g = honeycomb_lattice() # create geometry
  return supercelltk.target_angle_volume(g,angle=1./3.,volume=3,
          same_length=True)







def kagome_ribbon(n=5):
  """Create a Kagome ribbon"""
  g = rectangular_kagome_lattice() # 2d geometry
  from . import ribbonizate
  g = ribbonizate.bulk2ribbon(g,n=n) # create ribbon from 2d
  return g




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
  dim = int(open("DIMENSIONALITY.OUT").read())
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
#def bulk2ribbon(h,n=10):
#  """Converts a hexagonal bulk hamiltonian into a ribbon hamiltonian"""
#  from .hexagonal import bulk2ribbon as br
#  return br(h,n=n)


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
  rs = []
  for ir in r: # loop over atoms
     store = True # store this atom
     for jr in rs: # loop over stored
       dr = ir-jr
       dr = dr.dot(dr) # distance
       if dr<0.01: store = False
     if store: # store this atom
       rs.append(ir.copy())
  return np.array(rs) # return unrepeated atoms




def get_reciprocal(a1,a2,a3):
  """Return the reciprocal lattice vectors
  By definition, ai*bj = delta_ij"""
  (ux,uy,uz) = (a1,a2,a3)
#  ux = ux/np.sqrt(ux.dot(ux))
#  uy = uy/np.sqrt(uy.dot(uy))
#  uz = uz/np.sqrt(uz.dot(uz))
  a2kn = np.matrix([ux,uy,uz]) # matrix for the change of basis
  r2a = np.matrix([ux,uy,uz]).T.I # from real space to lattice vectors
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





def cubic_diamond_lattice():
  """Return a diamond lattice"""
  fcc = [] # fcc vectors
  fcc += [np.array([0.,0.,0.])]
  fcc += [np.array([0.5,0.5,0.])]
  fcc += [np.array([0.,0.5,0.5])]
  fcc += [np.array([0.5,0.,0.5])]
  rs = fcc + [r + np.array([.25,.25,.25]) for r in fcc] # all the positions
  fac = np.sqrt(3)/4. # distance to FN
  rs = [r/fac for r in rs] # positions
  g = Geometry() # create geometry
  g.a1 = np.array([1.,0.,0.])/fac # lattice vector
  g.a2 = np.array([0.,1.,0.])/fac # lattice vector
  g.a3 = np.array([0.,0.,1.])/fac # lattice vector
  g.dimensionality = 3 # three dimensional system
  g.has_sublattice = True
  g.sublattice = np.array([1 for i in range(4)] + [-1 for i in range(4)])
  g.r = np.array(rs) # store
  g.r2xyz() # create r coordinates
  g.get_fractional()
  return g



def diamond_lattice_minimal():
  """Return a diamond lattice"""
  fcc = [] # fcc vectors
  fcc += [np.array([0.,0.,0.])]
  rs = fcc + [r + np.array([-.25,.25,.25]) for r in fcc] # all the positions
  fac = np.sqrt(3)/4. # distance to FN
  rs = [r/fac for r in rs] # positions
  g = Geometry() # create geometry
  g.a1 = np.array([-.5,.5,0.])/fac # lattice vector
  g.a2 = np.array([0.,.5,.5])/fac # lattice vector
  g.a3 = np.array([-.5,0.,.5])/fac # lattice vector
  g.dimensionality = 3 # three dimensional system
  g.has_sublattice = True
  g.sublattice = np.array([1,-1])
  g.r = np.array(rs) # store
  g.r2xyz() # create r coordinates
  g.get_fractional()
  g = sculpt.set_xy_plane(g) # a1 and a2 in the xy plane
  return g




def pyrochlore_lattice():
  """Return a pyrochlore lattice"""
  rs = [np.array([0.,0.,0.])]
  rs += [np.array([-.25,.25,0.])]
  rs += [np.array([0.,.25,.25])]
  rs += [np.array([-.25,0.,.25])]
  fac = np.sqrt(rs[1].dot(rs[1])) # distance to FN
  rs = [np.array(r)/fac for r in rs] # positions
  g = Geometry() # create geometry
  g.a1 = np.array([-.5,.5,0.])/fac # lattice vector
  g.a2 = np.array([0.,.5,.5])/fac # lattice vector
  g.a3 = np.array([-.5,0.,.5])/fac # lattice vector
  g.dimensionality = 3 # three dimensional system
  g.has_sublattice = True
  g.sublattice_number = 4 # three sublattices
  g.sublattice = [1,0,3,2] # the three sublattices
#  g.sublattice = np.array([1,-1])
  g.r = np.array(rs) # store
  g.r2xyz() # create r coordinates
  g.center() # center the geometry
  g.get_fractional()
  return g



def tetrahedral_lattice():
  """Return a single layer of the pyrochlore lattice"""
  g = pyrochlore_lattice()
  from . import films
  g = films.geometry_film(g,nz=1)
  return g







# use the cubic one as the default one
#diamond_lattice = cubic_diamond_lattice 
diamond_lattice = diamond_lattice_minimal



# two dimensional geometries
geometries2d = [] 
geometries2d += [honeycomb_lattice]
geometries2d += [square_lattice]
geometries2d += [kagome_lattice]
geometries2d += [triangular_lattice]



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
  else: d = np.array(d)
  go = g.copy() # copy geometry
  go = go.supercell(nrep) # create supercell
  d = replicate_array(g,d,nrep=nrep) # replicate
  if normal_order:
      m = np.array([go.x,go.y,go.z,d]).T
      header = "x        y       z        profile"
  else:
      m = np.array([go.x,go.y,d,go.z]).T
      header = "x        y     profile      z"
  np.savetxt(name,m,fmt='%.5f',delimiter="    ",header=header) # save in file



from .indexing import get_index



def same_site(r1,r2):
    """Check if it is the same site"""
    dr = r1-r2
    dr = dr.dot(dr)
    if dr<0.0001: return 1.0
    else: return 0.0



def hyperhoneycomb_lattice():
  """Return a hyperhoneycomb lattice"""
  g = Geometry() # create geometry
  g.a1 = np.array([np.sqrt(3.),0.,0.]) # lattice vector
  g.a2 = np.array([0.,np.sqrt(3.),0.]) # lattice vector
  g.a3 = np.array([-np.sqrt(3.)/2.,np.sqrt(3.)/2.,3.]) # lattice vector
  rs = [] # vectors in the unit cell
  rs.append(np.array([0.,0.,-0.5])+g.a1/2.) # site
  rs.append(np.array([0.,0.,0.])) # site
  rs.append(np.array([0.,0.,1.])) # site
  rs.append(np.array([0.,0.,1.5])+g.a2/2.) # site
  g.dimensionality = 3 # three dimensional system
  g.has_sublattice = True
  g.sublattice = np.array([1,-1,1,-1])
  g.r = np.array(rs) # store
  g.r2xyz() # create r coordinates
  g.get_fractional()
  return g




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
    else: raise # not implemented
    s.center()
    s.get_fractional()
    return s








from .kpointstk.mapping import get_k2K_generator


gdict = dict() # dictionary
gdict["chain"] = chain
gdict["square"] = square_lattice
gdict["honeycomb"] = honeycomb_lattice
gdict["triangular"] = triangular_lattice
gdict["kagome"] = kagome_lattice
gdict["lieb"] = lieb_lattice
gdict["pyrochlore"] = pyrochlore_lattice
gdict["diamond"] = diamond_lattice
gdict["cubic"] = cubic_lattice


def get_geometry(g):
    """Return a certain geometry"""
    if type(g)==Geometry: return g
    elif type(g)==str:
        if g in gdict: return gdict[g]() # return the geometry
    elif g is None: return get_geometry("square") # default geometry
    else: raise


def sierpinski(**kwargs):
    from .geometrytk import fractals
    return fractals.sierpinski(**kwargs)



