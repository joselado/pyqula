from __future__ import print_function
import numpy as np
from copy import deepcopy
from scipy.sparse import bmat
from scipy.sparse import csc_matrix as csc
from . import sculpt
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
    self.lattice_name = "" # lattice name
    self.atoms_names = [] # no name for the atoms
    self.atoms_have_names = False # atoms do not have names
    self.ncells = 2 # number of neighboring cells returned
  def neighbor_distances(self,**kwargs):
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
      """Sum two geometries"""
      return sum_geometries(self,g1)
  def __sub__(self,a):
      return self + (-1)*a
  def __radd__(self,g1):
      return sum_geometries(self,g1)
  def plot_geometry(self):
    """Plots the system"""
    return plot_geometry(self)
  def get_kmesh(self,**kwargs):
      """Return the k-mesh"""
      from . import klist
      return klist.kmesh(self.dimensionality,**kwargs)
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
      if n==0:
        ii = self.closest_index(r)
        return self.r[ii] # return this position
      else:
        iis = sculpt.get_closest(self,n=n,r0=r)
        return [self.r[ii] for ii in iis] # return positions
  def supercell(self,nsuper):
    """Creates a supercell"""
    if self.dimensionality==0: return self # zero dimensional
    if np.array(nsuper).shape==(3,3):
      return supercelltk.non_orthogonal_supercell(self,nsuper)
    if self.dimensionality==1:
      if checkclass.is_iterable(nsuper): nsuper = nsuper[0]
      return supercell1d(self,nsuper)
    elif self.dimensionality==2:
      try: # two numbers given
        nsuper1 = nsuper[0]
        nsuper2 = nsuper[1]
      except: # one number given
        nsuper1 = nsuper
        nsuper2 = nsuper
      if abs(nsuper1-np.round(nsuper1))>1e-6 or abs(nsuper2-np.round(nsuper2))>1e-6:
          return supercelltk.target_angle_volume(self,angle=None,
                  volume=nsuper1*nsuper2)
      else: return supercell2d(self,n1=nsuper1,n2=nsuper2)
    elif self.dimensionality==3:
      try: # two number given
        nsuper1 = nsuper[0]
        nsuper2 = nsuper[1]
        nsuper3 = nsuper[2]
      except: # one number given
        nsuper1 = nsuper
        nsuper2 = nsuper
        nsuper3 = nsuper
      s = supercell3d(self,n1=nsuper1,n2=nsuper2,n3=nsuper3)
    else: raise
    s.center()
    s.get_fractional()
    return s
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
    return get_k2K(self)
  def get_k2K_generator(self):
    R = self.get_k2K() # get the matrix
    def f(k):
#      return R@np.array(k) # to natural coordinates
      r = np.matrix(k).T # real space vectors
      return np.array((R*r).T)[0]
    return f # return function
  def reciprocal2natural(self,v):
      """
      Return a natural vector in real reciprocal coordinates
      """
      return self.get_k2K_generator()(v)
  def get_fractional(self,center=False):
    """Fractional coordinates"""
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
#  def fn_distance(self):
#    return fn_distance(self)
  def get_sublattice(self):
    """Initialize the sublattice"""
    if self.has_sublattice: self.sublattice = get_sublattice(self.r)
  def shift(self,r0):
    """Shift all the positions by r0"""
    self.x[:] -= r0[0]
    self.y[:] -= r0[1]
    self.z[:] -= r0[2]
    self.xyz2r() # update
    if self.dimensionality>0:
      self.get_fractional(center=True)
      self.fractional2real()
  def write_function(self,fun,name="FUNCTION.OUT"):
    """Write a certain function"""
    f = open(name,"w")
    ir = 0
    for r in self.r: # loop over positions
      o = fun(r) # evaluate
      f.write(str(ir)+"  ")
      for io in o:  f.write(str(io)+"  ")
      f.write("\n")
      ir += 1
    f.close() # close file
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
  def replicas(self,d=[0.,0.,0.]):
    """Return replicas of the atoms in the unit cell"""
    return [ri + self.a1*d[0] + self.a2*d[1] + self.a3*d[2] for ri in self.r]
  def multireplicas(self,n):
      """
      Return several replicas of the unit cell
      """
      return multireplicas(self,n)
  def bloch_phase(self,d,k):
    """
    Return the Bloch phase for this d vector
    """
    if self.dimensionality == 0: return 1.0
    elif self.dimensionality == 1: 
      try: kp = k[0] # extract the first component
      except: kp = k # ups, assume that it is a float
      dt = np.array(d)[0]
      kt = np.array([kp])[0]
      return np.exp(1j*dt*kt*np.pi*2.) 
    elif self.dimensionality == 2: 
      dt = np.array(d)[0:2]
      kt = np.array(k)[0:2]
      return np.exp(1j*dt.dot(kt)*np.pi*2.) 
    elif self.dimensionality == 3: 
      dt = np.array(d)[0:3]
      kt = np.array(k)[0:3]
      return np.exp(1j*dt.dot(kt)*np.pi*2.) 
  def remove(self,i=0):
      """
      Remove one site
      """
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
  def get_k2K_generator(self,toreal=False):
    """
    Function to turn a reciprocal lattice vector to natural units
    """
    R = self.get_k2K() # get matrix
    if toreal: R = R.H # transform to real coordinates
    def fun(k0):
      if len(k0)==3: k = k0 # do nothing
      elif len(k0)==2: k = np.array([k0[0],k0[1],0.]) # convert to three comp
      r = np.matrix(k).T # real space vectors
      out = np.array((R*r).T)[0] # change of basis
      if len(k0)==2: return np.array([out[0],out[1]]) # return two
      return out
    return fun
  def fractional2real(self):
    """
    Convert fractional coordinates to real coordinates
    """
    fractional2real(self)
  def real2fractional(self):
    self.get_fractional() # same function
  def get_connections(self):
    """
    Return the connections of each site
    """
    from . import neighbor
#    if self.dimensionality==0:
    if True:
      self.connections = neighbor.connections(self.r,self.r)
      return self.connections # return list
    else: raise


def add(g1,g2):
  """
  Adds two geometries
  """
  raise
  gs = Geometry()
  gs.x = np.array(g1.x.tolist() + g2.x.tolist())  
  gs.y = np.array(g1.y.tolist() + g2.y.tolist())  
  gs.z = np.array(g1.z.tolist() + g2.z.tolist())  
  gs.celldis = max([g1.celldis,g2.celldis]) 
  return gs






def squid_square(width=4,inner_radius=6,arm_length=8,arm_width=2,fill=False):
  nt = width + inner_radius # half side of the big square
  g = Geometry() # create the geometry of the system
  xc = [] # empty list
  yc = [] # empty list
  shift_y = float(arm_width-1)/2.0
  for i in range(-nt,nt+1): 
    for j in range(-nt,nt+arm_width):
      # if in the ring
      yy = float(j)-shift_y  # temporal y
      if abs(i)>inner_radius or abs(yy)>(inner_radius+shift_y) or fill: 
        xc.append(float(i))  # add x coordinate
        yc.append(yy)  # add y coordinate
  # now add right and left parts
  xr = [] # empty list
  yr = [] # empty list
  xl = [] # empty list
  yl = [] # empty list
  shift_y = float(arm_width-1)/2.0
  min_x = min(xc) - 1.0
  max_x = max(xc) + 1.0
  for i in range(arm_length):
    for j in range(arm_width): # double width of the arms
      xr.append(float(i)+max_x)
      xl.append(-float(i)+min_x)
      yr.append(float(j)-shift_y)
      yl.append(float(j)-shift_y)
  x = np.array(xr+xc+xl)  # all the x positions
  y = np.array(yr+yc+yl)  # all the y positions
  g.x = x  # add positions in x
  g.y = y  # add positions in y
  g.celldis = max(x) - min(x) +1.0 # distance to neighbor +1.0
  g.dimensionality = 1 # 0 dimensional system
  return g










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


def square_ribbon(natoms):
  """ Creates the hamiltonian of a square ribbon lattice"""
  from numpy import array
  x=array([0.0 for i in range(natoms)]) # create x coordinates
  y=array([float(i) for i in range(natoms)])  # create y coordinates
  y=y-sum(y)/float(natoms) # shift to the center
  g = Geometry() # create geometry class
  g.x = x  # add to the x atribute
  g.y = y  # add to the y atribute
  g.z = y*0.0  # add to the y atribute
  g.celldis = 1.0 # add distance to the nearest cell
  g.a1 = np.array([1.0,0.,0.]) # add distance to the nearest cell
  g.xyz2r() # create r coordinates
  g.has_sublattice = False # does not have sublattice
  return g



def bisquare_ribbon(ncells):
  g = square_lattice()
  g = g.supercell((1,ncells))
  g.dimensionality = 1
  return g



def chain(n=1):
  """ Create a chain """
  g = square_ribbon(1) 
  g = g.supercell(n)
  g.has_sublattice = False
#  g.sublattice = [(-1)**i for i in range(len(g.x))]
  return g



def bichain(n=1):
  """ Create a chain """
  g = square_ribbon(1) 
  g = g.supercell(2)
  g.has_sublattice = True
  g.sublattice = [(-1)**i for i in range(len(g.x))]
  g = g.supercell(n)
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



def honeycomb_lattice_zigzag_cell():
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



def plot_geometry(g):
   """Shows a 2d plot of the current geometry,
      returns a figure"""
   import pylab
   fig = pylab.figure() # create the figure
   sp = fig.add_subplot(111)
   x = g.x # x coordinates
   y = g.y # y coordinates
   sp.scatter(x,y,marker = "o",s=80,color="red") # create central cell
   celldis = g.celldis # distance to neighboring cell
   if not celldis== None: # if there is neighbor
     sp.scatter(x+celldis,y,marker = "o",s=80,color="black") # create right cell
     sp.scatter(x-celldis,y,marker = "o",s=80,color="black") # create left cell
   sp.set_xlabel("X")
   sp.set_xlabel("Y")
   sp.axis("equal") # same scale in axes
   fig.set_facecolor("white") # white figure  
   return fig


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
  g.update_reciprocal() # update reciprocal lattice vectors
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
  Creates a triangular lattice with three sites per unit cell
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



def square_lattice():
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



def single_square_lattice():
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
  g.sublattice = [(-1.)**i for i in range(len(g.x))] # subattice number
  return g






def cubic_lattice_minimal():
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
#  g.sublattice = [(-1.)**i for i in range(len(g.x))] # subattice number
  return g


def cubic_lattice():
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
  g = cubic_lattice()
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
  g = honeycomb_lattice() # create geometry
  go = deepcopy(g)
  go.a1 =  g.a1 + g.a2
  go.a2 = g.a1 - g.a2
  go.x = np.concatenate([g.x,g.x-g.a1[0]])  # new x coordinates
  go.y = np.concatenate([g.y,g.y-g.a1[1]])  # new y coordinates
  go.z = np.concatenate([g.z,g.z])  # new z coordinates
  go.has_sublattice = True # does not have sublattice index
  go.sublattice = [1.,-1.,1.,-1.] # sublattice index
  go.xyz2r() # create r coordinates
  go.center()
  g.update_reciprocal() # update reciprocal lattice vectors
  return go

def honeycomb_lattice_square_cell_v2():
  """
  Creates a honeycomb lattice
  """
  g = honeycomb_lattice() # create geometry
  go = deepcopy(g)
  go.a1 =  g.a1
  go.a2 = - g.a1 + 2*g.a2
  go.x = np.concatenate([g.x,g.x+g.a2[0]])  # new x coordinates
  go.y = np.concatenate([g.y,g.y+g.a2[1]])  # new y coordinates
  go.z = np.concatenate([g.z,g.z])  # new z coordinates
  go.xyz2r() # create r coordinates
  return go




def honeycomb_lattice_C6():
  """
  Geometry for a honeycomb lattice, taking a unit cell
  with C6 rotational symmetry
  """
  g = honeycomb_lattice() # create geometry
  return supercelltk.target_angle_volume(g,angle=1./3.,volume=3,
          same_length=True)
#  m = [[2,1,0],[1,2,0],[0,0,1]]
#  g = non_orthogonal_supercell(g,m)
#  g.r[0] = g.r[0] + g.a2
#  g.r[1] = g.r[1] + g.a2
#  g.r[4] = g.r[4] - g.a2 + g.a1
#  # if it looks stupid but it works, it is not stupid
#  g.r2xyz()
#  g.update_reciprocal() # update reciprocal lattice vectors
#  return g
 







def kagome_ribbon(n=5):
  """Create a Kagome ribbon"""
  g = rectangular_kagome_lattice() # 2d geometry
  from . import ribbonizate
  g = ribbonizate.bulk2ribbon(g,n=n) # create ribbon from 2d
  return g










def supercell2d(g,n1=1,n2=1,use_fortran=use_fortran):
  """ Creates a supercell for a 2d system"""
  go = g.copy() # copy geometry
  use_fortran = False
  if use_fortran:
    from . import supercellf90
    go.r = supercellf90.supercell2d(g.r,g.a1,g.a2,n1,n2)
    go.r2xyz() # update xyz
#    print("Using FORTRAN")
  else:
    nc = len(g.x) # number of atoms in a cell
    n = nc*n1*n2 # total number of positions
    a1 = g.a1 # first vector
    a2 = g.a2 # second vector
    rs = []
    for i in range(n1): 
      for j in range(n2): 
        for k in range(nc): 
          ri = i*a1 + j*a2 + g.r[k] 
          rs.append(ri) # store
    go.r = np.array(rs) # store
    go.r2xyz()
  go.a1 = go.a1*n1
  go.a2 = go.a2*n2
  # shift to zero
  go.center() 
  if g.has_sublattice: # supercell sublattice
    go.sublattice = np.concatenate([g.sublattice for i in range(n1*n2)])
  if g.atoms_have_names: # supercell sublattice
    go.atoms_names = g.atoms_names*n1*n2
  go.get_fractional() # get fractional coordinates
  return go



def supercell3d(g,n1=1,n2=1,n3=1):
  """ Creates a supercell for a 3d system"""
  nc = len(g.x) # number of atoms in a cell
  n = nc*n1*n2*n3 # total number of positions
  ro = np.array([[0.,0.,0.] for i in range(n)])
  ik = 0 # index of the atom
  a1 = g.a1 # first vector
  a2 = g.a2 # second vector
  a3 = g.a3 # third vector
  for i in range(n1): 
    for j in range(n2): 
      for l in range(n3): 
        for k in range(nc): 
          ro[ik] = a1*i + a2*j + a3*l + g.r[k] # store position
          ik += 1 # increase counter
  go = deepcopy(g) # copy geometry
  go.r = ro # store positions
  go.r2xyz() # update xyz
  go.a1 = a1*n1
  go.a2 = a2*n2
  go.a3 = a3*n3
  # shift to zero
  go.center() 
  if g.has_sublattice: # supercell sublattice
    go.sublattice = np.concatenate([g.sublattice for i in range(n1*n2*n3)])
  if g.atoms_have_names: # supercell sublattice
    go.atoms_names = g.atoms_names*n1*n2*n3
  go.get_fractional() # get fractional coordinates
  return go





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




def write_positions(g,output_file = "POSITIONS.OUT"):
  """Writes the geometry associatted with a hamiltonian in a file"""
  x = g.x  # x posiions
  y = g.y  # y posiions
  z = g.z  # z posiions
  fg = open(output_file,"w")
  fg.write(" # x    y     z   (without spin degree)\n")
  for (ix,iy,iz) in zip(x,y,z):
    fg.write(str(ix)+ "     "+str(iy)+"   "+str(iz)+"  \n")
  fg.close()


write_geometry = write_positions


def write_sublattice(g,output_file = "SUBLATTICE.OUT"):
  """Writes the geometry associatted with a hamiltonian in a file"""
  if not g.has_sublattice: return
  fg = open(output_file,"w")
  for i in g.sublattice:
    fg.write(str(i)+"  \n")
  fg.close()







def write_lattice(g,output_file = "LATTICE.OUT"):
  """Writes the lattice in a separte file"""
  open("DIMENSIONALITY.OUT","w").write(str(g.dimensionality))
  if g.dimensionality==0: return
  fg = open(output_file,"w")
  if g.dimensionality==1:
    fg.write(str(g.a1[0])+"   "+str(g.a1[1])+"  "+str(g.a1[2])+"\n")
  elif g.dimensionality==2:
    fg.write(str(g.a1[0])+"   "+str(g.a1[1])+"  "+str(g.a1[2])+"\n")
    fg.write(str(g.a2[0])+"   "+str(g.a2[1])+"  "+str(g.a2[2])+"\n")
  elif g.dimensionality==3:
    fg.write(str(g.a1[0])+"   "+str(g.a1[1])+"  "+str(g.a1[2])+"\n")
    fg.write(str(g.a2[0])+"   "+str(g.a2[1])+"  "+str(g.a2[2])+"\n")
    fg.write(str(g.a3[0])+"   "+str(g.a3[1])+"  "+str(g.a3[2])+"\n")
  else: raise
  fg.close()





def write_xyz(gin,output_file = "crystal.xyz",units=0.529,nsuper=1):
  """Writes the geometry associatted with a hamiltonian in a file"""
  g = gin.copy() # copy geometry
  if g.atoms_have_names: g = remove_duplicated(g)
  else: g = gin.copy()
  if g.dimensionality>0: # create supercell
    if nsuper>1:
      g = g.supercell(nsuper)
  x = g.x*units  # x posiions
  y = g.y*units  # y posiions
  z = g.z*units  # z posiions
  fg = open(output_file,"w")
  # get hte names of the atoms
  if g.atoms_have_names:
    names = g.atoms_names
  else:
    names = ["C" for ix in x ] # create names
  # check that there are as many names as positions
  if len(names)!=len(x): raise
  fg.write(str(len(x))+"\nGenerated with Python\n") # number of atoms
  for (n,ix,iy,iz) in zip(names,x,y,z):
    fg.write(n+"   "+str(ix)+ "     "+str(iy)+"   "+str(iz)+"  \n")
  fg.close()




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
  return rs # return unrepeated atoms




def get_k2K(g):
  """Return a matrix that converts vectors
  in the reciprocal space into natural units, useful for drawing
  2D quantities"""
  if g.dimensionality == 2:
    (ux,uy,uz) = (g.a1,g.a2,np.array([0.,0.,1]))
  elif g.dimensionality == 3:
    (ux,uy,uz) = (g.a1,g.a2,g.a3)
  else: raise
  ux = ux/np.sqrt(ux.dot(ux))
  uy = uy/np.sqrt(uy.dot(uy))
  uz = uz/np.sqrt(uz.dot(uz))
  a2kn = np.matrix([ux,uy,uz]) # matrix for the change of basis
  r2a = np.matrix([ux,uy,uz]).T.I # from real space to lattice vectors
  R = a2kn@r2a@a2kn.T # rotation matrix
  return R



def get_reciprocal(a1,a2,a3):
  """Return a matrix that converts vectors
  in the reciprocal space into natural units, useful for drawing
  2D quantities"""
  (ux,uy,uz) = (a1,a2,a3)
  ux = ux/np.sqrt(ux.dot(ux))
  uy = uy/np.sqrt(uy.dot(uy))
  uz = uz/np.sqrt(uz.dot(uz))
  a2kn = np.matrix([ux,uy,uz]) # matrix for the change of basis
  r2a = np.matrix([ux,uy,uz]).T.I # from real space to lattice vectors
  b1,b2,b3 = r2a[0,:],r2a[1,:],r2a[2,:]
  b1 = np.array(b1).reshape(3)
  b2 = np.array(b2).reshape(3)
  b3 = np.array(b3).reshape(3)
  return b1,b2,b3


def get_fractional_function(g,center=False):
  """Get fractional coordinates"""
#  if g.dimensionality<2: raise # stop 
  dim = g.dimensionality # dimensionality
  if dim==0: return lambda x: x
  elif dim==1: # one dimensional
    R = np.array([g.a1,[0.,1.,0.],[0.,0.,1.]]).T # transformation matrix
    if np.max(np.abs(g.a1[1:2]))>1e-6: raise
  elif dim==2: # two dimensional
    R = np.array([g.a1,g.a2,[0.,0.,1.]]).T # transformation matrix
    if np.abs(g.a1[2])>1e-6 or np.abs(g.a2[2])>1e-6: raise
  elif dim==3:
    R = np.array([g.a1,g.a2,g.a3]).T # transformation matrix
  else: raise
  g.has_fractional = True # has fractional coordinates
  L = lg.inv(R) # inverse matrix
  def f(r):
      if center: return (L@np.array(r))%1.0  # transform
      else: return L@np.array(r)  # transform
  return f


def get_fractional(g,center=False):
  dim = g.dimensionality # dimensionality
  if dim==0: return
  f = get_fractional_function(g,center=center)
  store = [f(r) for r in g.r] # empty list
  store = np.array(store) # convert to array
  # if you remove the shift the Berry Green formalism does not work
  if dim>0: g.frac_x = store[:,0]
  if dim>1: g.frac_y = store[:,1]
  if dim>2: g.frac_z = store[:,2]
  g.frac_r = store



def fractional2real(self):
  """Write real coordinates using the fractional ones"""
  if self.dimensionality==0: raise
  elif self.dimensionality==1: # 1D
    self.x = self.frac_x*self.a1[0] 
  elif self.dimensionality==2: # 2D
    self.x = self.frac_x*self.a1[0] +  self.frac_y*self.a2[0] 
    self.y = self.frac_x*self.a1[1] +  self.frac_y*self.a2[1] 
  elif self.dimensionality==3: # 3D
    self.x = self.frac_x*self.a1[0] +  self.frac_y*self.a2[0] + self.frac_z*self.a3[0] 
    self.y = self.frac_x*self.a1[1] +  self.frac_y*self.a2[1] + self.frac_z*self.a3[1] 
    self.z = self.frac_x*self.a1[2] +  self.frac_y*self.a2[2] + self.frac_z*self.a3[2] 
  else: raise
  self.xyz2r() # update xyz
  self.center() # center




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





def get_sublattice(rs):
  """Return indexes of the sublattice, assuming that there is sublattice"""
  n = len(rs) # loop over positions
  sublattice = [0 for i in range(n)] # list for the sublattice
  ii = np.random.randint(n)
  sublattice[ii] = -1 # start with this atom
  print("Looking for a sublattice")
  while True: # infinite loop
    for i in range(n): # look for a neighbor for site i
      if sublattice[i]!=0: continue
      for j in range(n): # loop over the rest of the atoms
        if sublattice[j]==0: continue # next one
        dr = rs[i] - rs[j] # distance to site i
        if 0.9<dr.dot(dr)<1.01: # if NN and sublattice 
          sublattice[i] = -sublattice[j] + 0 # opposite
          continue # next one
    if np.min(np.abs(sublattice))!=0: break
  return sublattice


from .neighbor import neighbor_directions
from .neighbor import neighbor_cells


def replicate_array(g,v,nrep=1):
   """Replicate a certain array in a supercell"""
   if len(np.array(v).shape)>1: # not one dimensional
       return np.array([replicate_array(g,vi,nrep=nrep) for vi in v.T]).T
   else: return np.array(v.tolist()*(nrep**g.dimensionality))


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










def multireplicas(self,n):
    """
    Return several replicas of the unit cell, it is similar
    to supercell but without the shift of the center and going to positive
    and negative positions
    """
    if n==0: return self.r # return positions
    out = [] # empty list with positions
    dl = self.neighbor_directions(n) # list with neighboring cells to take
    if self.dimensionality==0: return self.r
    else: # bigger dimensionality
        for d in dl:  out += self.replicas(d) # add this direction
    return np.array(out)



def write_vasp(g0,s=1.42):
    """Turn a geometry into vasp geometry"""
    g = g0.copy() # copy geometry
    if g.dimensionality==3: pass
    elif g.dimensionality==2:
        g.r[:,2] -= np.min(g.r[:,2])
        z = np.max(g.r[:,2]) - np.min(g.r[:,2])
        a3 = np.array([0.,0.,z+9.0])
        g.a3 = a3 # set the lattice vector
        g.dimensionality = 3
        g.get_fractional() # get fractional coordinates
    else: raise # not implemented
    f = open("vasp.vasp","w") # input file
    f.write("Structure\n 1.0\n")
    for i in range(3): f.write(str(s*g.a1[i])+"  ")
    f.write("\n")
    for i in range(3): f.write(str(s*g.a2[i])+"  ")
    f.write("\n")
    for i in range(3): f.write(str(s*g.a3[i])+"  ")
    # write the atoms
    if not g.atoms_have_names: # no name provided
      f.write("\n C\n "+str(len(g.r))+"\n Direct\n")
      # write all the atoms in fractional coordinates
      for ir in g.frac_r:
          for i in range(3): f.write(str(ir[i])+"  ")
          f.write("\n")
    else: # atoms have labels
        namedict = dict() # dictionary
        for (key,n) in zip(g.atoms_names,range(len(g.r))):
            if not key in namedict: 
                namedict[key] = [n] # store
            else: namedict[key].append(n) # store
        f.write("\n") # next line
        for key in namedict: f.write(str(key)+"   ")
        f.write("\n") # next line
        for key in namedict: f.write(str(len(namedict[key]))+"   ")
        f.write("\n Direct \n") # next line
        for key in namedict: # loop over types
            ns = namedict[key] # list with atoms
            for ii in ns: # loop over atoms
              for i in range(3): f.write(str(g.frac_r[ii][i])+"  ")
              f.write("\n")
    f.close()



def sum_geometries(g1,g2):
    """Sum two geometries"""
    if type(g2)==Geometry:
        if g1.dimensionality!=g2.dimensionality: raise
        g = g1.copy()
        g.r = np.concatenate([g1.r,g2.r])
        g.r2xyz()
        if g.has_sublattice:
            g.sublattice = np.concatenate([g1.sublattice,g2.sublattice])
        if g.atoms_have_names:
            g.atoms_names = np.concatenate([g1.atoms_names,g2.atoms_names])
        return g
    elif type(g2)==np.ndarray: # array input
        g = g1.copy() # copy geometry
        g.r = g.r + g2 # shift all the positions
        g.r2xyz()
        return g
    else: 
        print(type(g2))
        raise

from .neighbor import neighbor_distances


def array2function(g,v):
    r = g.r # positions
    def f(ri):
        return array2function_jit(r,v,ri)
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

