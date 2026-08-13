# Lattice/ribbon factory functions for Geometry (honeycomb_lattice,
# kagome_lattice, chain, ...), moved out of geometry.py so that module
# stays the thin Geometry class + generic geometry infrastructure.
#
# geometry.py imports this module (via "from .geometrytk.lattices import
# *") to re-export these functions as geometry.honeycomb_lattice etc., and
# this module needs the Geometry class back to construct instances -- a
# genuine two-way dependency. Importing geometry at module level here
# would race that re-export: whichever of the two modules is imported
# LAST would find the other only partially initialized (still missing
# either Geometry or these factory functions, depending on import order).
# _Geometry()/_geometry_class() defer the import to first CALL instead, by
# which point both modules have always finished loading.
import numpy as np
from .. import sculpt
from .. import supercell as supercelltk
from ..ribbon import bulk2ribbon


def _geometry_class():
    from ..geometry import Geometry
    return Geometry


def _Geometry(*args, **kwargs):
    return _geometry_class()(*args, **kwargs)


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
  g = _Geometry() # create geometry class
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
  g = _Geometry() # create geometry class
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
  g = g.get_supercell((1,ncells))
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
    g = _Geometry() # create geometry class
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
  g = _Geometry() # create geometry class
  g.x = x  # add to the x atribute
  g.y = y  # add to the y atribute
  g.z = y*0.0  # add to the z atribute
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
  g = _Geometry() # create geometry class
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
  g = _Geometry() # create geometry class
  g.x = x  # add to the x atribute
  g.y = -y  # add to the y atribute
  g.z = y*0.0  # add to the y atribute
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
    g = sculpt.rotate_a2b(g,g.a1,np.array([1.0,0.0,0.0]))
    return g


################################################
########### begin 2d geometries ################
################################################

def honeycomb_lattice(n=1):
  """
  Create a honeycomb lattice
  """
  g = _Geometry() # create geometry
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
  from .. import films
  g = diamond_lattice_minimal()
  g = films.geometry_film(g,nz=n)
  return g



def triangular_lattice(n=1):
  """
  Creates a triangular lattice
  """
  g = _Geometry() # create geometry
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
  rs = [] # empty list
  rs.append([0.,0.,0.]) # first position
  rs.append([1.,0.,0.]) # second position
  rs.append([1./2.,np.sqrt(3.)/2.,0.]) # third position
  rs = np.array(rs)
  g = _Geometry() # create geometry
  g.r = np.array(rs) # store array
  g.r2xyz() # tranform
  g.has_sublattice = False # has sublattice index
  g.a1 = np.array(rs[1]+rs[2])
  g.a2 = np.array(-rs[1]+2*rs[2])
  g.dimensionality = 2 # two dimensional system
  g.get_fractional() # update reciprocal lattice vectors
  return g
#  g = triangular_lattice()
#  return supercelltk.target_angle_volume(g,angle=1./3.,volume=3,
#          same_length=True)



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
  go = sculpt.rotate_a2b(go,go.a1,np.array([1.0,0.0,0.0]))
  # setup the cell dis parameter (deprecated)
  go.celldis = go.a1[0]
  return go



def square_lattice_bipartite():
  """
  Creates a square lattice
  """
  g = _Geometry() # create geometry
  g.x = np.array([-0.5,0.5,0.5,-0.5])
  g.y = np.array([-0.5,-0.5,0.5,0.5])
  g.z = g.x*0.
  g.a1 = np.array([2.,0.,0.]) # first lattice vector
  g.a2 = np.array([0.,2.,0.]) # second lattice vector
  g.dimensionality = 2 # two dimensional system
  g.xyz2r() # create r coordinates
  g.has_sublattice = True # has sublattice index
  g.sublattice = [-1,1,1,-1] # sublattice number
  g.update_reciprocal() # update reciprocal lattice vectors
  return g



def square_lattice():
  """
  Creates a square lattice
  """
  g = _Geometry() # create geometry
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
  g = _Geometry() # create geometry
  g.r = np.array([[0.,0.,0.]])
  g.x = np.array([0.0])
  g.y = np.array([0.0])
  g.z = np.array([0.0])
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
  g = _Geometry() # create geometry
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


def lieb_ribbon(n):
    """Return a ribbon of the Lieb lattice"""
    g = lieb_lattice() # 2d lattice
    g = bulk2ribbon(g,n=n+1) # make a ribbon
    from ..sculpt import remove_unibonded
    g = remove_unibonded(g) # remove single bonded sites
    return g




def lieb_lattice():
  """
  Create a 2d Lieb lattice
  """
  g = _Geometry() # create geometry
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
  g = _Geometry() # create geometry
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
  g.get_fractional()
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
  from ..supercell import target_angle_volume
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
  from .. import ribbonizate
  g = ribbonizate.bulk2ribbon(g,n=n) # create ribbon from 2d
  return g



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
  g = _Geometry() # create geometry
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
  g = _Geometry() # create geometry
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


# use the cubic one as the default one
#diamond_lattice = cubic_diamond_lattice
diamond_lattice = diamond_lattice_minimal


def pyrochlore_lattice():
  """Return a pyrochlore lattice"""
  rs = [np.array([0.,0.,0.])]
  rs += [np.array([-.25,.25,0.])]
  rs += [np.array([0.,.25,.25])]
  rs += [np.array([-.25,0.,.25])]
  fac = np.sqrt(rs[1].dot(rs[1])) # distance to FN
  rs = [np.array(r)/fac for r in rs] # positions
  g = _Geometry() # create geometry
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
  from .. import films
  g = films.geometry_film(g,nz=1)
  return g


def hyperhoneycomb_lattice():
  """Return a hyperhoneycomb lattice"""
  g = _Geometry() # create geometry
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


def sierpinski(**kwargs):
    from . import fractals
    return fractals.sierpinski(**kwargs)


# two dimensional geometries
geometries2d = []
geometries2d += [honeycomb_lattice]
geometries2d += [square_lattice]
geometries2d += [kagome_lattice]
geometries2d += [triangular_lattice]


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
    if type(g)==_geometry_class(): return g
    elif type(g)==str:
        if g in gdict: return gdict[g]() # return the geometry
    elif g is None: return get_geometry("square") # default geometry
    else: raise
