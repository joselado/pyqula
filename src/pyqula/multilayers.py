# library to create multilayer systems
from copy import deepcopy
import numpy as np
from scipy.sparse import csc_matrix as csc
from scipy.sparse import csc_matrix 
from scipy.sparse import bmat
from . import geometry


def honeycomb_ribbon(n=1,rtype="zigzag",stype="AB",has_spin=True,
                     t = 0.1,efield=0.1):
  """Return the hamiltonian of a ribbon"""
  if rtype=="zigzag":  g = geometry.honeycomb_zigzag_ribbon(n)
  elif rtype=="armchair":  g = geometry.honeycomb_zigzag_ribbon(n)
  else: raise
  if rtype=="amrchair": raise # not implemented yet
  h = g.get_hamiltonian(has_spin=has_spin) # hamiltonian
  if stype=="ABC": # trilayer ABC
    h = build_honeycomb_trilayer(h,t,mvl=[0.,1.])
  elif stype=="AB": # trilayer ABC
    h = build_honeycomb_bilayer(h,t,mvl=[0.,1.])
  add_electric_field(h,e=efield)
  return h




def bilayer_aa(h,t = 0.1):
  """ Creates a bilayer from a honeycomb ribbon"""
  nlayers = 2 # number of layers
  g = h.geometry # get the geometry
  go = deepcopy(g) # copy the geometry
  go.x = [] 
  go.y = []
  go.z = []
  if g.name == "honeycomb_armchair_ribbon": dx ,dy = 1. ,0.
  if g.name == "honeycomb_zigzag_ribbon": dx ,dy = 0. ,-1.
  for (xi,yi) in zip(g.x,g.y):  # modify the geometry
    go.x.append(xi)
    go.x.append(xi+dx)
    go.y.append(yi)
    go.y.append(yi+dy)
    go.z.append(1.)
    go.z.append(-1.)
  go.x,go.y,go.z = np.array(go.x),np.array(go.y),np.array(go.z) # put arrays
  # now modify the hamiltonian
  ho = deepcopy(h)
  n = len(ho.intra) # dimension
  intra = [[0. for i in range(2*n)] for j in range(2*n)]
  inter = [[0. for i in range(2*n)] for j in range(2*n)]
  norb = n # number of orbitals
  # get the atoms which hop according to monolayer type...
  if h.has_spin: 
    norb = norb/2
    tl = [] # interlayer pairs
    x, y, z = go.x, go.y, go.z
    for i in range(len(x)): # loop over atoms
      for j in range(len(x)): # loop over atoms
        if 1.9 < np.abs(z[i]-z[j]) < 2.1: # if in contiguous layers 
          dd = (x[i]-x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2
          if 3.9<dd<4.1:
            tl.append([i,j])
    for i in range(norb):
      for j in range(norb):  # assign interlayer hopping
        for s in range(2):
         for l in range(nlayers):
          intra[2*nlayers*i+s+2*l][2*nlayers*j+s+2*l] = h.intra[2*i+s,2*j+s]
          inter[2*nlayers*i+s+2*l][2*nlayers*j+s+2*l] = h.inter[2*i+s,2*j+s]
    # now put the interlayer hopping
    for p in tl:
      for s in range(2): # loop over spin
        intra[2*p[0]+s][2*p[1]+s] = t  
        intra[2*p[1]+s][2*p[0]+s] = np.conjugate(t)  
  else: raise # not implemented...
  if h.has_eh: raise # not implemented ....
  ho.intra = np.matrix(intra)
  ho.inter = np.matrix(inter)
  ho.geometry = go
  return ho   



def add_electric_field(h,e = 0.0):
  """Adds electric field to the system"""
  h.add_onsite(lambda r: e*r[2])
  return h

def multilayered_hamiltonian(h,dr=np.array([0.,0.,0.])):
  """ Creates a multilayered hamiltonian by adding several layers """






def add_interlayer(t,ri,rj,has_spin=True,is_sparse=True):
  """Calculate interlayer coupling"""
  m = np.matrix([[0. for i in ri] for j in rj])
  if has_spin: m = bmat([[csc(m),None],[None,csc(m)]]).todense()
  zi = [r[2] for r in ri]
  zj = [r[2] for r in rj]
  for i in range(len(ri)): # add the interlayer
    for j in range(len(ri)): # add the interlayer
      rij = ri[i] - rj[j] # distance between atoms
      dz = zi[i] - zj[j] # vertical distance
      if (3.99<rij.dot(rij)<4.01) and (3.99<(dz*dz)<4.01): # check if connect
        if has_spin: # spin polarized
          m[2*i,2*j] = t
          m[2*i+1,2*j+1] = t
        else:  # spin unpolarized
          m[i,j] = t
  return m







def build_honeycomb_bilayer(h,t=0.1,mvl = None ):
  """Builds a multilayer based on a hamiltonian, it is assumed that
  there are no new intercell hoppings in the multilayer"""
  g = h.geometry  # save geometry
  ho = deepcopy(h) # copy the hamiltonian
  go = deepcopy(g) # copy the geometry
  if mvl is None: # if not provided assume firs neighbors
    mvl = g.r[0] - g.r[1]
  def mono2bi(m):
    """Increase the size of the matrices"""    
    if h.is_sparse:
      return bmat([[csc(m),None],[None,csc(m)]])
    else:
      return bmat([[csc(m),None],[None,csc(m)]]).todense()  
  # modify the geometry
  x,y,z = np.array(g.x), np.array(g.y) ,np.array(g.z) # store positions
  go.x = np.concatenate([x,x+mvl[0]])  # move 
  go.y = np.concatenate([y,y+mvl[1]])  # move
  go.z = np.concatenate([z-1.0,z+1.0])  # separate two units
  go.xyz2r() # update r coordinates
  if g.has_sublattice: # if has sublattice, keep the indexes
    go.sublattice = g.sublattice*2 # two times
  ho.geometry = go # update geometry
  # modify the hamiltonian
  ho.intra = mono2bi(h.intra)  # increase intracell matrix
  ho.intra = ho.intra + add_interlayer(t,go.r,go.r,has_spin=h.has_spin) # add interlayer coupling
  if h.dimensionality==2: # two dimensional system
    ho.tx = mono2bi(h.tx)  # increase intracell matrix
    ho.ty = mono2bi(h.ty)  # increase intracell matrix
    ho.txy = mono2bi(h.txy)  # increase intracell matrix
    ho.txmy = mono2bi(h.txmy)  # increase intracell matrix
  elif h.dimensionality==1: # one dimensional system
    ho.inter = mono2bi(h.inter)  # increase intercell matrix
    dx = g.a1
    ho.inter += add_interlayer(t,go.r,go.r+dx) # add interlayer coupling
  else:
    raise
  return ho



def build_honeycomb_trilayer(h,t=0.1,mvl=None):
  """Builds a multilayer based on a hamiltonian, it is assumed that
  there are no new intercell hoppings in the multilayer"""
  g = h.geometry  # save geometry
  ho = deepcopy(h) # copy the hamiltonian
  go = deepcopy(g) # copy the geometry
  if mvl is None: # if not provided assume firs neighbors
    mvl = g.r[0] - g.r[1]
  def mono2tri(m):
    """Increase the size of the matrices"""    
    mo = [[None for i in range(3)] for j in range(3)]
    for i in range(3): 
      mo[i][i] = csc(m)
    return bmat(mo).todense()
  # modify the geometry
  x,y,z = np.array(g.x), np.array(g.y) ,np.array(g.z) # store positions
  go.x = np.concatenate([x-mvl[0],x,x+mvl[0]])  # move one unit to the right
  go.y = np.concatenate([y-mvl[1],y,y+mvl[1]])  # leave invariant
  go.z = np.concatenate([z-2.0,z,z+2.0])  # separate two units
  go.xyz2r() # update r coordinates
  if g.has_sublattice: # if has sublattice, keep the indexes
    go.sublattice = g.sublattice*3 # three times
  ho.geometry = go # update geometry
  # modify the hamiltonian
  ho.intra = mono2tri(h.intra)  # increase intracell matrix
  ho.intra += add_interlayer(t,go.r,go.r) # add interlayer coupling
  if h.dimensionality==2: # two dimensional system
    ho.tx = mono2tri(h.tx)  # increase intracell matrix
    ho.ty = mono2tri(h.ty)  # increase intracell matrix
    ho.txy = mono2tri(h.txy)  # increase intracell matrix
    ho.txmy = mono2tri(h.txmy)  # increase intracell matrix
  elif h.dimensionality==1: # one dimensional system
    ho.inter = mono2tri(h.inter)  # increase intercell matrix
    dx = g.a1
    ho.inter += add_interlayer(t,go.r,go.r+dx) # add interlayer coupling
  else:
    raise
  ## add sublattice index, might break
  return ho




def get_armchair_bilayer(n=10):
  """Get hamiltonian of an armchair bilayer"""
  g = geometry.honeycomb_armchair_ribbon(30)
  drs = [np.array([0.,0.,-1.]),np.array([np.sqrt(3.)/2.,0.,1.])]
  g = geometry.apilate(g,drs=drs)
  h = g.get_hamiltonian()
  return h


def get_zigzag_bilayer(n=10):
  """Get hamiltonian of a zigzag bilayer"""
  g = geometry.honeycomb_zigzag_ribbon(40)
  drs = [np.array([0.,0.,-1.]),np.array([0.,1.,1.])]
  g = geometry.apilate(g,drs=drs)
  h = g.get_hamiltonian()
  return h



def bilayer_geometry(g,mvl=None,dz=2.0):
  if mvl is None: # if not provided assume firs neighbors
    mvl = g.r[0] - g.r[1]
  go = g.copy() # copy geometry
  x,y,z = np.array(g.x), np.array(g.y) ,np.array(g.z) # store positions
  go.x = np.concatenate([x,x+mvl[0]])  # move 
  go.y = np.concatenate([y,y+mvl[1]])  # move
  go.z = np.concatenate([z-dz/2,z+dz/2])  # separate two units
  go.xyz2r() # update r coordinates
  if g.has_sublattice: # if has sublattice, keep the indexes
    sl = g.sublattice
    go.sublattice = np.concatenate([sl,sl]) # two times
  return go



def get_geometry(name,dz=2.0,armchair=True):
  """Return the geometry for multilayer graphene"""
  g = geometry.honeycomb_lattice() # honeycomb lattice
  if name=="AA":
    g = bilayer_geometry(g,mvl=[0.,0.],dz=1.0)
  elif name=="AB":
    g = bilayer_geometry(g,mvl=None,dz=1.0)
  else: raise
  g.has_sublattice = False # no sublattice
  if armchair: g = g.supercell([[1,-1,0],[0,1,0],[0,0,1]]) # armchair unit cell
  g.z *= dz ; g.xyz2r() # increase distance
  return g



def multilayer_hopping(dz=2.0,ti=0.3):
  """Function to calcualte the hopping in multilayer systems"""
  def fun(r1,r2):
    dr = r1-r2
    if 0.9<dr.dot(dr)<1.1: return 1.0 # first neighbor
    if abs(dr.dot(dr)-dz**2)<0.1: 
      if abs(dr[2]**2-dz**2)<0.1: 
        return ti # first neighbor
    return 0.0
  return fun

