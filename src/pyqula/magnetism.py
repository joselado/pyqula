from scipy.sparse import coo_matrix,bmat
from .rotate_spin import sx,sy,sz
from .increase_hilbert import get_spinless2full,get_spinful2full
import numpy as np
from . import checkclass
from . import geometry

def float2array(z):
    if checkclass.is_iterable(z): return z # iterable, input is an array
    else: return [0.,0.,z] # input is a number

def add_zeeman(h,zeeman=[0.0,0.0,0.0]):
  """ Add Zeeman to the hamiltonian """
  # convert the input into a list
  def evaluate_J(z,r,i):
    if checkclass.is_iterable(z): # it is a list/array
        if checkclass.is_iterable(z[0]):  # each element is a list/array
            return np.array(z[i]) # iterable, input is an array
        out = [0.,0.,0.] # not iterable
        for j in range(len(z)): # loop over elements
            if callable(z[j]):  # each element is a function
               out[j] = z[j](r) # call the function
            else: # if it is number
               out[j] = z[j]
        return np.array(out)
    elif callable(z): # it is a function
        m = z(r) # call
        if checkclass.is_iterable(m): return np.array(m) # it is an array
        else: return np.array([0.,0.,m]) # number
    else: return np.array([0.,0.,z]) # just a number
  from scipy.sparse import coo_matrix as coo
  from scipy.sparse import bmat
  if h.has_spin: # only if the system has spin
   # no = h.num_orbitals # number of orbitals (without spin)
    no = len(h.geometry.r) # number of orbitals (without spin)
    # create matrix to add to the hamiltonian
    bzee = [[None for i in range(no)] for j in range(no)]
    # assign diagonal terms
    r = h.geometry.r  # z position
    for i in range(no):
        JJ = evaluate_J(zeeman,r[i],i) # evaluate the exchange
        bzee[i][i] = JJ[0]*sx+JJ[1]*sy+JJ[2]*sz
    bzee = bmat(bzee) # create matrix
    h.intra = h.intra + h.spinful2full(bzee) # Add matrix 
  if not h.has_spin:  # still have to implement this...
    raise





def add_antiferromagnetism(h,m):
  """ Adds to the intracell matrix an antiferromagnetic imbalance """
  if not h.has_spin: h.turn_spinful()
  intra = h.intra # intracell hopping
  if h.geometry.has_sublattice: pass  # if has sublattice
  else: # if does not have sublattice
#    print("WARNING, no sublattice present")
    return 0. # if does not have sublattice
  if h.has_spin:
    natoms = len(h.geometry.x) # number of atoms
    out = [[None for j in range(natoms)] for i in range(natoms)] # output matrix
    # create the array
    if checkclass.is_iterable(m): # iterable, input is an array
      if len(m)!=len(h.geometry.r): raise
      mass = m # use the input array
    elif callable(m): # input is a function
      mass = [m(h.geometry.r[i]) for i in range(natoms)] # call the function
    else: # assume it is a float
      mass = [m for i in range(natoms)] # create list
    for i in range(natoms): # loop over atoms
      mi = mass[i] # select the element
      # add contribution to the Hamiltonian
      mi = float2array(mi) # convert to array
      out[i][i] = (sx*mi[0] + sy*mi[1] + sz*mi[2])*h.geometry.sublattice[i]
    out = bmat(out) # turn into a matrix
    h.intra = h.intra + h.spinful2full(out) # Add matrix 
  else:
    print("no AF for unpolarized hamiltonian")
    raise






def add_magnetism(h,m):
  """ Adds magnetism to the intracell hopping"""
  intra = h.intra # intracell hopping
  if h.has_spin:
    natoms = len(h.geometry.r) # number of atoms
    # create the array
    out = [[None for j in range(natoms)] for i in range(natoms)] # output matrix
    if checkclass.is_iterable(m):
      if checkclass.is_iterable(m[0]) and len(m)==natoms: # input is an array
        mass = m # use as arrays
      elif len(m)==3: # single exchange provided
        mass = [m for i in range(natoms)] # use as arrays
      else: raise
    elif callable(m): # input is a function
      mass = [m(h.geometry.r[i]) for i in range(natoms)] # call the function
    else: 
      print("Wrong input in add_magnetism")
      raise 
    for i in range(natoms): # loop over atoms
      mi = mass[i] # select the element
      # add contribution to the Hamiltonian
      out[i][i] = sx*mi[0] + sy*mi[1] + sz*mi[2]
    out = bmat(out) # turn into a matrix
    h.intra = h.intra + h.spinful2full(out) # Add matrix 
  else:
    print("no AF for unpolarized hamiltonian")
    raise





def add_frustrated_antiferromagnetism(h,m):
  """Add frustrated magnetism"""
  if h.geometry.sublattice_number==3:
    g = geometry.kagome_lattice()
  elif h.geometry.sublattice_number==4:
    g = geometry.pyrochlore_lattice()
    g.center()
  else: raise # not implemented
  ms = []
  for i in range(len(h.geometry.r)): # loop
    ii = h.geometry.sublattice[i] # index of the sublattice
    if callable(m):
      ms.append(-g.r[int(ii)]*m(h.geometry.r[i])) # save this one
    else:
      ms.append(-g.r[int(ii)]*m) # save this one
  h.add_magnetism(ms) # add the magnetization





def get_magnetization(h,nkp=20):
  """Return the magnetization of the system"""
  totkp = nkp**(h.dimensionality)
  nat = h.intra.shape[0]//2 # number of atoms
  eigvals,eigvecs = h.get_eigenvectors(nk=nkp)
  voccs = [] # accupied vectors
  eoccs = [] # accupied eigenvalues
  occs = [] # accupied eigenvalues
  for (e,v) in zip(eigvals,eigvecs): # loop over eigenvals,eigenvecs
    if e<0.0000001:  # if level is filled, add contribution
      voccs.append(v) # store
      eoccs.append(e) # store
  pdup = np.array([[2*i,2*i] for i in range(nat)]) # up density
  pddn = pdup + 1 # down density
  pxc = np.array([[2*i,2*i+1] for i in range(nat)]) # exchange
  from . import correlatorsf90
  vdup = correlatorsf90.correlators(voccs,pdup)/totkp
  vddn = correlatorsf90.correlators(voccs,pddn)/totkp
  vxc = correlatorsf90.correlators(voccs,pxc)/totkp
  magnetization = np.array([vxc.real,vxc.imag,vdup-vddn]).transpose().real
  from .scftypes import write_magnetization
  write_magnetization(magnetization)


