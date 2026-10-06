from scipy.sparse import coo_matrix,bmat
from .rotate_spin import sx,sy,sz
from .increase_hilbert import get_spinless2full,get_spinful2full
import numpy as np
from . import checkclass
from . import geometry
from .check import require_spin

def float2array(z):
    if checkclass.is_iterable(z): return z # iterable, input is an array
    else: return [0.,0.,z] # input is a number


def exchange_matrix(ms):
    """Block-diagonal matrix with m_x sx + m_y sy + m_z sz on the spin
    block of each site, ms the (n,3) array of the n local fields, built
    in time linear in n (see algebra.block_diagonal)"""
    from .algebra import block_diagonal
    ms = np.asarray(ms,dtype=np.complex128)
    if ms.ndim!=2 or ms.shape[1]!=3:
        raise ValueError("the exchange field needs three components "
                "[mx,my,mz] per site, got an array of shape "+str(ms.shape))
    pauli = np.array([[[0.,1.],[1.,0.]],[[0.,-1j],[1j,0.]],
        [[1.,0.],[0.,-1.]]],dtype=np.complex128) # sx, sy, sz
    return block_diagonal(np.einsum("na,abc->nbc",ms,pauli))

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
  if not h.has_spin:  h.turn_spinful()
  no = len(h.geometry.r) # number of orbitals (without spin)
  r = h.geometry.r  # z position
  JJ = [evaluate_J(zeeman,r[i],i) for i in range(no)] # exchange per site
  bzee = exchange_matrix(JJ) # create matrix
  h.intra = h.intra + h.spinful2full(bzee) # Add matrix 





def add_antiferromagnetism(h,m):
  """ Adds to the intracell matrix an antiferromagnetic imbalance """
  if not h.has_spin: h.turn_spinful()
  intra = h.intra # intracell hopping
  if h.geometry.has_sublattice: pass
  else: # if does not have sublattice
#      try:
          h.geometry.get_sublattice() # generate the sublattice
#      except: # try
#          return 0
  sublattice = h.geometry.sublattice  # if has sublattice
  if h.has_spin:
    natoms = len(h.geometry.x) # number of atoms
    # create the array
    if checkclass.is_iterable(m): # iterable, input is an array
      if len(m)!=len(h.geometry.r):
        raise ValueError("an antiferromagnetic array needs one value per site")
      mass = m # use the input array
    elif callable(m): # input is a function
      mass = [m(h.geometry.r[i]) for i in range(natoms)] # call the function
    else: # assume it is a float
      mass = [m for i in range(natoms)] # create list
    # the field of each atom, staggered with its sublattice
    mi = [np.array(float2array(mass[i]))*sublattice[i] for i in range(natoms)]
    out = exchange_matrix(mi) # turn into a matrix
    h.intra = h.intra + h.spinful2full(out) # Add matrix 
  else: require_spin(h,"antiferromagnetism")






def add_magnetism(h,m):
  """ Adds magnetism to the intracell hopping"""
  intra = h.intra # intracell hopping
  if h.has_spin:
    natoms = len(h.geometry.r) # number of atoms
    # create the array
    if checkclass.is_iterable(m):
      if checkclass.is_iterable(m[0]) and len(m)==natoms: # input is an array
        mass = m # use as arrays
      elif len(m)==3: # single exchange provided
        mass = [m for i in range(natoms)] # use as arrays
      else:
        raise ValueError("the exchange must be a single [mx,my,mz] vector or "
                "one such vector per site")
    elif callable(m): # input is a function
      mass = [m(h.geometry.r[i]) for i in range(natoms)] # call the function
    else: 
      raise TypeError("the exchange must be a vector, an array of vectors, or "
              "a callable of the position, and not a "+str(type(m)))
    mi = [float2array(mass[i]) for i in range(natoms)] # field per atom
    out = exchange_matrix(mi) # turn into a matrix
    h.intra = h.intra + h.spinful2full(out) # Add matrix 
  else: require_spin(h,"an exchange field")





def add_frustrated_antiferromagnetism(h,m):
  """Add frustrated magnetism"""
  if h.geometry.sublattice_number==3:
    g = geometry.kagome_lattice()
  elif h.geometry.sublattice_number==4:
    g = geometry.pyrochlore_lattice()
    g.center()
  else:
    raise NotImplementedError("frustrated antiferromagnetism is only "
            "implemented for lattices with three (kagome) or four "
            "(pyrochlore) sublattices")
  ms = []
  for i in range(len(h.geometry.r)): # loop
    ii = h.geometry.sublattice[i] # index of the sublattice
    if callable(m):
      ms.append(-g.r[int(ii)]*m(h.geometry.r[i])) # save this one
    else:
      ms.append(-g.r[int(ii)]*m) # save this one
  h.add_magnetism(ms) # add the magnetization





def compute_magnetization(h,**kwargs):
  """Return the magnetization of the system"""
  require_spin(h,"the magnetization")
  if h.has_eh:
    raise NotImplementedError("the magnetization is not implemented for "
            "Hamiltonians with the electron-hole (Nambu) degree of freedom")
  from .densitymatrix import full_dm
  dm = full_dm(h,**kwargs) # compute density matrix
  n = dm.shape[0]//2 # number of orbitals
  mz = np.array([dm[2*i,2*i] - dm[2*i+1,2*i+1] for i in range(n)])/2..real
  mx = np.array([dm[2*i,2*i+1].real for i in range(n)])
  my = np.array([dm[2*i,2*i+1].imag for i in range(n)])
  return (mx,my,mz)

