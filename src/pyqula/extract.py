# routines to extract channels from a matrix
from __future__ import division
import numpy as np
from . import algebra
from .htk.matrixcomponent import spin_mixing_part
from . import superconductivity
from .check import require_spin, require_nambu, require_sublattice

def spin_channel(m,spin_column=None,spin_row=None,has_spin=True):
  """Extract a channel from a matrix"""
  if not has_spin: return m # return initial
  if (spin_row is None) or (spin_column is None): return m # return initial
  n = m.shape[0] # shape of the matrix
  n2 = n//2 # number of orbitals
  out = np.zeros((n,n),dtype=np.complex128)
  if spin_column=="up": ii = 0
  else: ii = 1
  if spin_row=="up": jj = 0
  else: jj = 1
  for i in range(n2):
    for j in range(n2): out[i,j] = m[2*i+ii,2*j+jj]
  return np.array(out)



def swave(m):
    """Extract the swave pairing from a matrix, assuming
    the Nambu spinor basis"""
    n = m.shape[0]//4 # number of sites
    ds = np.zeros(n,dtype=np.complex128) # pairing
    for i in range(n):
      ds[i] = m[4*i,4*i+2] # get the pairing
    return ds




def mz(m):
    """Extract the z component of the magnetism,
    assume spin degree of freedom"""
    n = m.shape[0]//2 # number of sites
    ds = np.zeros(n).real # pairing
    for i in range(n):
      ds[i] = -(m[2*i+1,2*i+1] - m[2*i,2*i]).real/2. # get the pairing
    return ds



def mx(m):
  """Extract the x component of the magnetism, assume spin degree of freedom"""
  n = m.shape[0]//2 # number of sites
  ds = np.zeros(n).real # pairing
  for i in range(n):
    ds[i] = m[2*i,2*i+1].real
  return ds



def my(m):
  """Extract the y component of the magnetism, assume spin degree of freedom"""
  n = m.shape[0]//2 # number of sites
  ds = np.zeros(n).real # pairing
  for i in range(n):
    ds[i] = -m[2*i,2*i+1].imag
  return ds



def onsite(m,has_spin=True):
    """Extract the onsite energy"""
    if has_spin: # has spin degree of freedom
        n = m.shape[0]//2 # number of sites
        ds = np.zeros(n).real # pairing
        for i in range(n):
          ds[i] = (m[2*i,2*i].real + m[2*i+1,2*i+1].real)/2.
        return ds
    else:
        n = m.shape[0] # number of sites
        ds = np.zeros(n).real # pairing
        for i in range(n):
          ds[i] = m[i,i].real
        return ds



def hopping_spinful(m,cutoff=0.001):
  """Extract hopping"""
  n = m.shape[0]//2 # number sites
  ii = []
  jj = []
  ts = []
  for i in range(n):
    for j in range(i,n):
      t = (np.abs(m[2*i,2*j]) + np.abs(m[2*i+1,2*j+1]))/2.
      t = (np.abs(m[2*i,2*j+1]) + np.abs(m[2*i+1,2*j]))/2.
      if t>cutoff:
        ii.append(i)
        jj.append(j)
        ts.append(t)
  return ii,jj,np.array(ts) # return pairs


def hopping_spinful_difference(m,cutoff=0.001,skip_same_site=False):
  """Extract hopping"""
  n = m.shape[0]//2 # number sites
  ii = []
  jj = []
  ts = []
  for i in range(n):
    for j in range(i,n):
      if i==j and skip_same_site: continue
      t = np.abs(m[2*i,2*j]) - np.abs(m[2*i+1,2*j+1])
      if abs(t)>cutoff:
        ii.append(i)
        jj.append(j)
        ts.append(t)
  return ii,jj,np.array(ts) # return pairs





def hopping_spinless(m,cutoff=0.001):
  """Extract hopping"""
  from scipy.sparse import coo_matrix
  m = coo_matrix(m) # transform to coo_matrix
  m.eliminate_zeros() # remove zeros
  row,col,data = m.row,m.col,m.data
  absd = np.abs(data) # absolute value
  row = row[absd>cutoff]
  col = col[absd>cutoff]
  data = data[absd>cutoff]
  return row,col,data




  
# The extractable quantities live in a registry (name -> builder) rather
# than in an if/elif chain, so the accepted names are derived from the
# dispatch instead of being re-listed by hand in the error message, and
# adding a quantity is one dict entry. Every builder takes the Hamiltonian
# and a dense copy of it, plus the caller's keyword bag.


def _extract_density(self,h0,**kwargs):
    if self.has_eh:
        h0.remove_nambu()
        m = h0.intra
    else: m = self.intra
    if not self.non_hermitian: # Hermitian case
        return onsite(m,has_spin=self.has_spin)
    else: # non Hermitian case
        from .nonhermitiantk.extract import onsite as onsite_NH
        return onsite_NH(m,has_spin=self.has_spin)


def _extract_magnetization(component):
    """Builder for one Cartesian component of the magnetization"""
    f = {"mx":mx,"my":my,"mz":mz}[component]
    def builder(self,h0,**kwargs):
        # a spinless Hamiltonian used to fall through to the else-branch
        # and be reported as an unknown quantity, which named neither the
        # real requirement nor the remedy
        require_spin(self,"the magnetization '"+component+"'")
        if self.has_eh: h0.remove_nambu() # not implemented
        return f(h0.intra)
    return builder


def _extract_swave(self,h0,**kwargs):
    if self.check_mode("spinful_nambu"):
        return swave(self.intra)
    elif self.check_mode("spinless_nambu"):
        from .sctk import spinless
        return spinless.extract_swave(self.intra)
    else: # has_eh but neither Nambu mode: the guard names the requirement
        require_nambu(self,"extracting the s-wave pairing")


def _extract_CDW(self,h0,**kwargs):
    # without a sublattice this used to return None silently
    require_sublattice(self,"the charge density wave order parameter")
    v = self.extract("density")
    v = v - np.mean(v) # remove average
    return v*np.array(self.geometry.sublattice)


def _extract_superfluidity(self,h0,**kwargs):
    # without the Nambu degree of freedom this used to return None silently
    require_nambu(self,"the superfluidity")
    from .superconductivity import dict2absdeltas
    (uu,dd,ud) = dict2absdeltas(self.get_multihopping().get_dict())
    return uu+dd+ud


def _extract_from_sctk(fname):
    """Builder deferring to one of the reciprocal-space routines of sctk"""
    def builder(self,h0,**kwargs):
        from .sctk import extract as scextract
        return getattr(scextract,fname)(self,**kwargs)
    return builder


# name -> builder(self,h0,**kwargs); several names are aliases of one builder
_extractors = {
  "density": _extract_density,
  "onsite": _extract_density,
  "mx": _extract_magnetization("mx"),
  "my": _extract_magnetization("my"),
  "mz": _extract_magnetization("mz"),
  "swave": _extract_swave,
  "SC": _extract_swave,
  "CDW": _extract_CDW,
  "spin_mixing": lambda self,h0,**kw: extract_spin_mixing(self),
  "hopping_spin_mixing": lambda self,h0,**kw: extract_hopping_spin_mixing(self),
  "superfluidity": _extract_superfluidity,
  "deltak": _extract_from_sctk("extract_pairing_kmap"),
  "absolute_delta": _extract_from_sctk("extract_absolute_pairing"),
  "absolute_spatial_delta": _extract_from_sctk("extract_absolute_spatial_pairing"),
  }


# every quantity understood by extract(), derived from the dispatch above
extractable_names = list(_extractors)


def get_extractable_names():
    """Return every quantity that h.extract() accepts"""
    return list(_extractors)


def extract_from_hamiltonian(self,name,**kwargs):
    """Extract a quantity from a Hamiltonian"""
    if name not in _extractors:
        raise ValueError("unknown quantity to extract '"+str(name)+"'; the "
          "accepted names are "+str(extractable_names))
    h0 = self.copy()
    if self.is_sparse: h0 = h0.get_dense() # turn into dense form
    return _extractors[name](self,h0,**kwargs)


def extract_onsite_matrix_function(h,**kwargs):
    """Extract a certain function"""
    h = h.copy() # copy
    if h.check_mode("spinful"): # not implemented
      m = h.intra # get the matrix
      n = len(h.geometry.r) # number of sites
      if 2*n!=m.shape[0]:
        raise ValueError("the intracell matrix does not have two spin "
                "components per site")
      def f(r):
        ind = h.geometry.get_index(r,**kwargs) # get the index
        if ind is None: return np.zeros((2,2),dtype=np.complex128)
        else: return m[2*ind:2*ind+2,2*ind:2*ind+2]
    return f # return function

def extract_magnetism_function(h,**kwargs):
    """Function that return the magnetization"""
    fm = extract_onsite_matrix_function(h,**kwargs) # create the function
    def f(r):
        m = fm(r) # get the matrix
        mx = m[0,1].real
        my = m[0,1].imag
        mz = (m[0,0] - m[1,1]).real/2.
        return np.array([mx,my,mz])
    return f # return function


def extract_onsite_function(h,**kwargs):
    """Function that return the onsite energy"""
    fm = extract_onsite_matrix_function(h,**kwargs) # create the function
    def f(r):
        m = fm(r) # get the matrix
        return (m[0,0] + m[1,1]).real/2.
    return f # return function


def extract_spin_mixing(h):
    """Extract the spin mixing part of a Hamiltonian"""
    h = h.copy()
    h.remove_nambu() # remove nambu
    require_spin(h,"the spin mixing")
    dt = h.get_dict() # get the multihopping object
    out = 0 # output
    for key in dt: # loop
        m = dt[key] # get the matrix
        m = spin_mixing_part(m) # spin mixing part
        m = np.abs(np.array(algebra.todense(m))) # absolute value
        m = m*m # square value
        m = np.mean(m,axis=0) # sum over the first axis
        out = out + m # add to the output
    return out # return the mixing


def extract_hopping_spin_mixing(h):
    """Extract the spin mixing part of a Hamiltonian"""
    h = h.copy()
    h.remove_nambu() # remove nambu
    require_spin(h,"the hopping spin mixing")
    dt = h.get_dict() # get the multihopping object
    out = 0 # output
    for key in dt: # loop
        m = dt[key] # get the matrix
        m = spin_mixing_part(m) # spin mixing part
        if key==(0,0,0): # discard onsite terms
            for i in range(m.shape[0]): m[i,i] = 0.0 # set to zero
        m = np.abs(np.array(algebra.todense(m))) # absolute value
        m = m*m # square value
        m = np.mean(m,axis=0) # sum over the first axis
        out = out + m # add to the output
    return out # return the mixing







