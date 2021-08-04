# routines to extract channels from a matrix
from __future__ import division
import numpy as np
from . import algebra
from .htk.matrixcomponent import spin_mixing_part
from . import superconductivity

def spin_channel(m,spin_column=None,spin_row=None,has_spin=True):
  """Extract a channel from a matrix"""
  if not has_spin: return m # return initial
  if (spin_row is None) or (spin_column is None): return m # return initial
  n = m.shape[0] # shape of the matrix
  n2 = n//2 # number of orbitals
  out = np.zeros((n,n),dtype=np.complex)
  if spin_column=="up": ii = 0
  else: ii = 1
  if spin_row=="up": jj = 0
  else: jj = 1
  for i in range(n2):
    for j in range(n2): out[i,j] = m[2*i+ii,2*j+jj]
  return np.matrix(out)



def swave(m):
  """Extract the swave pairing from a matrix, assuming
  the Nambu spinor basis"""
  n = m.shape[0]//4 # number of sites
  ds = np.zeros(n,dtype=np.complex) # pairing
  for i in range(n):
    ds[i] = m[4*i,4*i+2] # get the pairing
  return ds




def mz(m):
  """Extract the z component of the magnetism, assume spin degree of freedom"""
  n = m.shape[0]//2 # number of sites
  ds = np.zeros(n).real # pairing
  for i in range(n):
    ds[i] = -(m[2*i+1,2*i+1] - m[2*i,2*i]).real/2. # get the pairing
  return ds



def mx(m):
  """Extract the z component of the magnetism, assume spin degree of freedom"""
  n = m.shape[0]//2 # number of sites
  ds = np.zeros(n).real # pairing
  for i in range(n):
    ds[i] = m[2*i,2*i+1].real
  return ds



def my(m):
  """Extract the z component of the magnetism, assume spin degree of freedom"""
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




  
def extract_from_hamiltonian(self,name):
    """Extract a quantity from a Hamiltonian"""
    h0 = self.copy()
    if name=="density":
      if self.has_eh: 
          h0.remove_nambu()
          m = h0.intra
      else: m = self.intra
      return onsite(m,has_spin=self.has_spin)
    elif name=="mx" and self.has_spin:
      if self.has_eh: h0.remove_nambu() # not implemented
      return mx(h0.intra)
    elif name in ["swave","SC"]:
        if self.check_mode("spinful_nambu"): 
            return swave(self.intra)
        elif self.check_mode("spinless_nambu"): 
            from .sctk import spinless
            return spinless.extract_swave(self.intra)
        else: raise
    elif name=="CDW":
        if self.geometry.has_sublattice: # if it has sublattice
            v = self.extract("density")
            v = v - np.mean(v) # remove average
            return v*np.array(self.geometry.sublattice)
    elif name=="my" and self.has_spin:
      if self.has_eh: h0.remove_nambu() # not implemented
      return my(h0.intra)
    elif name=="mz" and self.has_spin:
      if self.has_eh: h0.remove_nambu() # not implemented
      return mz(h0.intra)
    elif name=="spin_mixing":
        return extract_spin_mixing(self)
    elif name=="hopping_spin_mixing":
        return extract_hopping_spin_mixing(self)
    elif name=="superfluidity": # extract the absolute value
        if self.has_eh:
            from .superconductivity import dict2absdeltas
            (uu,dd,ud) = dict2absdeltas(self.get_multihopping().get_dict())
            return uu+dd+ud
    else: raise



def extract_onsite_matrix_function(h,**kwargs):
    """Extract a certain function"""
    h = h.copy() # copy
    if h.check_mode("spinful"): # not implemented
      m = h.intra # get the matrix
      n = len(h.geometry.r) # number of sites
      if 2*n!=m.shape[0]: raise
      def f(r):
        ind = h.geometry.get_index(r,**kwargs) # get the index
        if ind is None: return np.zeros((2,2),dtype=np.complex)
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
    if not h.has_spin: raise
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
    if not h.has_spin: raise
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




