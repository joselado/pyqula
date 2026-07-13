import numpy as np
from numba import jit,njit


# vectorized mode seems to be faster, explicit kept as a reference
mode = "explicit"
mode = "vectorized"

def full_dm_python(es,vs,delta=1e-7):
  """Calculate the density matrix"""
  n = len(vs[0]) # size of the matrix from the first vector
  if mode=="explicit":
      return full_dm_explicit(n,np.array(es),np.array(vs),delta=delta)
  elif mode=="vectorized":
      return full_dm_vectorized(np.array(es),np.array(vs),delta=delta)
  else: raise


def full_dm_python_d(es,vs,ks,d,delta=1e-7):
  """Calculate the density matrix"""
  n = len(vs[0]) # size of the matrix from the first vector
  if mode=="explicit":
      return full_dm_python_d_jit(n,np.array(es),
                                  np.array(vs),
                                  np.array(ks),
                                  np.array(d),
                                  delta=delta)
  elif mode=="vectorized":
      return full_dm_d_vectorized(np.array(es),
                                  np.array(vs,dtype=np.complex128),
                                  np.array(ks),
                                  np.array(d),
                                  delta=delta)
  else: raise



@jit(nopython=True)
def full_dm_explicit(n,es,vs,delta=1e-7):
  """Auxiliary function to compute the density matrix"""
  dm = np.zeros((n,n),dtype=np.complex128)
  for ie in range(len(es)): # loop
      occ = 1./(1. + np.exp(es[ie]/delta)) # occupation, fermi-dirac
      for i in range(n):
        for j in range(n):
          dm[i,j] = dm[i,j] + occ*vs[ie][i].conjugate()*vs[ie][j] # add contribution
  return dm


# vectorized version, lets see if it is faster
@jit(nopython=True)
def full_dm_vectorized(es, vs, delta=1e-7):
    occ = 1.0 / (1.0 + np.exp(es / delta))          # shape (len_es,)
    dm = np.conj(vs).T @ (occ[:, None] * vs)        # (n, n) complex
    return dm







@jit(nopython=True)
def full_dm_python_d_jit(n,es,vs,ks,d,delta=1e-7):
  """Auxiliary function to compute the density matrix"""
  dm = np.zeros((n,n),dtype=np.complex128)
  for ie in range(len(es)): # loop
    k = ks[ie] # get kpoint
    kd = k[0]*d[0] + k[1]*d[1] + k[2]*d[2] # compute scalar product
    phi = np.exp(1j*np.pi*kd*2) # compute phase
    occ = 1./(1. + np.exp(es[ie]/delta)) # occupation, fermi-dirac
    phi = phi*occ # phase times occupation
    for i in range(n):
      for j in range(n):
        dm[i,j] = dm[i,j] + phi*vs[ie][i].conjugate()*vs[ie][j] # add contribution
  return dm


from scipy.special import expit
def fermi_dirac(es, delta):
    x = es / delta
    return expit(-x)  # stable, vectorized

# vectorized version with python
def full_dm_d_python_vectorized(es, vs, ks, d, delta=1e-7):
    # es: (m,), vs: (m, n), ks: (m, 3), d: (3,)
    kd = ks @ d             # (m,)
    theta = 2.0*np.pi*kd
    occ = fermi_dirac(es,delta) # occupations
    phi = occ * np.exp(1j*theta)        # (m,)
    # Sum_ie phi_ie * outer(conj(v_ie), v_ie)
    # Equivalent einsum:
    # dm[i,j] = sum_ie phi[ie] * conj(vs[ie,i]) * vs[ie,j]
    dm = np.einsum('e,ei,ej->ij', phi, np.conjugate(vs), vs, optimize=True)
    return dm



# vectorized version, this should be faster
@jit(nopython=True)
def full_dm_d_vectorized(es, vs, ks, d, delta=1e-7):
    # vs must be shape (n, M)
    M = es.shape[0]
    weight = np.empty(M, dtype=np.complex128)
    for i in range(M): # loop over eigenvectors
        kd = ks[i,0]*d[0] + ks[i,1]*d[1] + ks[i,2]*d[2] # kvector
        occ = 1.0 / (1.0 + np.exp(es[i] / delta)) # occupation
        weight[i] = np.exp(1j * 2.0 * np.pi * kd) * occ
    return (np.conj(vs.T) * weight) @ vs   # (n, n)
