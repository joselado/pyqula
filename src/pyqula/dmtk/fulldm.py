import numpy as np
from numba import jit,njit,prange


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


@jit(nopython=True,parallel=True)
def full_dm_batch_vectorized(es_batch,vs_batch,delta=1e-7):
    """Density-matrix contribution for a batch of kpoints, one kpoint per
    numba thread. vs_batch has shape (nb,n,n) with columns as
    eigenvectors (the htk.eigenvectors.parallel_diagonalization
    convention). Returns one (n,n) contribution per kpoint -- the caller
    pools (sums) them at the end, so each thread only ever writes its own
    output slot."""
    nb = es_batch.shape[0] # number of kpoints in this batch
    n = vs_batch.shape[1] # matrix dimension
    out = np.zeros((nb,n,n),dtype=np.complex128)
    for ik in prange(nb): # loop over kpoints in the batch, in parallel
        es = es_batch[ik]
        w = vs_batch[ik] # columns are eigenvectors
        occ = 1.0/(1.0+np.exp(es/delta))
        out[ik] = (np.conj(w)*occ) @ w.T
    return out







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


@jit(nopython=True,parallel=True)
def full_dm_batch_d_vectorized(es_batch,vs_batch,ks_batch,d,delta=1e-7):
    """Same as full_dm_batch_vectorized, but weighting each kpoint's
    contribution by the Bloch phase exp(2*pi*i*k.d) for a single hopping
    direction d. ks_batch has shape (nb,3), one kpoint per row."""
    nb = es_batch.shape[0]
    n = vs_batch.shape[1]
    out = np.zeros((nb,n,n),dtype=np.complex128)
    for ik in prange(nb): # loop over kpoints in the batch, in parallel
        es = es_batch[ik]
        w = vs_batch[ik]
        k = ks_batch[ik]
        kd = k[0]*d[0]+k[1]*d[1]+k[2]*d[2]
        phase = np.exp(1j*2.0*np.pi*kd)
        occ = 1.0/(1.0+np.exp(es/delta))
        weight = occ*phase
        out[ik] = (np.conj(w)*weight) @ w.T
    return out
