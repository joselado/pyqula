from .. import parallel
from .. import algebra
from .. import gpu
from ..klist import kmesh
import numpy as np
import scipy.sparse.linalg as slg

def hk_matrix_batch(f,ks):
  """Evaluate a Hamiltonian generator f at every k in ks and stack the
  results into one dense complex128 array, densifying any sparse output
  along the way (see algebra.todense). Every call site that batches
  k-point Hamiltonians for parallel_diagonalization/peigvalsh should go
  through this, so a sparse Hamiltonian never reaches numba's dense eigh
  as a scipy sparse matrix (which numba cannot handle)."""
  return np.array([algebra.todense(f(k)) for k in ks],dtype=np.complex128)


def get_eigenvectors(h,nk=10,kpoints=False,k=None,sparse=False,
        numw=None,energy=0.0):
  from scipy.sparse import csc_matrix as csc
  if numw is not None: sparse = True
  if h.dimensionality==0:
    if not sparse: vv = algebra.eigh(h.intra)
    if sparse: vv = slg.eigsh(csc(h.intra),k=numw,
            which="LM",sigma=energy,tol=1e-5)
    vecs = np.array([v for v in vv[1].transpose()])
    if kpoints: return vv[0],vecs,np.array([[0.,0.,0.] for e in vv[0]])
    else: return vv[0],vecs
  elif h.dimensionality>0:
    f = h.get_hk_gen()
    if k is None:
      kp = kmesh(h.dimensionality,nk=nk) # generate a mesh
    else:  kp = np.array([k]) # kpoint given on input
#    vvs = [lg.eigh(f(k)) for k in kp] # diagonalize k hamiltonian
    nkp = len(kp) # total number of k-points
    if not sparse: # dense Hamiltonians
      # every k-point yields the same number of eigenstates, so the
      # batched diagonalization is unpacked with pure reshapes instead
      # of a per-eigenstate Python loop
      mats = hk_matrix_batch(f,kp) # H(k) for every k, densified
      es_batch,ws_batch = parallel_diagonalization(mats) # batched numba eigh
      n = es_batch.shape[1] # number of eigenstates per k-point
      eigvals = es_batch.reshape(-1) # eigenvalues, k-point by k-point
      # eigh returns the eigenvectors as columns, they are stored as rows
      eigvecs = ws_batch.transpose(0,2,1).reshape(nkp*n,n) # eigenvectors
      if kpoints: # return also the kpoints, one per eigenstate
        return eigvals,eigvecs,np.repeat(np.array(kp),n,axis=0)
      else:
        return eigvals,eigvecs
    # sparse Hamiltonians, eigsh may return a different number of
    # eigenstates per k-point, so they are unpacked one by one
    fk = lambda k: slg.eigsh(csc(f(k)),k=numw,which="LM",sigma=energy,tol=1e-5)
    vvs = parallel.pcall(fk,kp)
    nume = sum([len(v[0]) for v in vvs]) # number of eigenvalues calculated
    eigvecs = np.zeros((nume,h.intra.shape[0]),dtype=np.complex128) # eigenvectors
    eigvals = np.zeros(nume) # eigenvalues
    iv = 0
    kvectors = [] # empty list
    for ik in range(len(kp)): # loop over kpoints
      vv = vvs[ik] # get eigenvalues and eigenvectors
      for (e,v) in zip(vv[0],vv[1].transpose()):
        eigvecs[iv] = v.copy()
        eigvals[iv] = e.copy()
        kvectors.append(kp[ik])
        iv += 1
    if kpoints: # return also the kpoints
      return eigvals,eigvecs,np.array(kvectors)
    else:
      return eigvals,eigvecs
  else:
    raise ValueError("the Hamiltonian must have a non-negative dimensionality")







# function to diagonalize many in parallel
from numba import jit,prange
import numpy.linalg as nlg

@jit(nopython=True,parallel=True,cache=True)
def _parallel_diagonalization_numba(hks):
    """Diagonalize many matrices at once"""
    n = hks.shape[1] # size of the Hamiltonian
    nh = hks.shape[0] # number of Hamiltonians
    es = np.zeros((nh,n),dtype=np.float64) # storage for eigenenergies
    ws = np.zeros((nh,n,n),dtype=np.complex128) # storage for eigenenergies
    for i in prange(nh): # loop over Hamiltonians
        e,w = nlg.eigh(hks[i,:,:]) # diagonalize this Hamiltonian
        es[i,:] = e[:] # store
        ws[i,:,:] = w[:,:] # store
    return es,ws




@jit(nopython=True,parallel=True,cache=True)
def _peigvalsh_numba(hks):
    """Diagonalize many matrices at once in parallel"""
    n = hks.shape[1] # size of the Hamiltonian
    nh = hks.shape[0] # number of Hamiltonians
    es = np.zeros((nh,n),dtype=np.float64) # storage for eigenenergies
    for i in prange(nh): # loop over Hamiltonians
        e,w = nlg.eigh(hks[i,:,:]) # diagonalize this Hamiltonian
        es[i,:] = e[:] # store
    return es # return eigenergies


# Below this matrix size the GPU loses to the 8-thread numba kernels in
# double precision however large the batch, measured on a GTX 1060 (see
# documentation/gpu_porting_plan.md): the solves are too small to fill the
# card and the host-device transfers are not amortized. So the device is
# used only from here up, even under pyqula.gpu.set_gpu(True)
gpu_min_dimension = 32


def _on_gpu(hks):
    """Whether this stack goes to the device: the package-wide switch is on
    and the matrices are big enough for it to pay"""
    # np.shape, not hks.shape: a caller may hand in a list of matrices
    return gpu.get_gpu() and np.shape(hks)[1]>=gpu_min_dimension


def parallel_diagonalization(hks,eigh_prec="double"):
    """Eigenvalues and eigenvectors of many matrices at once.

    Runs the numba kernel on the CPU, or the jax one on the GPU under
    pyqula.gpu.set_gpu(True) (see htk/eigenvectorsjax.py). eigh_prec picks
    the precision of the device solve; "single" is several times faster
    again on a consumer card, at ~1e-6 relative eigenvalue error, which is
    fine for a density of states or a band structure and not for anything
    differencing eigenvectors. The CPU kernel is double precision only"""
    if _on_gpu(hks):
        from .eigenvectorsjax import peigh_gpu
        return peigh_gpu(hks,prec=eigh_prec)
    return _parallel_diagonalization_numba(hks)


peigh = parallel_diagonalization # alias


def peigvalsh(hks,eigh_prec="double"):
    """Eigenvalues of many matrices at once, the eigenvector-free
    counterpart of parallel_diagonalization -- see there for the backend
    and precision"""
    if _on_gpu(hks):
        from .eigenvectorsjax import peigvalsh_gpu
        return peigvalsh_gpu(hks,prec=eigh_prec)
    return _peigvalsh_numba(hks)
