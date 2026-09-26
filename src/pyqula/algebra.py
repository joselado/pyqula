from scipy.sparse import issparse,bmat,block_array
from scipy.sparse import csc_matrix as csc
from scipy.sparse import csc_matrix
from scipy.sparse import identity as sparse_identity
import scipy.linalg as dlg
import scipy.sparse.linalg as slg
import numpy as np
#from .algebratk import sparsetensor
from numba import jit
from . import parallel


arpack_tol = 1e-5
arpack_maxiter = 10000



import numbers
def isnumber(s):
    return isinstance(s, numbers.Number)



maxsize = 10000

def ismatrix(m):
    return (type(m)==np.ndarray and len(m.shape)==2) or issparse(m) or type(m)==np.matrix


def isvector(m):
    return type(m)==np.ndarray and len(m.shape)==1


def hermitian(m):
    return m.conjugate().transpose() 
#    return np.transpose(np.conjugate(m))

H = hermitian # alias
get_dagger = hermitian
dagger = hermitian

def inv(m):
    # numpy.linalg.inv avoids scipy.linalg.inv's input-validation overhead
    # (_asarray_validated/asarray_chkfinite); both are LAPACK LU-based exact
    # inverses, agreeing to float noise -- this matters here because many
    # call sites (e.g. topologytk/green.py's Green's-function Berry
    # curvature) invert large numbers of small matrices, where Python/call
    # overhead dominates over the O(N^3) FLOPs.
    return np.linalg.inv(todense(m))


def trace(m):
    return np.trace(m)


def densebmat(m):
    """Turn a block matrix dense"""
    ms = [[todense(mi) for mi in mij] for mij in m]
    return todense(block_array(ms)) # return block matrix

def direct_sum(ms):
    mout = [[None for i in range(len(ms))] for j in range(len(ms))]
    for i in range(len(ms)): mout[i][i] = ms[i]
    return densebmat(mout)


def dot(a,b):
    """Compute the scalar product"""
    return np.dot(np.conjugate(a),b)

def braket_wAw(w,A,wi=None):
  """
  Compute the braket of a wavefunction
  """
  if wi is None: wi = w
  if issparse(A): # sparse matrices
    return (np.conjugate(wi)@A@w) # modern way
  else: # matrices and arrays
    return (np.conjugate(wi)@np.array(A)@w) # modern way


def todense(m):
    """Turn a matrix dense"""
    if m is None: return None
    if issparse(m):
        if m.shape[0]<maxsize: return m.toarray()
        else:
            raise MemoryError("refusing to densify a sparse matrix larger than "
                    "algebra.maxsize; keep it sparse, or raise algebra.maxsize "
                    "if the dense matrix really does fit in memory")
    else: return np.array(m,dtype=np.complex128)


def braket_ww(w,wi):
  """
  Compute the braket of two wavefunctions
  """
  w = matrix2vector(w) # convert to vector
  wi = matrix2vector(wi) # convert to vector
  return (np.conjugate(w)@wi) # modern way




def disentangle_manifold(wfs,A):
    """
    Disentangles the wavefunctions of a degenerate manifold
    by expressing them in terms of eigenvalues of an input operator
    """
    ma = get_representation(wfs,A) # get the matrix form of the operator
    wfsout = [] # empty list
    evals,evecs = dlg.eigh(ma) # diagonalize
    evecs = np.conjugate(evecs.T) # transpose eigenvectors
#    print("Representation")
#    print(np.round(ma,2))
#    print("Eigenvectors")
#    print(np.round(evecs,2))
    for v in evecs: # loop over eigenvectors
      wf = wfs[0]*0.0j
      for (i,iv) in zip(range(len(v)),v): # loop over components
        wf = wf + iv*wfs[i] # add contribution
      wfsout.append(wf.copy()) # store wavefunction
    return wfsout



def get_representation(wfs,A):
    """
    Gets the matrix representation of a certain operator
    """
    n = len(wfs) # number of eigenfunctions
    ma = np.zeros((n,n),dtype=np.complex128) # representation of A
    A = np.array(A)
    for i in range(n):
      vi = wfs[i] # first wavefunction
      for j in range(n):
        vj = np.conjugate(wfs[j]) # first wavefunction
        data = vi@A@vj
        ma[i,j] = data
    return ma





## routines for diagonalization ##

error = 1e-7

def todouble(vs,ind):
    """Double the eigenvectors"""
    nv = vs.shape[0]
    dim = vs.shape[1]
    vout = np.zeros((dim*2,nv),dtype=np.complex128) # output vector
    return todouble_jit(vs,ind,vout,nv,dim)

@jit(nopython=True)
def todouble_jit(vs,ind,vout,nv,dim):
    """Double the eigenvectors, jit routine"""
    for i in range(dim):
        vout[2*i+ind,:] = vs[i,:]
    return vout

eig = dlg.eig # non Hermitian diagonalization
eigvals = dlg.eigvals # non Hermitian diagonalization

accelerate = False 

def eigh(m):
    """Wrapper for linalg"""
    m = todense(m)
    if np.max(np.abs(m.imag))<error: m = m.real # real matrix
    if not accelerate: return dlg.eigh(m)
    # check if doing slices helps
    n = m.shape[0] # size of the matrix
    mo = m[0:n:2,1:n:2] # off diagonal is zero
#    if False: # assume block diagonal
    if np.max(np.abs(mo))<error: # assume block diagonal
        # detected block diagonal
        # fixed 2-item split: a process pool (pcall) would only add
        # spawn/IPC overhead here, so just call eigh directly twice
        (es0,vs0) = dlg.eigh(m[0:n:2,0:n:2])
        (es1,vs1) = dlg.eigh(m[1:n:2,1:n:2])
   #     (es0,vs0) = eigh(m[0:n:2,0:n:2]) # recall
   #     (es1,vs1) = eigh(m[1:n:2,1:n:2]) # recall
        es = np.concatenate([es0,es1]) # concatenate array
        vs0 = todouble(vs0,0) # double the degrees of freedom
        vs1 = todouble(vs1,1) # double the degrees of freedom
        vs = np.concatenate([vs0.T,vs1.T]).T
        return (es,vs) # return the eigenvaleus and eigenvectors

    else:
      if np.max(np.abs(m.imag))<error: # assume real
          return dlg.eigh(m.real) # diagonalize real matrix
      else: return dlg.eigh(m) # diagonalize complex matrix


def eigvalsh(m):
    """Wrapper for linalg"""
    m = todense(m) # turn the matrix dense
    if np.max(np.abs(m.imag))<error: m = m.real # real matrix
    if not accelerate: return dlg.eigvalsh(m)
    # check if doing slices helps
    n = m.shape[0] # size of the matrix
    mo = m[0:n:2,1:n:2] # off diagonal is zero
#    if False: # assume block diagonal
    if np.max(np.abs(mo))<error: # assume block diagonal
        # detected block diagonal
        es0 = dlg.eigvalsh(m[0:n:2,0:n:2]) # recall
        es1 = dlg.eigvalsh(m[1:n:2,1:n:2]) # recall
        es = np.concatenate([es0,es1]) # concatenate array
        return es

    else:
      if np.max(np.abs(m.imag))<error: # assume real
          return dlg.eigvalsh(m.real) # diagonalize real matrix
      else: return dlg.eigvalsh(m) # diagonalize complex matrix



def matrix2vector(v):
    """Transform a matrix into a vector"""
    if issparse(v): # sparse matrix
      v = v.toarray() # convert to conventional array
    v = np.array(v) # convert to array
    if len(v.shape)==1: return v
    else: return v.reshape(v.shape[0]*v.shape[1])




def arpack_eigh(m,k=6,return_eigenvectors=True,v0=None,**kwargs):
    """scipy's eigsh, returning orthonormal eigenvectors.

    ARPACK has no complex Hermitian driver, so scipy's eigsh hands a
    complex matrix -- every H(k) away from the time-reversal points, and
    every H(k) of a supercell -- to the non-Hermitian eigs. Its
    eigenvectors are exact, but inside a degenerate level they are just
    some basis of the eigenspace, not an orthonormal one: at the zone
    center of a 4x4 triangular supercell, with an 8-fold and a 5-fold
    level, their overlap matrix was 0.65-0.83 away from the identity.
    Anything summed over states -- an operator weight, a local density
    sum_n |v_n|^2, an occupied manifold -- then counted the same direction
    several times. A Rayleigh-Ritz step on the returned subspace makes
    them orthonormal again without changing the subspace, so the
    eigenvalues are unchanged. The eigenpairs come out in ascending order.

    The starting vector is a seeded random one unless one is given, see
    smalleig for why neither an unseeded nor a structured one will do.
    Every other keyword goes to scipy's eigsh."""
    if v0 is None: v0 = np.random.RandomState(0).randn(m.shape[0])
    if not return_eigenvectors:
        return slg.eigsh(m,k=k,v0=v0,return_eigenvectors=False,**kwargs)
    es,vs = slg.eigsh(m,k=k,v0=v0,**kwargs)
    return rayleigh_ritz(m,vs)


def rayleigh_ritz(m,vs):
    """Orthonormal eigenpairs of the Hermitian m within the subspace
    spanned by the columns of vs, in ascending order of energy. The
    subspace must be an invariant one of m, as a set of its eigenvectors
    is, and this is checked: a set that did not span one would come out
    with eigenpairs that are not eigenpairs of m"""
    q = np.linalg.qr(np.asarray(vs))[0] # orthonormal basis of the subspace
    mq = np.asarray(m@q) # m applied to it
    hq = np.conjugate(q.T)@mq # m in that basis
    es,ws = dlg.eigh((hq+np.conjugate(hq.T))/2.) # Hermitian by construction
    vs = q@ws # the eigenvectors, orthonormal
    residual = np.max(np.abs(mq@ws - vs*es[None,:])) if len(es)>0 else 0.
    # ARPACK's own tolerance bounds the residual by tol*|m| only, so the
    # check is relative to |m| (its largest absolute row sum), and loose:
    # it is there to catch a direction that is no eigenvector at all
    scale = max(1.,float(np.max(np.abs(csc_matrix(m)).sum(axis=1))))
    if residual>1e-3*scale:
        raise ValueError("ARPACK returned vectors that do not span an "
                "invariant subspace (largest residual "+str(residual)+
                "); tighten its tolerance or ask for fewer states")
    return es,vs



def smalleig(m,numw=10,evecs=False,e0=0.,tol=arpack_tol):
    """
    Return the smallest eigenvalues using arpack
    """
    m = csc_matrix(m) # sparse matrix
    # fixed-seed (but still random) starting vector: eigsh defaults to
    # an unseeded random v0, which reproduces correctly-summed densities
    # only when num_waves happens to span whole degenerate eigenspaces --
    # whenever it cuts one in half (common on symmetric lattices), an
    # unseeded v0 makes the specific eigenvectors returned, and hence
    # e.g. LDOS built from them, vary run to run for identical input.
    # A *structured* fixed vector (e.g. all-ones) is the wrong fix: it is
    # invariant under any exchange/permutation symmetry of m (e.g. two
    # translationally-equivalent replicas in a supercell), so Lanczos
    # never leaves that symmetric sector and silently misses whole
    # antisymmetric-sector eigenspaces -- deterministic but wrong. A
    # seeded random vector keeps runs reproducible without that blind
    # spot (probability zero of exact alignment with any symmetry
    # subspace). That seeding fixes *which* subspace comes back when a
    # level is cut; arpack_eigh, which puts the same seeded vector in
    # place, also makes the vectors orthonormal *within* it, which the
    # eigs driver that eigsh uses for a complex matrix does not.
    try:
        if not evecs: # eigenvalues only, nothing to orthonormalize
            return arpack_eigh(m,k=numw,which="LM",sigma=e0,tol=tol,
                                        return_eigenvectors=False)
        eig,eigvec = arpack_eigh(m,k=numw,which="LM",sigma=e0,tol=tol)
        return eig,eigvec.transpose()  # return eigenvectors
    except:
        print("Switch to dense")
        if m.shape[0]>maxsize: raise
        else:
            if not evecs: return eigvalsh(todense(m))
            else:
                eig,eigvec = eigh(todense(m))
                return eig,eigvec.transpose() # match the ARPACK branch's convention



def spectral_gap(m,numw=10,**kwargs):
    """
    Compute the spectral gap
    """
    es = smalleig(m,numw=numw,evecs=False,**kwargs)
    ev = es[es<0.]
    ec = es[es>0.]
    if len(ev)==0 or len(ec)==0:
        # the window held states of a single sign, so widen it; this used
        # to call gap(), a name that does not exist, and to ask smalleig
        # for a hardcoded 10 eigenvalues, so neither half could work
        if numw<100: return spectral_gap(m,numw=2*numw,**kwargs)
        else:
            raise ValueError("no spectral gap was found around zero after "
                    "enlarging the number of computed eigenvalues to 100")
    g = np.min(np.abs(ev))+np.min(ec) # gap
    return g # return gap







def sqrtm(M):
    """Square root for Hermitian matrix"""
    (m2,evecs) = sqrtm_rotated(M)
    m2 = dagger(evecs) @ m2 @ evecs  # change of basis
    return m2 # return matrix



def sqrtm_rotated(M,positive=True):
    """Square root for Hermitian matrix in the diagonal basis,
    and rotation matrix"""
    M = (M + dagger(M))/2. # make Hermitian
    (evals,evecs) = dlg.eigh(M) # eigenvals and eigenvecs
    if positive:
        if np.min(evals)<0.:
#            print("Matrix is not positive defined",print(evals[evals<0.]))
            evals[evals<0.] = 0.
    evecs = dagger(np.array(evecs)) # change of basis
    m2 = np.array([[0.0j for i in evals] for j in evals]) # create matrix
    for i in range(len(evals)):
        m2[i,i] = np.sqrt(np.abs(evals[i])) # square root
    return (m2,evecs) # return matrix




def spectrum_bottom(m,tol=arpack_tol):
    """
    Return the most negative energy state
    """
    if m.shape[0]>1000: # use arpack 
        m = csc_matrix(m) # sparse matrix
        eig,eigvec = slg.eigsh(m,k=3,which="SA",tol=tol)
        return np.min(eig)
    else:
        return np.min(eigvalsh(todense(m)))


def angle(v1,v2):
    """Given two vectors, return the angle between them"""
    v1 = v1/np.sqrt(v1.dot(v1)) # normalize
    v2 = v2/np.sqrt(v2.dot(v2)) # normalize
    c = v1.dot(v2) # cosine
    v3 = np.cross(v1,v2) # cross product
    raise NotImplementedError("algebra.angle is not implemented; use "
            "sculpt.get_angle instead")






def applyinverse(A,b):
    """Apply A^-1 to b"""
    if A.shape[0]<30: return inv(A)@b
    else: return slg.spsolve(A,b)



def is_zero(m):
    """Check if a matrix is zero"""
    m = todense(m)
    return np.max(np.abs(m))<1e-6


def bmat(M):
    from scipy.sparse import bmat
    if len(M)==1: 
        return todense(M[0][0])
    else: return todense(bmat(M))

def expm(M):
    M = todense(M)
    return dlg.expm(M)


def det(M):
    """Determinant of a matrix"""
    if issparse(M): 
        raise NotImplementedError("the determinant of a sparse matrix is not "
                "implemented; densify it first")
    else: return dlg.det(M)


def identity(M):
    """Return identity matrix"""
    if issparse(M): 
        return sparse_identity(M.shape[0],dtype=np.complex128)
    else:
        return np.identity(M.shape[0],dtype=np.complex128)

