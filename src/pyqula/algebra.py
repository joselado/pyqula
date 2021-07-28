from scipy.sparse import issparse,bmat
from scipy.sparse import csc_matrix as csc
from scipy.sparse import csc_matrix
import scipy.linalg as dlg
import scipy.sparse.linalg as slg
import numpy as np
#from .algebratk import sparsetensor
from numba import jit
from . import parallel




import numbers
def isnumber(s):
    return isinstance(s, numbers.Number)



maxsize = 10000

def ismatrix(m):
    return type(m)==np.ndarray or issparse(m) or type(m)==np.matrix



def hermitian(m):
    return np.transpose(np.conjugate(m))

H = hermitian # alias
get_dagger = hermitian
dagger = hermitian

def inv(m):
    return dlg.inv(todense(m))


def trace(m):
    return np.trace(m)


def densebmat(m):
    """Turn a block matrix dense"""
    ms = [[todense(mi) for mi in mij] for mij in m]
    return todense(bmat(ms)) # return block matrix

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
    if issparse(m):
        if m.shape[0]<maxsize: return np.array(m.todense())
        else: raise
    else: return m


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

  evecs = evecs.transpose() # transpose eigenvectors
  for v in evecs: # loop over eigenvectors
    wf = wfs[0]*0.0j
    for (i,iv) in zip(range(len(v)),v): # loop over components
      wf += iv*wfs[i] # add contribution
    wfsout.append(wf.copy()) # store wavefunction
  return wfsout



def get_representation(wfs,A):
  """
  Gets the matrix representation of a certain operator
  """
  n = len(wfs) # number of eigenfunctions
  ma = np.zeros((n,n),dtype=np.complex) # representation of A
  sa = csc(A) # sparse matrix
  for i in range(n):
    vi = csc(np.conjugate(wfs[i])) # first wavefunction
    for j in range(n):
      vj = csc(wfs[j]).transpose() # second wavefunction
      data = (vi*sa*vj).todense()[0,0]
      ma[i,j] = data
  return ma





## routines for diagonalization ##

error = 1e-7

def todouble(vs,ind):
    """Double the eigenvectors"""
    nv = vs.shape[0]
    dim = vs.shape[1]
    vout = np.zeros((dim*2,nv),dtype=np.complex) # output vector
    return todouble_jit(vs,ind,vout,nv,dim)

@jit(nopython=True)
def todouble_jit(vs,ind,vout,nv,dim):
    """Double the eigenvectors, jit routine"""
    for i in range(dim):
        vout[2*i+ind,:] = vs[i,:]
    return vout



accelerate = False 

def eigh(m):
    """Wrapper for linalg"""
    m = todense(m)
    if np.max(np.abs(m.imag))<error: m = m.real # real matrix
    if not accelerate: return dlg.eigh(m)
#    from . import algebraf90
    # check if doing slices helps
    n = m.shape[0] # size of the matrix
    mo = m[0:n:2,1:n:2] # off diagonal is zero
#    if False: # assume block diagonal
    if np.max(np.abs(mo))<error: # assume block diagonal
        # detected block diagonal
        ms = [m[0:n:2,0:n:2],m[1:n:2,1:n:2]]
        out = parallel.pcall(lambda x: dlg.eigh(x),ms)
        (es0,vs0) = out[0]
        (es1,vs1) = out[1]
   #     (es0,vs0) = eigh(m[0:n:2,0:n:2]) # recall
   #     (es1,vs1) = eigh(m[1:n:2,1:n:2]) # recall
        es = np.concatenate([es0,es1]) # concatenate array
        #vs0 = algebraf90.todouble(vs0.T,0)
        #vs1 = algebraf90.todouble(vs1.T,1)
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
      v = v.todense() # convert to conventional matrix
    v = np.array(v) # convert to array
    if len(v.shape)==1: return v
    else: return v.reshape(v.shape[0]*v.shape[1])




def smalleig(m,numw=10,evecs=False,tol=1e-7):
  """
  Return the smallest eigenvalues using arpack
  """
  m = csc_matrix(m) # sparse matrix
  eig,eigvec = slg.eigsh(m,k=numw,which="LM",sigma=0.0,
                                  tol=tol)
  if evecs:  return eig,eigvec.transpose()  # return eigenvectors
  else:  return eig  # return eigenvalues



def spectral_gap(m,numw=10,**kwargs):
    """
    Compute the spectral gap
    """
    es = smalleig(m,numw=10,evecs=False,**kwargs)
    ev = es[es<0.]
    ec = es[es>0.]
    if len(ev)==0 or len(ec)==0:
        if numw<100: return gap(m,numw=2*numw,**kwargs)
        else: raise
    g = np.min(np.abs(ev))+np.min(ec) # gap
    return g # return gap



