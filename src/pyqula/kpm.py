# kernel polynomial method libraries
from __future__ import print_function,division
import scipy.sparse.linalg as lg
from scipy.sparse import csc_matrix as csc
import numpy.random as rand
from scipy.sparse import coo_matrix,csc_matrix,bmat
import numpy as np
from scipy.signal import hilbert
from . import algebra
from numba import jit


# numba version
from .kpmtk.kpmnumba import kpm_moments_v as get_moments_v
from .kpmtk.kpmnumba import kpm_moments_batch as get_moments_batch





def python_kpm_moments(v,m,n=100):
    """Python routine to calculate moments"""
    mus = np.array([0.0j for i in range(2*n)]) # empty arrray for the moments
    am = v.copy() # zero vector
    a = m@v  # vector number 1
    bk = algebra.braket_ww(v,v)
  #  bk = (np.transpose(np.conjugate(v))*v)[0,0] # scalar product
    bk1 = algebra.braket_ww(a,v)
  #  bk1 = (np.transpose(np.conjugate(a))*v)[0,0] # scalar product
    
    mus[0] = bk.copy()  # mu0
    mus[1] = bk1.copy() # mu1
    for i in range(1,n): 
        ap = 2*m@a - am # recursion relation
        bk = algebra.braket_ww(a,a)
        bk1 = algebra.braket_ww(ap,a)
        mus[2*i] = 2.*bk
        mus[2*i+1] = 2.*bk1
        am = a.copy() # new variables
        a = ap.copy() # new variables
    mu0 = mus[0] # first
    mu1 = mus[1] # second
    for i in range(1,n): 
      mus[2*i] +=  - mu0
      mus[2*i+1] += -mu1 
    return mus


def python_kpm_moments_clear(v,m,n=100):
  """Python routine to calculate moments"""
  mus = np.array([0.0j for i in range(2*n)]) # empty arrray for the moments
  a0 = v.copy() # first vector
  am = v.copy() # first vector
  a = m*v  # vector number 1
  mus[0] = 1.  # mu0
  mu = (np.transpose(np.conjugate(a0))*a)[0,0] # scalar product
  mus[1] = mu # mu1
  for i in range(1,2*n): 
      ap = 2*m*a - am # recursion relation
      mu = (np.transpose(np.conjugate(a0))*a)[0,0] # scalar product
      mus[i] = mu # store
      am = a.copy() # new variables
      a = ap.copy() # new variables
  return mus






from .kpmtk.kpmnumba import kpm_moments_A_batch as get_moments_A_batch


def get_momentsA(v,m,n=100,A=None,**kwargs):
    """ Get the first n moments of a certain vector, weighted by operator
    A, using the Chebychev recursion relations (see get_moments_A_batch
    for the batched numba implementation)"""
    if A is None: # only for a certain A
      raise ValueError("get_momentsA needs the weighting operator, pass it as "
              "A")
    v = algebra.matrix2vector(v) # zero vector
    mus = get_moments_A_batch(np.array([v]),m,A,n=n,**kwargs)
    return mus[0]


from .kpmtk.kpmnumba import kpm_moments_ij as get_moments_ij

from .kpmtk.kpmnumba import kpm_moments_vivj as get_moments_vivj
from .kpmtk.scaleguard import check_scale


def operator_norm_bound(A):
    """Upper bound on the spectral norm of the operator A, the bound on
    the operator-weighted moments <T_n(m)v|A|v> of a unit vector v:
    sqrt(|A|_1 |A|_inf), the largest column and row sums of |A|. One for
    no operator"""
    if A is None: return 1.
    A = abs(csc(A))
    return np.sqrt(np.max(A.sum(axis=0))*np.max(A.sum(axis=1)))


def full_trace(m_in,n=200,**kwargs):
  """ Get full trace of the matrix, one site-basis vector per numba
  thread (see kpm_moments_batch)"""
  m = csc(m_in) # saprse matrix
  nd = m.shape[0] # length of the matrix
  from .kpmtk.ldos import index2vector
  vs = np.array([index2vector(i,nd) for i in range(nd)])
  mus = get_moments_batch(vs,m,n=n,**kwargs) # (nd,2n) moments, one row per site
  return np.sum(mus,axis=0)/nd









from .kpmtk.ldos import moments_local_dos


from .kpmtk.ldos import get_ldos as ldos


ldos0d = ldos



def tdos(m_in,scale=10.,npol=None,ne=500,kernel="jackson",
              ntries=20,ewindow=None,frand=None,
              operator=None,x=None,**kwargs):
  """Return two arrays with energies and local DOS. Extra **kwargs (e.g.
  kpm_prec) are forwarded to random_trace/get_moments_batch; the CPU/GPU
  backend is the package-wide switch of pyqula.gpu."""
  if npol is None: npol = ne
  mus = random_trace(m_in/scale,ntries=ntries,n=npol,fun=frand,
          operator=operator,**kwargs)
  check_scale(mus,scale,bound=operator_norm_bound(operator),
          kpm_prec=kwargs.get("kpm_prec","double"))
  return moments_to_dos(mus,scale=scale,ne=ne,ewindow=ewindow,
          kernel=kernel,x=x)


def moments_to_dos(mus,scale=10.,ne=500,ewindow=None,kernel="jackson",
        x=None):
  """Turn the Chebyshev moments of m/scale into a density on an energy
  grid, ne points spanning the spectrum or the window ewindow,
  interpolated onto the energies x when they are given"""
  if ewindow is None or abs(ewindow)>scale: # no window provided
    xs = np.linspace(-1.0,1.0,ne,endpoint=True)*1.01 # energies
  else:
    xx = abs(ewindow/scale) # scale
    xs = np.linspace(-xx,xx,ne,endpoint=True)*1.01 # energies
  ys = generate_profile(mus,xs,kernel=kernel).real
  (xs,ys) = (scale*xs,ys/scale)
  if x is not None:
    from scipy.interpolate import interp1d
    f = interp1d(xs,ys,bounds_error=False,fill_value=0.)
#    f = interp1d(xs,ys,bounds_error=False,fill_value=[ys[0],ys[-1]])
    return x,f(x)
  else: return xs,ys


def factored_moments(m,U,n=200,**kwargs):
  """Chebyshev moments of Tr[U U^dagger T_j(m)] = sum_a <u_a|T_j(m)|u_a>,
  j=0,...,2n-1, for the r columns u_a of the N x r matrix U, exactly.

  This is the trace of an operator of rank r that is given as U U^dagger,
  which takes r Chebyshev recursions, one per column, where the trace of a
  general operator takes a stochastic average over random vectors. It is
  how KITE computes the momentum-resolved spectral function, from Bloch
  states as starting vectors (Joao et al., R. Soc. Open Sci. 7, 191809
  (2020), arXiv:1910.05194, Sec. 4.4.1, Eqs. 41-43). The columns run as
  one batch, one per numba thread or on the device, see
  get_moments_batch. Returns an (r,2n) array with the moments of each
  column normalized to one, and the squared norms of the columns, which
  weigh them in the trace; the columns that are zero are left out of
  both"""
  from scipy.sparse import issparse
  vs = U.T.toarray() if issparse(U) else np.array(U).T # one row per column
  vs = np.array(vs,dtype=np.complex128)
  w = np.sum(np.abs(vs)**2,axis=1) # squared norms
  vs = vs[w>0.] ; w = w[w>0.] # an empty column adds nothing
  if len(w)==0: return np.zeros((0,2*n),dtype=np.complex128),w
  vs = vs/np.sqrt(w)[:,None] # unit vectors
  mus = get_moments_batch(vs,csc(m),n=n,**kwargs) # (r,2n)
  return mus,w


def factored_dos(m_in,U,scale=10.,npol=None,ne=500,kernel="jackson",
        ewindow=None,x=None,**kwargs):
  """Density of Tr[U U^dagger delta(E-m)], with no stochastic trace: the
  weight of the rank-r operator U U^dagger in the density of states of m,
  from one Chebyshev expansion per column of U (see factored_moments)"""
  if npol is None: npol = ne
  mus,w = factored_moments(m_in/scale,U,n=npol,**kwargs)
  if len(w)>0: # each row is the moments of a unit vector, bounded by one
    check_scale(mus,scale,bound=1.,kpm_prec=kwargs.get("kpm_prec","double"))
  mus = np.sum(w[:,None]*mus,axis=0) # the trace, weighted by the norms
  return moments_to_dos(mus,scale=scale,ne=ne,ewindow=ewindow,
          kernel=kernel,x=x)


def pdos(m,P=None,**kwargs):
    """Compute the projected density of states, assuming the operator
    fufills P^2 = P"""
    frand = kwargs.pop("frand",None) # caller-provided random vector generator
    from .randomtk import randomwf
    fun0 = frand if frand is not None else randomwf(m.shape[0]) # generator
    if P is not None: # operator provided
        from .operators import Operator
        op = Operator(P).get_matrix() # redefine
        from scipy.sparse import csc_matrix
        op = csc_matrix(op)
        def fun():
            r = fun0()
            r = op@r
            r = r/np.sqrt(np.abs(np.sum(np.conjugate(r)*r)))
            return r
#        print("aaa",fun0().shape,(op@fun0()).shape)
#        fun = lambda : op@fun0() # define new generator
    else: fun = fun0 # original generator
    return tdos(m,frand=fun,**kwargs) # call TDOS with the generator



tdos0d = tdos # redefine


def total_energy(m_in,scale=10.,npol=None,ne=500,ntries=20):
   x,y = tdos0d(m_in,scale=scale,npol=npol,ne=ne,ntries=ntries)
   z = .5*(np.sign(x)+1.)*x*y # function to integrate
   return np.trapezoid(z,x)



def random_trace(m_in,ntries=20,n=200,fun=None,operator=None,**kwargs):
  """ Calculates local DOS using the KPM. Extra **kwargs (e.g. kpm_prec)
  are forwarded to get_moments_batch/get_moments_A_batch; the CPU/GPU
  backend is the package-wide switch of pyqula.gpu."""
  m = csc(m_in) # sparse matrix
  nd = m.shape[0] # length of the matrix
  if fun is not None: # check that dimensions are fine
    v0 = fun()
    if len(v0) != m_in.shape[0]:
      raise ValueError("the vector returned by fun does not have the same "
              "size as the matrix")
  if fun is None:
#    def fun(): return rand.random(nd) -.5 + 1j*rand.random(nd) -.5j
      from .randomtk import randomwf
      fun = randomwf(nd) # generator
  if operator is None: # common case: batch the tries, one vector per numba thread
    vs = np.array([fun() for i in range(ntries)])
    vs = vs/np.sqrt(np.sum(np.conjugate(vs)*vs,axis=1))[:,None] # normalize each row
    mus = get_moments_batch(vs,m,n=n,**kwargs) # (ntries,2n) moments
    return np.mean(mus,axis=0)
  else: # operator-weighted moments: batch the tries, one vector per numba thread
    vs = np.array([fun() for i in range(ntries)])
    vs = vs/np.sqrt(np.sum(np.conjugate(vs)*vs,axis=1))[:,None] # normalize each row
    mus = get_moments_A_batch(vs,m,operator,n=2*n,**kwargs) # (ntries,2n) moments
    return np.mean(mus,axis=0)



def random_trace_A(m_in,ntries=20,n=200,A=None,**kwargs):
  """ Calculates local DOS using the KPM, batching the tries over numba
  threads (see get_moments_A_batch)"""
  m = csc(m_in) # saprse matrix
  nd = m.shape[0] # length of the matrix
  vs = rand.random((ntries,nd)) -.5 + 1j*rand.random((ntries,nd)) -.5j
  vs = vs/np.sqrt(np.sum(np.conjugate(vs)*vs,axis=1))[:,None] # normalize each row
  mus = get_moments_A_batch(vs,m,A,n=n,**kwargs) # (ntries,n) moments
  return np.mean(mus,axis=0)



def full_trace_A(m_in,n=200,A=None,**kwargs):
  """ Calculates full trace using the KPM, one site-basis vector per
  numba thread (see get_moments_A_batch)"""
  m = csc(m_in) # saprse matrix
  nd = m.shape[0] # length of the matrix
  from .kpmtk.ldos import index2vector
  vs = np.array([index2vector(i,nd) for i in range(nd)])
  mus = get_moments_A_batch(vs,m,A,n=n,**kwargs) # (nd,n) moments, one row per site
  return np.sum(mus,axis=0)/nd



def correlator0d(m_in,i=0,j=0,scale=10.,npol=None,ne=500,write=True,
    x=None):
    """Return two arrays with energies and local DOS"""
    if npol is None: npol = ne
    mus = get_moments_ij(m_in/scale,n=npol,i=i,j=j)
    check_scale(mus,scale)
    if x is None: xs = np.linspace(-1.0,1.0,ne,endpoint=True)*0.99 # energies
    else: xs = x/scale # use from input
    ys = generate_green_profile(mus,xs,kernel="jackson")/scale*np.pi # so it is the Green function
  #  imys = hilbert(ys).imag
    if write: 
        np.savetxt("CORRELATOR_KPM.OUT",np.array([scale*xs,-ys.imag,ys.real]).T)
    return (scale*xs,ys.real,ys.imag)




def dm_ij_energy(m_in,i=0,j=0,scale=10.,npol=None,ne=500,x=None):
  """Return the correlation function"""
  if npol is None: npol = ne
  mus = get_moments_ij(m_in/scale,n=npol,i=i,j=j)
  check_scale(mus,scale)
  if x is None: xs = np.linspace(-1.0,1.0,ne,endpoint=True)*0.99 # energies
  else: xs = x/scale # use from input
  ysr = generate_profile(mus.real,xs,kernel="jackson")/scale*np.pi # so it is the Green function
  ysi = generate_profile(mus.imag,xs,kernel="jackson")/scale*np.pi # so it is the Green function
  ys = ysr - 1j*ysi
  return (scale*xs,ys)



def dm_vivj_energy(m_in,vi,vj,scale=10.,npol=None,ne=500,x=None):
  """Return the correlation function"""
  if npol is None: npol = ne
  mus = get_moments_vivj(m_in/scale,vi,vj,n=npol)
  check_scale(mus,scale,
          bound=np.linalg.norm(np.ravel(vi))*np.linalg.norm(np.ravel(vj)))
  if np.sum(np.abs(mus.imag))>0.001:
#    print("WARNING, off diagonal has nonzero imaginary elements",np.sum(np.abs(mus.imag)))
    pass
  if x is None: xs = np.linspace(-1.0,1.0,ne,endpoint=True)*0.99 # energies
  else: xs = x/scale # use from input
  ysr = generate_profile(mus.real,xs,kernel="lorentz")/scale*np.pi # so it is the Green function
  ysi = generate_profile(mus.imag,xs,kernel="jackson")/scale*np.pi # so it is the Green function
  ys = ysr - 1j*ysi
  return (scale*xs,ys)



from .kpmtk.momenttoprofile import generate_green_profile
from .kpmtk.momenttoprofile import generate_profile







def dos(m_in,xs,ntries=20,n=200,scale=10.):
  """Return the density of states"""
  if scale is None: scale = 10.*np.max(np.abs(m_in.data)) # estimate of the value
  mus = random_trace(m_in/scale,ntries=ntries,n=n)
  check_scale(mus,scale)
  ys = generate_profile(mus,xs/scale) # generate the DOS
  return ys # return the DOS 




from .kpmtk.kernels import fejer_kernel
from .kpmtk.kernels import lorentz_kernel
from .kpmtk.kernels import jackson_kernel



def edge_dos(intra0,inter0,scale=4.,w=20,npol=300,ne=500,bulk=False,
                use_random=True,nrand=20):
  """Calculated the edge DOS using the KPM"""
  h = [[None for j in range(w)] for i in range(w)]
  intra = csc_matrix(intra0)
  inter = csc_matrix(inter0)
  for i in range(w): h[i][i] = intra
  for i in range(w-1): 
    h[i+1][i] = algebra.dagger(inter)
    h[i][i+1] = inter
  h = bmat(h) # sparse hamiltonian
  ds = np.zeros(ne)
  dsb = np.zeros(ne)
  norb = intra0.shape[0] # orbitals ina cell
  for i in range(norb):
    (xs,ys) = ldos0d(h,i=i,scale=scale,npol=npol,ne=ne) 
    ds += ys # store
    if bulk:
      (xs,zs) = ldos0d(h,i=w*norb//2 + i,scale=scale,npol=npol,ne=ne) 
      dsb += zs # store
  if not bulk: return (xs,ds/w)
  else: return (xs,ds/w,dsb/w)








