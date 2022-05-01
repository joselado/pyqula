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

# check that the fortran library exists
try: 
  from . import kpmf90 
  use_fortran = True
except:
  use_fortran = False # use python routines




def get_moments(v,m,n=100,use_fortran=use_fortran,test=False):
    """ Get the first n moments of a certain vector
    using the Chebychev recursion relations"""
    if use_fortran:
        from .kpmf90 import get_momentsf90 # fortran routine
        mo = coo_matrix(m) # convert to coo matrix
        v = algebra.matrix2vector(v)
    # call the fortran routine
        mus = get_momentsf90(mo.row+1,mo.col+1,mo.data,v,n) 
        return mus # return fortran result
    else:
        if test: return python_kpm_moments_clear(v,m,n=n)
        else: return python_kpm_moments(v,m,n=n)


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






def get_momentsA(v,m,n=100,A=None):
    """ Get the first n moments of a certain vector
    using the Chebychev recursion relations"""
    if A is None: raise # only for a certain A
    mus = np.array([0.0j for i in range(n)]) # empty arrray for the moments
    v = algebra.matrix2vector(v) # zero vector
    A = csc_matrix(A) # turn sparse
    m = csc_matrix(m) # turn sparse
    return get_momentsA_jit(v,m,n,A,mus)

#@jit(nopython=True)
def get_momentsA_jit(v,m,n,A,mus):
    am = v*1.0
    a = m@v  # vector number 1
    bk = np.conjugate(v).dot(A@v) 
    bk1 = np.conjugate(a).dot(A@v)
    mus[0] = bk  # mu0
    mus[1] = bk1 # mu1
    for i in range(2,n): 
        ap = 2.*m@a - am # recursion relation
        bk = np.conjugate(ap).dot(A@v) 
        mus[i] = bk
        am = a.copy() # new variables
        a = ap.copy() # new variables
    mu0 = mus[0] # first
    mu1 = mus[1] # second
    return mus




def get_moments_ij(m0,n=100,i=0,j=0,use_fortran=use_fortran):
  """ Get the first n moments of a the |i><j| operator
  using the Chebychev recursion relations"""
  m = coo_matrix(m0,dtype=np.complex)
  if use_fortran:
    mus = kpmf90.get_moments_ij(m.row+1,m.col+1,m.data,n,m.shape[0],i+1,j+1)
    return mus
  else:
    mus = np.zeros(n,dtype=np.complex) # empty arrray for the moments
    v = np.zeros(m.shape[0],dtype=np.complex) ; v[i] = 1.0 # initial vector
    v = np.matrix([v]).T # zero vector
    am = v.copy()
    a = m*v  # vector number 1
    bk = v[j] # scalar product
    bk1 = a[j,0] # scalar product
    mus[0] = bk  # mu0
    mus[1] = bk1 # mu1
    for ii in range(2,n): 
      ap = 2.*m*a - am # recursion relation
      bk = ap[j,0] # scalar product
      mus[ii] = bk
      am = a.copy() # new variables
      a = ap.copy() # new variables
    return mus

def get_moments_vivj(m0,vi,vj,n=100,use_fortran=False):
  if not use_fortran: return get_moments_vivj_python(m0,vi,vj,n=n)
  else: return get_moments_vivj_fortran(m0,vi,vj,n=n)


def get_moments_vivj_python(m0,vi,vj,n=100):
  """ Get the first n moments of a the |i><j| operator
  using the Chebychev recursion relations"""
  m = csc_matrix(m0,dtype=np.complex)
  mus = np.zeros(n,dtype=np.complex) # empty arrray for the moments
  v = vi.copy()
  am = v.copy()
  a = m@v  # vector number 1
  bk = algebra.braket_ww(vj,v)
#  bk = (vj.H*v).todense().trace()[0,0] # calculate bk
  bk1 = algebra.braket_ww(vj,a)
#  bk1 = (vj.H*a).todense().trace()[0,0] # calculate bk
  mus[0] = bk  # mu0
  mus[1] = bk1 # mu1
  for ii in range(2,n): 
    ap = 2.*m@a - am # recursion relation
    bk = algebra.braket_ww(vj,ap)
#    bk = (vj.H*ap).todense().trace()[0,0]
    mus[ii] = bk
    am = a.copy() # new variables
    a = ap.copy() # new variables
  return mus



def get_moments_vivj_fortran(m0,vi,vj,n=100):
    raise # I haven't check this function
    mo = coo_matrix(m0) # convert to coo matrix
    vi1 = vi.todense() # convert to conventional vector
    vj1 = vj.todense() # convert to conventional vector
# call the fortran routine
    mus = get_moments_vivj(mo.row+1,mo.col+1,mo.data,vi,vj,n) 
    return mus # return fortran result



def full_trace(m_in,n=200,use_fortran=use_fortran):
  """ Get full trace of the matrix"""
  m = csc(m_in) # saprse matrix
  nd = m.shape[0] # length of the matrix
  mus = np.array([0.0j for i in range(2*n)])
#  for i in range(ntries):
  for i in range(nd):
    mus += local_dos(m_in,i=i,n=n,use_fortran=use_fortran)
  return mus/nd









def local_dos(m_in,i=0,n=200,use_fortran=use_fortran):
  """ Calculates local DOS using the KPM"""
  m = csc(m_in) # saprse matrix
  nd = m.shape[0] # length of the matrix
  mus = np.array([0.0j for j in range(2*n)])
  v = rand.random(nd)*0.
  v[i] = 1.0 # vector only in site i 
  v = csc(v).transpose()
# get the chebychev moments
  mus += get_moments(v,m,n=n,use_fortran=use_fortran) 
  return mus



def ldos(m_in,i=0,scale=10.,npol=None,ne=500,kernel="jackson"):
  """Return two arrays with energies and local DOS"""
  if npol is None: npol = ne
  mus = local_dos(csc_matrix(m_in)/scale,i=i,n=npol) # get coefficients
  xs = np.linspace(-1.0,1.0,ne,endpoint=True)*0.99 # energies
  ys = generate_profile(mus,xs,kernel=kernel)
  return (scale*xs,ys/scale)



ldos0d = ldos



def tdos(m_in,scale=10.,npol=None,ne=500,kernel="jackson",
              ntries=20,ewindow=None,frand=None,
              operator=None,x=None):
  """Return two arrays with energies and local DOS"""
  if npol is None: npol = ne
  mus = random_trace(m_in/scale,ntries=ntries,n=npol,fun=frand,
          operator=operator) 
  if ewindow is None or abs(ewindow)>scale: # no window provided
    xs = np.linspace(-1.0,1.0,ne,endpoint=True)*0.99 # energies
  else:
    xx = abs(ewindow/scale) # scale
    xs = np.linspace(-xx,xx,ne,endpoint=True)*0.99 # energies
  ys = generate_profile(mus,xs,kernel=kernel).real
  (xs,ys) = (scale*xs,ys/scale)
  if x is not None:
    from scipy.interpolate import interp1d
    f = interp1d(xs,ys,bounds_error=False,fill_value=0.0)
    return x,f(x)
  else: return xs,ys


def pdos(m,P=None,**kwargs):
    """Compute the projected density of states, assuming the operator
    fufills P^2 = P"""
    from .randomtk import randomwf
    fun0 = randomwf(m.shape[0]) # generator
    if P is not None: # operator provided
        from .operators import Operator
        op = Operator(P).get_matrix() # redefine
        fun = lambda : op@fun0() # define new generator
    else: fun = fun0 # original generator
    return tdos(m,frand=fun,**kwargs) # call TDOS with the generator



tdos0d = tdos # redefine


def total_energy(m_in,scale=10.,npol=None,ne=500,ntries=20):
   x,y = tdos0d(m_in,scale=scale,npol=npol,ne=ne,ntries=ntries)
   z = .5*(np.sign(x)+1.)*x*y # function to integrate
   return np.trapz(z,x)



def random_trace(m_in,ntries=20,n=200,fun=None,operator=None):
  """ Calculates local DOS using the KPM"""
  m = csc(m_in) # sparse matrix
  nd = m.shape[0] # length of the matrix
  if fun is not None: # check that dimensions are fine
    v0 = fun()
    if len(v0) != m_in.shape[0]: raise
  if fun is None:
#    def fun(): return rand.random(nd) -.5 + 1j*rand.random(nd) -.5j
      from .randomtk import randomwf
      fun = randomwf(nd) # generator
  def pfun(x):
    v = fun()
    v = v/np.sqrt(v.dot(np.conjugate(v))) # normalize the vector
#    v = csc(v).transpose()
    if operator is None:
        mus = get_moments(v,m,n=n) # get the chebychev moments
    else:
#        mus = get_moments_vivj(m,v,operator@v,n=2*n,use_fortran=False)
        mus = get_momentsA(v,m,n=2*n,A=operator) # get the chebychev moments
    return mus
  from . import parallel
  out = parallel.pcall(pfun,range(ntries))
  mus = np.zeros(out[0].shape,dtype=np.complex)
  for o in out: mus = mus + o # add contribution
  return mus/ntries



def random_trace_A(m_in,ntries=20,n=200,A=None):
  """ Calculates local DOS using the KPM"""
  m = csc(m_in) # saprse matrix
  nd = m.shape[0] # length of the matrix
  mus = np.array([0.0j for j in range(n)])
  for i in range(ntries): # loop over tries
    #v = rand.random(nd) - .5
    v = rand.random(nd) -.5 + 1j*rand.random(nd) -.5j
    v = v/np.sqrt(v.dot(v)) # normalize the vector
    v = csc(v).transpose()
    mus += get_momentsA(v,m,n=n,A=A) # get the chebychev moments
  return mus/ntries



def full_trace_A(m_in,ntries=20,n=200,A=None):
  """ Calculates local DOS using the KPM"""
  m = csc(m_in) # saprse matrix
  nd = m.shape[0] # length of the matrix
  mus = np.array([0.0j for j in range(2*n)])
  for i in range(nd): # loop over tries
    #v = rand.random(nd) - .5
    v = rand.random(nd)*0.
    v[i] = 1.0 # vector only in site i 
    v = csc(v).transpose()
    mus += get_momentsA(v,m,n=n,A=A) # get the chebychev moments
  return mus/nd



def correlator0d(m_in,i=0,j=0,scale=10.,npol=None,ne=500,write=True,
  x=None):
  """Return two arrays with energies and local DOS"""
  if npol is None: npol = ne
  mus = get_moments_ij(m_in/scale,n=npol,i=i,j=j,use_fortran=True)
  if np.sum(np.abs(mus.imag))>0.001:
#    print("WARNING, off diagonal has nonzero imaginary elements",np.sum(np.abs(mus.imag)))
     pass
  if x is None: xs = np.linspace(-1.0,1.0,ne,endpoint=True)*0.99 # energies
  else: xs = x/scale # use from input
  ys = generate_green_profile(mus,xs,kernel="jackson",use_fortran=False)/scale*np.pi # so it is the Green function
#  imys = hilbert(ys).imag
  if write: np.savetxt("CORRELATOR_KPM.OUT",np.matrix([scale*xs,-ys.imag,ys.real]).T)
  return (scale*xs,ys.real,ys.imag)




def dm_ij_energy(m_in,i=0,j=0,scale=10.,npol=None,ne=500,x=None):
  """Return the correlation function"""
  if npol is None: npol = ne
  mus = get_moments_ij(m_in/scale,n=npol,i=i,j=j,use_fortran=use_fortran)
  if np.sum(np.abs(mus.imag))>0.001:
#    print("WARNING, off diagonal has nonzero imaginary elements",np.sum(np.abs(mus.imag)))
    pass
  if x is None: xs = np.linspace(-1.0,1.0,ne,endpoint=True)*0.99 # energies
  else: xs = x/scale # use from input
  ysr = generate_profile(mus.real,xs,kernel="jackson",use_fortran=use_fortran)/scale*np.pi # so it is the Green function
  ysi = generate_profile(mus.imag,xs,kernel="jackson",use_fortran=use_fortran)/scale*np.pi # so it is the Green function
  ys = ysr - 1j*ysi
  return (scale*xs,ys)



def dm_vivj_energy(m_in,vi,vj,scale=10.,npol=None,ne=500,x=None):
  """Return the correlation function"""
  if npol is None: npol = ne
  mus = get_moments_vivj(m_in/scale,vi,vj,n=npol)
  if np.sum(np.abs(mus.imag))>0.001:
#    print("WARNING, off diagonal has nonzero imaginary elements",np.sum(np.abs(mus.imag)))
    pass
  if x is None: xs = np.linspace(-1.0,1.0,ne,endpoint=True)*0.99 # energies
  else: xs = x/scale # use from input
  ysr = generate_profile(mus.real,xs,kernel="lorentz",use_fortran=use_fortran)/scale*np.pi # so it is the Green function
  ysi = generate_profile(mus.imag,xs,kernel="jackson",use_fortran=use_fortran)/scale*np.pi # so it is the Green function
  ys = ysr - 1j*ysi
  return (scale*xs,ys)







def generate_profile(mus,xs,kernel="jackson",use_fortran=use_fortran):
  """ Uses the Chebychev expansion to create a certain profile"""
  # initialize polynomials
#  xs = np.array([0.])
  tm = np.zeros(xs.shape) +1.
  t = xs.copy()
  if kernel=="jackson": mus = jackson_kernel(mus)
  elif kernel=="lorentz": mus = lorentz_kernel(mus)
  else: raise
  if use_fortran: # call the fortran routine
    ys = kpmf90.generate_profile(mus,xs) 
  else: # do a python loop
    ys = np.zeros(xs.shape,dtype=np.complex) + mus[0] # first term
    # loop over all contributions
    for i in range(1,len(mus)):
      mu = mus[i]
      ys += 2.*mu*t # add contribution
      tp = 2.*xs*t - tm # chebychev recursion relation
      tm = t + 0.
      t = 0. + tp # next iteration
    ys = ys/np.sqrt(1.-xs*xs) # prefactor
  ys = ys/np.pi
  return ys




def generate_green_profile(mus,xs,kernel="jackson",use_fortran=use_fortran):
  """ Uses the Chebychev expansion to create a certain profile"""
  # initialize polynomials
#  xs = np.array([0.])
  tm = np.zeros(xs.shape) +1.
  t = xs.copy()
  ys = np.zeros(xs.shape,dtype=np.complex) + mus[0]/2 # first term
  if kernel=="jackson": mus = jackson_kernel(mus)
  elif kernel=="lorentz": mus = lorentz_kernel(mus)
  else: raise
  if True:
    for i in range(1,len(mus)): # loop over mus
      ys += np.exp(1j*i*np.arccos(xs))*mus[i] # add contribution
    ys = ys/np.sqrt(1.-xs*xs)
    return 1j*2*ys/np.pi
#    if use_fortran: # call the fortran routine
#      ys = kpmf90.generate_profile(mus,xs) 
#    return ys








def dos(m_in,xs,ntries=20,n=200,scale=10.):
  """Return the density of states"""
  if scale is None: scale = 10.*np.max(np.abs(m_in.data)) # estimate of the value
  mus = random_trace(m_in/scale,ntries=ntries,n=n)
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
    h[i+1][i] = inter.H
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








