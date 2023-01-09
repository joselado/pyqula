import numpy as np
from .. import algebra


def berry_green_generator(f,k=[0.,0.,0.],dk=0.05,operator=None,
              full=False):
  """Function that returns the energy resolved Berry curvature"""
  k = np.array(k) # set as array
  dx = np.array([dk,0.,0.])
  dy = np.array([0.,dk,0.])
  def fint(e): # function to integrate
#    g = f(e=e,k=k) # compute at this k and this energy
    gxp = f(e=e,k=k+dx) # compute at this k and this energy
    gxm = f(e=e,k=k-dx) # compute at this k and this energy
    gyp = f(e=e,k=k+dy) # compute at this k and this energy
    gym = f(e=e,k=k-dy) # compute at this k and this energy
    g = (gxp + gyp + gxm + gym)/4. # average Green function
    # Now apply the formula
    gI = algebra.inv(g) # inverse
    # the derivative of g^-1 is -g^-1*g'*g^-1
    omega = -(gxp-gxm)@gI@(gyp-gym) + (gyp-gym)@gI@(gxp-gxm)
#    omega = ((gxp-gxm)@(gyp-gym) - (gyp-gym)@(gxp-gxm))@gI
#    omega = g*((gxp.I-gxm.I)*(gyp-gym) -(gyp.I-gym.I)*(gxp-gxm))
#    omega += -g*(gyp.I-gym.I)*(gxp-gxm)
    if operator is not None: omega = operator(omega,k=k)
    if full: return omega/(4.*dk*dk*2.*np.pi) # return the full matrix
    else: return np.trace(omega)/(4.*dk*dk*2.*np.pi) # return contribution
  return fint # return the function



def berry_green(f,emin=-10.0,k=[0.,0.,0.],ne=100,dk=0.0001,operator=None):
  """Return the Berry curvature using Green functions"""
  import scipy.integrate as integrate
  fint = berry_green_generator(f,k=k,dk=dk,operator=operator)
  es = np.linspace(emin,0.,ne) # energies used for the integration
  ### The original function is defined in the coplex plane,
  # we will do a change of variables of the form z = re^(iphi) - r0
  # so that dz = re^(iphi) i dphi
  def fint2(x):
    """Function to integrate using a complex contour, from 0 to 1"""
    z0 = emin*np.exp(-1j*x*np.pi)/2.
    z = z0 + emin/2.
    return -(fint(z)*z0).imag*np.pi # integral after the change of variables
  return integrate.quad(fint2,0.0,1.0,limit=60,epsabs=0.1,epsrel=0.1)[0]
#  return integrate.quad(fint,emin,0.0,limit=60,epsabs=0.01,epsrel=0.01)[0]
#  return np.sum([fint(e) for e in es]) # return




def berry_operator(h,delta=1e-1,**kwargs):
    """Return ap operator that computes the Berry curvature for a certain
    wavefunction"""
    if h.dimensionality!=2: raise
    h = h.copy()
    hk = h.get_hk_gen() # get generator
    gk = h.get_gk_gen(delta=delta) # get generator
    if not h.is_sparse: # dense Hamiltonians
        def bk(k): return berry_green_generator(gk,k=k,full=True,**kwargs)
        def outf(w,k=[0.,0.,0.]):
            m = hk(k) # bloch Hamiltonian
            e = algebra.braket_wAw(w,m) # energy
            o = bk(k)(e)@(delta*w) # Berry curvature
            return o # return a vector
        return outf
    else:
        def gI(**kwargs): # workaround for inverse of Green's function
            return gk(**kwargs,inv=True) # make the inverse
        def bk(k): return berry_green_generator_sparse(gk,k=k,gI=gI,**kwargs)
        def outf(w,k=[0.,0.,0.]):
            m = hk(k) # bloch Hamiltonian
            e = algebra.dot(w,m*w) # energy
            w2 = bk(k)(e)*(delta*w) # Operator times WF
            # to think, perhaps this should be <v|B|v>|v>
            # currently it is B|v>
            # (this does not change the expectation value though)
            return w2 # return a vector
        return outf




def berry_green_generator_sparse(f,k=[0.,0.,0.],
        dk=0.05,gI=None):
  """Function that returns the energy resolved Berry curvature"""
  k = np.array(k) # set as array
  dx = np.array([dk,0.,0.])
  dy = np.array([0.,dk,0.])
  def fint(e): # function to return
    gxp = f(e=e,k=k+dx) # compute at this k and this energy
    gxm = f(e=e,k=k-dx) # compute at this k and this energy
    gyp = f(e=e,k=k+dy) # compute at this k and this energy
    gym = f(e=e,k=k-dy) # compute at this k and this energy
    # Now apply the formula
    gI0 = gI(e=e,k=k) # inverse of the Green function
    # the derivative of g^-1 is -g^-1*g'*g^-1
    omega = -(gxp-gxm)@gI0@(gyp-gym) + (gyp-gym)@gI0@(gxp-gxm)
    return omega/(4.*dk*dk*2.*np.pi) # return the full matrix
  return fint # return the function

