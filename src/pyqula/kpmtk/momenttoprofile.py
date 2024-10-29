import numpy as np

from numba import jit



from .kernels import fejer_kernel
from .kernels import lorentz_kernel
from .kernels import jackson_kernel



def generate_profile(mus,xs,kernel="jackson",**kwargs):
    """ Uses the Chebychev expansion to create a certain profile"""
    # initialize polynomials
    if kernel=="jackson": mus = jackson_kernel(mus)
    elif kernel=="lorentz": mus = lorentz_kernel(mus)
    elif kernel=="fejer": mus = fejer_kernel(mus)
    else: raise
    ys = generate_profile_jit(mus,xs)
    return ys


@jit(nopython=True)
def generate_profile_jit(mus,xs):
    """Numba function to generate the moments"""
    ys = np.zeros(xs.shape,dtype=np.complex128) + mus[0] # first term
    t = xs.copy()
    tm = np.zeros(xs.shape) + 1.
    t = xs.copy()
    for i in range(1,len(mus)):
      mu = mus[i]
      ys += 2.*mu*t # add contribution
      tp = 2.*xs*t - tm # chebychev recursion relation
      tm = t + 0.
      t = 0. + tp # next iteration
    ys = ys/np.sqrt(1.-xs*xs) # prefactor
    ys = ys/np.pi
    return ys




def generate_green_profile(mus,xs,kernel="jackson",**kwargs):
  """ Uses the Chebychev expansion to create a certain profile"""
  # initialize polynomials
  tm = np.zeros(xs.shape) +1.
  t = xs.copy()
  ys = np.zeros(xs.shape,dtype=np.complex128) + mus[0]/2 # first term
  if kernel=="jackson": mus = jackson_kernel(mus)
  elif kernel=="lorentz": mus = lorentz_kernel(mus)
  elif kernel=="fejer": mus = fejer_kernel(mus)
  else: raise
  for i in range(1,len(mus)): # loop over mus
    ys += np.exp(1j*i*np.arccos(xs))*mus[i] # add contribution
  ys = ys/np.sqrt(1.-xs*xs)
  return 1j*2*ys/np.pi






