# functions to compute the electronic density directly from the
# moments of the KPM LDOS


import numpy as np
from .ldos import moments_local_dos
from .kernels import jackson_kernel
from .kernels import lorentz_kernel
from .kernels import fejer_kernel

from .bandwidth import estimate_bandwidth

# damping kernels accepted by kernel=, the same three that
# momenttoprofile.generate_profile takes
kernels = {"jackson": jackson_kernel,
           "lorentz": lorentz_kernel,
           "fejer": fejer_kernel}

def get_density(m_in,scale=None,fermi=0.,
        delta = 1e-2, npol=None,
        kernel="jackson",**kwargs):
  """Return the electronic density below the energy fermi, for the
  vector selected by the keywords of moments_local_dos (i=site index).

  delta: energy resolution, which sets npol when it is not given
  npol: number of Chebyshev polynomials
  kernel: damping kernel of the expansion"""
  if kernel not in kernels:
      raise ValueError("unknown kernel "+repr(kernel)+"; the accepted ones "
              "are "+", ".join(sorted(kernels)))
  if scale is None: scale = estimate_bandwidth(m_in)
  if npol is None: npol = max([int(scale/delta),3])
  mus = moments_local_dos(m_in/scale,n=npol,**kwargs) # get coefficients
  mus = kernels[kernel](mus) # damp the moments
  return get_density_from_mus(mus,fermi/scale) # obtain the density directly



def get_density_from_mus(mus,x):
    """Integrate the Chebyshev expansion of the DOS from -1 to x,

    N(x) = (mu_0 arccos(-x) - 2 sum_n mu_n sin(n arccos(x))/n)/pi

    which follows from int_{-1}^x T_n(y)/sqrt(1-y^2) dy = -sin(n arccos(x))/n
    for n>0. mus are the already damped moments, x the energy in units of
    the scale of the expansion."""
    x = np.clip(x,-1.,1.) # the whole spectrum lies inside (-1,1)
    ns = np.arange(1,len(mus))
    Gn = np.sin(ns*np.arccos(x))/ns # vectorized over n>0
    rho = (mus[0]*np.arccos(-x) - 2*np.sum(mus[ns]*Gn))/np.pi
    return rho.real
