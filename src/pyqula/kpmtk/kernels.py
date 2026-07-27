import numpy as np
from numba import jit

# These damping-kernel coefficients depend only on n = len(mus), never on the
# moment VALUES themselves, so the same (n-dependent) coefficient sequence
# gets recomputed identically on every call with that n -- e.g. once per
# (row,col,k) triple in kpmtk.densitymatrix_kpm._dm_kpm_from_needed, which
# can mean tens of thousands of calls for a single VJinteraction SCF
# iteration. Plain Python loops calling np.cos/np.sin/np.tan per scalar
# element paid Python-level dispatch overhead on every one of those calls;
# @jit compiles the loop once and reuses machine code thereafter, cutting
# this from >80% of _dm_kpm_from_needed's total time (profiled: 17.4s of
# 21.4s on a 98-site/196-orbital honeycomb Hubbard system, nk=4, npol=200)
# to a small fraction of it, with no change to the (already correct)
# arithmetic.
@jit(nopython=True,cache=True)
def jackson_kernel(mus):
  """ Modify coeficient using the Jackson Kernel"""
  mo = mus.copy() # copy array
  n = len(mo)
  pn = np.pi/(n+1.) # factor
  for i in range(n):
    fac = ((n-i+1)*np.cos(pn*i)+np.sin(pn*i)/np.tan(pn))/(n+1)
    mo[i] *= fac
  return mo



@jit(nopython=True,cache=True)
def lorentz_kernel(mus):
  """ Modify coeficient using the Jackson Kernel"""
  mo = mus.copy() # copy array
  n = len(mo)
  pn = np.pi/(n+1.) # factor
  lamb = 3.
  for i in range(n):
    fac = np.sinh(lamb*(1.-i/n))/np.sinh(lamb)
    mo[i] *= fac
  return mo






@jit(nopython=True,cache=True)
def fejer_kernel(mus):
  """Default kernel"""
  n = len(mus)
  mo = mus.copy()
  for i in range(len(mus)):
    mo[i] *= (1.-float(i)/n)
  return mo


