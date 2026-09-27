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






# The energy resolution of a Jackson-damped expansion, and the one
# convention for turning a broadening delta into a number of polynomials.
#
# With N Chebyshev moments the Jackson kernel broadens a level at x=a of the
# rescaled spectrum into a peak that is very nearly a Gaussian, of standard
# deviation sigma = (pi/N) sqrt(1-a^2) (Weisse et al., Rev. Mod. Phys. 78,
# 275 (2006), arXiv:cond-mat/0504627, Eqs. (71), (75) and (76); the half
# width at half maximum of the kernel itself, measured at a=0 for N=400 to
# 4000, is 1.19 pi/N, against sqrt(2 ln 2) pi/N = 1.18 pi/N for the
# Gaussian). In energy units, for an expansion of m/scale, that is
# sigma = pi scale/N at the center of the spectrum.
#
# The convention is that delta is the half width at half maximum of that
# peak, sqrt(2 ln 2) sigma: the half width of the Lorentzian that delta
# gives in the exact-diagonalization and Green's function modes, where it is
# the imaginary part of the energy. The same delta then gives peaks of the
# same width in every mode, and a KPM curve can be laid over an ED one. The
# heights still differ, since a Gaussian of the same half width is about 1.5
# times taller than the Lorentzian and has no long tails.
#
# Every routine of the package computes 2*npol moments for npol polynomials
# (kpm_moments_v and its batched and pairwise versions), so N = 2 npol.
_HWHM_PER_SIGMA = np.sqrt(2.*np.log(2.)) # half width over standard deviation


def jackson_npol(scale,delta):
    """Number of polynomials npol (2*npol Chebyshev moments) for which the
    Jackson kernel broadens a level at the center of the spectrum of m/scale
    into a peak of half width at half maximum delta, the same width that
    delta gives as a Lorentzian in the exact-diagonalization and Green's
    function modes. The width narrows as sqrt(1-(E/scale)^2) away from the
    center, by 5% at E=0.3 scale. See jackson_hwhm for the inverse"""
    d = np.real(delta)
    if not np.isfinite(d) or d<=0.:
        raise ValueError("delta is the energy resolution of the Chebyshev "
                "expansion, the half width at half maximum of the peak a "
                "single level gives, and must be a finite positive number, "
                "got "+str(delta))
    nmoments = np.pi*scale*_HWHM_PER_SIGMA/d # N moments
    return max(int(np.ceil(nmoments/2.)),3)


def jackson_hwhm(scale,npol):
    """Half width at half maximum of the peak that a level at the center of
    the spectrum of m/scale gives in an expansion with npol polynomials
    (2*npol moments) and the Jackson kernel, the inverse of jackson_npol"""
    return np.pi*scale*_HWHM_PER_SIGMA/(2.*npol)



@jit(nopython=True,cache=True)
def fejer_kernel(mus):
  """Default kernel"""
  n = len(mus)
  mo = mus.copy()
  for i in range(len(mus)):
    mo[i] *= (1.-float(i)/n)
  return mo


