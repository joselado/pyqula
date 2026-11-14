# Laurent coefficients in the broadening of the dc limit of the general
# nonlinear recursion (nonlinearkernel.py), extracted on a Cauchy contour
# in the complex broadening. They are the cross-check of the recursion
# against the tau-decomposed dc formulas of nonlineardc.py; the derivation
# and the measured numbers are in future_development/nonlinear_response.md
# ("The dc limit and its Laurent structure in delta").
#
# At w_j = i delta for every field the order-n recursion gives
#
#   S(delta) = A_-n/delta^n + ... + A_-1/delta + A_0 + O(delta),
#
# meromorphic in delta, with poles at delta = -i dE_nm/p (p = 1..n) for
# every pair of bands in different degenerate multiplets on the mesh. A
# degenerate pair has its denominator exactly p i delta and feeds the pole
# at the origin. In the semiclassical picture with the same adiabatic
# switching, f_p = -(q/hbar) E.d f_(p-1)/(p delta), against the Boltzmann
# f_p = -tau (q/hbar) E.d f_(p-1), so
#
#   tau^n  (the n-field Drude term)  ->  1/(n! delta^n)
#   tau    (one pole, the Berry curvature dipole at n = 2)  ->  1/delta
#
# which gives A_-2 = sigma^ND/(2 tau^2) and A_-1 = sigma^BCD/tau at second
# order, and A_-3 = sigma^ND3/(6 tau^3) at third. At third order A_-1 and
# A_-2 mix several tau structures. A_0 is the delta^0 term of the
# adiabatic switching and matches none of the quantum-metric conventions.
#
# A polynomial fit over a real ladder of delta fails because of the poles
# at |delta| = |dE_nm|/p; the contour is exact up to the aliasing
# A_(k+N) r^N, which is (r/R)^N relative with R the convergence radius.
import numpy as np

from . import kubo
from . import nonlinearkernel
from . import shg
from ..htk.eigenvectors import hk_matrix_batch


_default_fraction = 0.3 # radius/convergence radius when no radius is given,
                        # (0.3)^24 = 3e-13 of aliasing with 24 points


def convergence_radius(h,order=2,nk=100,degeneracy_tol=1e-6):
    """Distance from the origin to the nearest pole off the origin of the
    order-n dc recursion on the nk mesh, min|dE_nm|/order over the pairs
    of bands in different degenerate multiplets (consecutive bands closer
    than degeneracy_tol times the hopping scale, the grouping of
    nonlineardc.py). The mesh sum has its poles exactly there, so this is
    the radius of convergence of its Laurent series. Infinity for a
    single band"""
    _check_order(order,h)
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    ks = kubo._kmesh(h,nk)
    gmin = np.inf
    n = tms.shape[1]
    nb = int(max(1,min(4096,2e8/(48.*n*n)))) # H(k), its list copy and eigvalsh
    for i0 in range(0,len(ks),nb): # loop over batches of k-points
        es = np.linalg.eigvalsh(hk_matrix_batch(hkgen,ks[i0:i0+nb]))
        gaps = np.diff(es,axis=1)
        gaps = gaps[gaps>=degeneracy_tol*scale] # different multiplets
        if gaps.size>0: gmin = min(gmin,float(np.min(gaps)))
    return gmin/order


def _check_order(order,h=None):
    if not isinstance(order,(int,np.integer)) or order<1: raise ValueError(
        "the order of the nonlinear conductivity is a positive integer, "
        "got "+repr(order))
    if h is not None and h.has_eh: raise NotImplementedError("the dc "
        "Laurent coefficients of the nonlinear conductivity are not "
        "implemented for Nambu (superconducting) Hamiltonians: the charge "
        "current operator there is not i[H,r]")


def laurent_coefficients(h,order=2,radius=None,npoints=24,nk=100,T=0.05,
        components=None,powers=None,degeneracy_tol=1e-6):
    """Laurent coefficients A_k of the dc limit of the order-n recursion,
    S(delta) = sum_k A_k delta^k at w_j = i delta for every field, from a
    Cauchy contour of npoints points delta_j = radius exp(2 pi i
    (j+1/2)/npoints):

      A_k = mean_j S(delta_j) delta_j^(-k)

    radius must be below convergence_radius(h,order,nk), the nearest pole
    off the origin; None takes 0.3 of it. powers are the k returned,
    -(order+1) to 0 by default, so that the first is a zero of the
    recursion. components are strings of order+1 Cartesian labels, the
    current first, as in nonlinearkernel.nonlinear_conductivity, all of
    them by default. T is the temperature of the occupations; the chemical
    potential is zero (use h.shift_fermi).

    Returns (powers, array) for a single component string, the array
    holding A_k for each power, and (powers, dict) keyed by the component
    otherwise. The coefficients are complex; for a real static field they
    are real and the imaginary parts are rounding.

    With the mapping tau^n -> 1/(n! delta^n) and tau -> 1/delta, A_-2 is
    sigma^ND/(2 tau^2) and A_-1 is sigma^BCD/tau at second order and A_-3
    is sigma^ND3/(6 tau^3) at third (see the header of this module)"""
    _check_order(order,h)
    if not isinstance(npoints,(int,np.integer)) or npoints<1:
        raise ValueError("npoints is the number of points of the contour, "
            "a positive integer, got "+repr(npoints))
    if powers is None: powers = np.arange(-order-1,1)
    powers = np.array(powers,dtype=np.int64)
    if powers.ndim!=1 or len(powers)==0: raise ValueError("powers are the "
        "integer powers k of the coefficients A_k, a nonempty list, got "
        +repr(powers))
    # A_k on the contour is A_k plus A_(k-N) r^(-N) + A_(k+N) r^N + ...;
    # the series starts at -order, so k-N has to stay below it
    if np.max(powers)>=npoints-order: raise ValueError("a contour of "
        +str(npoints)+" points aliases A_k onto A_(k-"+str(npoints)+"), "
        "which is nonzero for k >= "+str(npoints-order)+" at order "
        +str(order)+": use more points or lower powers, got powers up to "
        +str(int(np.max(powers))))
    R = convergence_radius(h,order=order,nk=nk,degeneracy_tol=degeneracy_tol)
    if radius is None:
        if not np.isfinite(R): raise ValueError("a single band has no pole "
            "off the origin, so no default radius: give radius explicitly")
        radius = _default_fraction*R
    if not radius>0.: raise ValueError("the radius of the contour in the "
        "complex broadening must be positive, got radius="+str(radius))
    if not radius<R: raise ValueError("the contour of radius "+str(radius)
        +" encloses a pole of the order-"+str(order)+" recursion, at "
        "|delta| = min|dE_nm|/order = "+str(R)+" on this mesh: use a "
        "radius below it, about 0.3 of it for 24 points")
    single = isinstance(components,str)
    if components is None: components = nonlinearkernel._all_components(order)
    if single: components = [components]
    ds = radius*np.exp(2j*np.pi*(np.arange(npoints)+0.5)/npoints)
    omegas = 1j*ds[:,None]*np.ones((1,order)) # every field at i delta
    s = nonlinearkernel.nonlinear_response(h,order,omegas,components,nk=nk,
            T=T,degeneracy_tol=degeneracy_tol)
    A = np.mean(s[:,None,:]*ds[:,None,None]**(-powers[None,:,None]),axis=0)
    if single: return powers,A[:,0]
    return powers,{c:A[:,i] for (i,c) in enumerate(components)}
