import numpy as np

# A Chebyshev expansion of m/scale is only meaningful when the spectrum of
# m lies inside [-scale,scale]. For a Hermitian matrix with its spectrum
# inside [-1,1] every moment is bounded, |<a|T_n(m)|b>| <= |a||b|, while
# T_n grows exponentially outside that interval, so a moment past its bound
# (or a NaN/inf) says that the scale is too small, exactly and at no extra
# cost. A comparison with a Gershgorin bound would instead refuse a valid
# scale between the spectral radius and the bound.

# the recursion in single precision drifts by roughly n*eps, 3e-5 at
# npol=500, so a spectrum edge sitting at +-1 needs a looser tolerance there
_tolerance = {"double": 1e-6, "single": 1e-3}


def moments_within_bound(mus,bound=1.,kpm_prec="double"):
    """Return True when every moment is finite and at most bound in
    modulus, up to the roundoff of the recursion"""
    mus = np.asarray(mus)
    if not np.all(np.isfinite(mus)): return False
    return np.max(np.abs(mus)) <= bound*(1. + _tolerance[kpm_prec])


def check_scale(mus,scale,bound=1.,kpm_prec="double"):
    """Raise if the Chebyshev moments of m/scale show part of the spectrum
    of m outside [-scale,scale]. bound is the largest modulus a moment can
    have when the scale is right: one for a unit starting vector or an
    average over them, |vi||vj| for the moments <vi|T_n|vj>, the norm of
    the operator for operator-weighted ones"""
    if moments_within_bound(mus,bound=bound,kpm_prec=kpm_prec): return
    raise ValueError("the KPM scale=%g does not cover the spectrum of the "
            "matrix: the Chebyshev expansion needs every eigenvalue inside "
            "[-scale,scale], and the moments grow past their bound of %g, "
            "which only happens when it is not. Pass a scale at least as "
            "large as the largest eigenvalue in modulus" % (scale,bound))
