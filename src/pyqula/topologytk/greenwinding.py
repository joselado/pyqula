"""Chern number of the bands of a Green's function inside a window of
frequencies, from the winding number of its inverse over the frequency and
the two momenta.

For a Bloch Hamiltonian H(q) the Green's function is G = (z - H(q))^-1, and
the Chern number of a group of bands is the winding of G over (z, q1, q2),

    N3 = 1/(24 pi^2) int dz d^2q eps^{mu nu rho}
             Tr[G d_mu G^-1  G d_nu G^-1  G d_rho G^-1] ,

the N2 of Wang, Qi and Zhang (arXiv:1004.4229), whose prefactor
(pi/3)/(2 pi)^3 is this 1/(24 pi^2). There the frequency runs along the
Matsubara axis, which encloses every occupied state. Here it runs on a
closed contour that crosses the real axis at the two ends of a window
[w_a, w_b], so it counts the bands inside the window and nothing else. The
three-form is closed, so N3 does not change under a smooth deformation of
G^-1 that keeps it invertible on the contour, and it is additive over
products, N3[AB] = N3[A] + N3[B]. As a result it takes any matrix function
G^-1(z, q) analytic in z inside the contour: a Green's function with a
frequency-dependent self-energy, or the kernel chi0^-1 - V of an RPA
response, whose zeros are its collective modes (chitk/responsewinding.py).

Why not topologytk/green.py. Its integrand, -d_xG G^-1 d_yG + (x<->y), is
the one above only when d_z G^-1 = 1, which is G^-1 = z - H. The extra
factor G d_z G^-1 here is what makes the result a winding number. A check
of the difference: G^-1 = D^dag (z - H) D with a constant non-unitary D gives
A_mu = D^-1 (g d_mu g^-1) D, a similarity, so the integrand here does not
change at any q and z, while the trace of the two-derivative integrand picks
up a factor (D D^dag)^-1 inside it.

The formula that is computed. By the cyclicity of the trace the six terms of
the epsilon sum are three copies of one, so with A_mu = G d_mu G^-1

    N3 = 1/(8 pi^2) int d^2q  contour-int dz  Tr[A_z (A_1 A_2 - A_2 A_1)] ,

with q in reduced coordinates on [0,1)^2 and no Jacobian, since N3 is the
integral of a three-form. For G^-1 = z - H the contour integral is done by
residues: with P the bands enclosed and Omega their summed Berry curvature
in the convention A = i<u|grad u>, it is -4 pi Omega(q). So N3 = -C for a
counterclockwise contour with the order (z, q1, q2), and what the functions
here return is C = -N3, the sign of h.get_chern() and of
topologytk/plaquette.py (tests/topology/test_green_winding.py checks it on
the Haldane model with both signs of the second-neighbor hopping).

The zeros of G. A pole of G^-1 inside the window, a zero of G, adds to the
result with the opposite sign of a band: if G^-1 goes as R(q)/(z - z0)
around it, with the range of R a bundle of Chern number c, the window gets
-c. So the result is the sum of the Chern numbers of the poles of G inside
the window minus those of its zeros, the interchangeable role of poles and
zeros discussed in arXiv:2301.05588. A self-energy from a bath level with a
smooth hybridization that never vanishes, V^2/(z - e_b) with the same V on
every site, has a zero whose residue spans the whole space, a trivial
bundle, so it adds nothing; a residue that projects on a band with a Chern
number does add, whether the projector commutes with H or not (the three
cases are in the test; in the last one the poles of G are counted with
the Chern number of their residues, which are not the bands of H).

The contour is the rectangle with corners w_a, w_b, w_b + i*height and
w_a + i*height, and its mirror image below the real axis, run
counterclockwise. The two ends of the window must lie in gaps at every q,
and nothing may be singular on the contour. Each of the three sides of the
upper half carries nz Gauss-Legendre nodes. When G^-1(conj z) = G^-1(z)^dag,
which holds for z - H(q) with H Hermitian and for any response built from a
Hermitian operator basis, the lower half is the complex conjugate of the
upper half and the contour integral is twice the real part of the upper
half, so only the upper half is evaluated (hermitian=True, checked on the
first node). The z integral converges exponentially in nz, at a rate set by
the distance from the window ends to the nearest band over the height of
the contour.

The discretization in q. The q integral is a sum over the nq x nq mesh
q = (i/nq, j/nq) + shift of topologytk.plaquette.mesh, the mesh of the
other routes, with nq of at least 3 here, which the central difference
needs, and the derivatives of G^-1 along q1 and q2 are
taken on the same mesh, so G^-1 is evaluated once per node and per q. As a
result the number is not an integer on a finite mesh; it converges to the
integer as nq grows, and its distance from the nearest integer is the
measure of how converged it is. The spectral derivative (an FFT along each
direction of the periodic mesh) is exact for a tight-binding H(q), a
trigonometric polynomial, once nq exceeds twice the range of its hoppings,
and the sum over the mesh of the analytic integrand then converges
exponentially, at a rate set by the gap at the window ends (measured on the
Haldane model: 2.7e-6 from the integer at nq=24 with a gap of 0.9, 8e-9 at
nq=18 with a gap of 1.8). The central difference converges as 1/nq^2
(0.023 and 0.013 at nq=24 and 32). This is the opposite of the plaquette sum
of topologytk/plaquette.py, which is an integer on every mesh and is right
only once the mesh resolves the states."""

import numpy as np

from . import plaquette


# C = SIGN*N3, with N3 the winding of the module docstring
SIGN = -1.


def _check_window(window):
    w = np.array(window, dtype=np.float64).reshape(-1)
    if w.shape[0] != 2 or not w[0] < w[1]:
        raise ValueError("window is the interval [w_a, w_b] of the real "
                "frequency axis that the contour encloses, two increasing "
                "numbers, got "+str(window))
    return w


def _check_count(n, name, least):
    if int(n) != n or n < least:
        raise ValueError(name+" must be an integer of at least "+str(least)
                +", got "+str(n))
    return int(n)


def contour(window, nz=12, height=None):
    """Return (zs, dzs): the nodes and the weights of the upper half of the
    counterclockwise rectangle around the window, from w_b up to
    w_b + i*height, along the top to w_a + i*height, and down to w_a, with
    nz Gauss-Legendre nodes on each side. sum(f(zs)*dzs) is the integral
    of f along that path. height defaults to half the width of the
    window."""
    wa, wb = _check_window(window)
    nz = _check_count(nz, "nz, the number of nodes on each side,", 1)
    if height is None: height = (wb - wa)/2.
    if not height > 0.:
        raise ValueError("height is the imaginary part of the top of the "
                "contour and must be positive, got "+str(height))
    x, w = np.polynomial.legendre.leggauss(nz)
    t, w = (x + 1.)/2., w/2. # nodes and weights on [0,1]
    # the left side reuses the nodes of the right one in reverse order, so
    # the two sides share their imaginary parts exactly, which a caller
    # can use to group the frequencies (chitk/responsewinding.py does)
    zs = np.concatenate([wb + 1j*height*t, # right side, going up
                         wb + (wa - wb)*t + 1j*height, # top, going left
                         wa + 1j*height*t[::-1]]) # left side, going down
    dzs = np.concatenate([1j*height*w, (wa - wb)*w + 0j, -1j*height*w[::-1]])
    return zs, dzs


def _central_derivative(f, axis):
    """Derivative along a periodic mesh of spacing 1/n, second order"""
    n = f.shape[axis]
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis))*n/2.


def _spectral_derivative(f, axis):
    """Derivative along a periodic mesh of period 1, from its Fourier
    series; the Nyquist mode of an even mesh has no derivative"""
    n = f.shape[axis]
    m = np.fft.fftfreq(n, d=1./n) # integer frequencies
    if n % 2 == 0: m[n//2] = 0.
    shape = [1]*f.ndim ; shape[axis] = n
    fm = np.fft.fft(f, axis=axis)*(2j*np.pi*m).reshape(shape)
    return np.fft.ifft(fm, axis=axis)


_DERIVATIVES = {"spectral": _spectral_derivative,
                "central": _central_derivative}


def get_derivative_names():
    """The accepted values of derivative="""
    return sorted(_DERIVATIVES)


def _evaluate(f, zs, q, batched):
    """f at every node of zs for one q, as a (nz, D, D) complex array"""
    if batched: m = np.asarray(f(zs, q), dtype=np.complex128)
    else: m = np.array([f(z, q) for z in zs], dtype=np.complex128)
    if m.ndim == 1 and m.shape[0] == len(zs): m = m[:, None, None] # 1x1
    if m.ndim != 3 or m.shape[0] != len(zs) or m.shape[1] != m.shape[2]:
        raise ValueError("ginv must return a square matrix at every "
                "frequency (a (nz, D, D) array when batched=True), got "
                "shape "+str(m.shape)+" for "+str(len(zs))+" frequencies at "
                "q="+str(q))
    return m


def _check_conjugation(ginv, z, q, batched, tol=1e-8):
    """Raise if ginv(conj z) is not ginv(z)^dag at one node"""
    a = _evaluate(ginv, np.array([z]), q, batched)[0]
    b = _evaluate(ginv, np.array([np.conj(z)]), q, batched)[0]
    err = np.max(np.abs(b - np.conj(a.T)))/max(np.max(np.abs(a)), 1e-300)
    if err > tol:
        raise ValueError("hermitian=True needs ginv(conj z) = ginv(z)^dag, "
            "but the relative difference is %.1e at z=%s, q=%s; pass "
            "hermitian=False to integrate the whole contour" % (err, z, q))


def window_chern_density(ginv, window, nq=10, nz=12, height=None,
                         dginv_dz=None, dz=None, derivative="spectral",
                         hermitian=True, batched=False, shift=(0., 0.)):
    """Return the (nq, nq) array c(q) on the mesh q = (i/nq, j/nq) + shift
    whose mean is the Chern number of the bands of G = ginv^-1 inside the
    window (minus that of its zeros, see the module docstring), with the
    sign of h.get_chern(). Its mean is an integer only in the limit of a
    fine mesh.

    ginv: callable (z, q) -> (D, D) matrix, the inverse Green's function at
        the complex frequency z and the reduced momentum q = [q1, q2, 0],
        analytic in z inside the contour. With batched=True it is called
        as ginv(zs, q) with an array of frequencies and returns a
        (len(zs), D, D) array, which lets a caller share work between
        frequencies.
    window: [w_a, w_b], the interval of the real axis enclosed. Both ends
        must lie in gaps of the spectrum at every q.
    nq: points of the mesh along each reciprocal lattice vector.
    nz, height: nodes on each side of the contour and the imaginary part
        of its top, see contour().
    dginv_dz: callable with the signature of ginv returning d ginv/dz, or
        None for the central difference (ginv(z+dz) - ginv(z-dz))/(2 dz)
        along the real direction, which is right for an analytic ginv.
        dz defaults to 1e-5 times the width of the window.
    derivative: how the q derivatives are taken on the mesh, "spectral"
        (exponential convergence in nq) or "central" (as 1/nq^2).
    hermitian: True when ginv(conj z) = ginv(z)^dag, so that only the upper
        half of the contour is evaluated; False evaluates both halves."""
    wa, wb = _check_window(window)
    nq = _check_count(nq, "nq, the number of points of the mesh along each "
                      "direction,", 3)
    # the points of the mesh of topologytk.plaquette, which checks the shift
    qs = list(plaquette.mesh(nq, shift).reshape(-1, 3))
    if derivative not in _DERIVATIVES:
        raise ValueError("unknown derivative "+repr(derivative)+"; the "
                "accepted ones are "+", ".join(get_derivative_names()))
    deriv = _DERIVATIVES[derivative]
    if not callable(ginv):
        raise TypeError("ginv must be a callable (z, q) -> matrix, got "
                +str(type(ginv)))
    zs, dzs = contour(window, nz=nz, height=height)
    if not hermitian: # the lower half, the mirror image run backwards
        zs = np.concatenate([zs, np.conj(zs)])
        dzs = np.concatenate([dzs, -np.conj(dzs)])
    if dz is None: dz = 1e-5*(wb - wa)
    nzt = len(zs)
    if hermitian: _check_conjugation(ginv, zs[nzt//2], qs[0], batched)
    gi, gz = [], [] # ginv and its z derivative at every q and node
    for q in qs:
        if dginv_dz is None: # three frequencies per node, in one call
            m = _evaluate(ginv, np.concatenate([zs, zs + dz, zs - dz]), q,
                          batched)
            gi.append(m[:nzt]) ; gz.append((m[nzt:2*nzt] - m[2*nzt:])/(2.*dz))
        else:
            gi.append(_evaluate(ginv, zs, q, batched))
            gz.append(_evaluate(dginv_dz, zs, q, batched))
    dim = gi[0].shape[1]
    gi = np.array(gi).reshape(nq, nq, nzt, dim, dim)
    gz = np.array(gz).reshape(nq, nq, nzt, dim, dim)
    total = np.zeros((nq, nq), dtype=np.complex128)
    for iz in range(nzt): # one node at a time, to bound the memory
        m = gi[:, :, iz]
        g = np.linalg.inv(m)
        a1 = g@deriv(m, 0) # G d_1 G^-1
        a2 = g@deriv(m, 1) # G d_2 G^-1
        az = g@gz[:, :, iz] # G d_z G^-1
        f = np.trace(az@(a1@a2 - a2@a1), axis1=2, axis2=3)
        total += f*dzs[iz]
    if hermitian: total = 2.*total.real # the lower half is the conjugate
    return SIGN*total.real/(8.*np.pi**2)


def window_chern(ginv, window, **kwargs):
    """Chern number of the bands of G = ginv^-1 inside the window, minus
    that of its zeros, as a float that converges to an integer as the mesh
    is refined. The arguments are those of window_chern_density, and the
    result is the mean of its array."""
    return float(np.mean(window_chern_density(ginv, window, **kwargs)))
