import numpy as np
from numba import jit


def ldos_fourier_transform(r,ldos_r,q):
    """Direct discrete Fourier transform sum_i ldos_r[i]*exp(-i q.r_i) of a
    real-space scalar field defined on arbitrary site positions r (not
    necessarily a regular grid, unlike np.fft.fft2 as used for the k-space
    convolution QPI modes in chitk/qpi.py -- this matters here since
    multi-sublattice lattices, e.g. honeycomb, are not on a square grid),
    evaluated directly on an explicit q-mesh. Returns the complex
    amplitude per q; the QPI intensity is np.abs of it.

    Only evaluate this at q commensurate with the supercell the LDOS was
    computed on (see commensurate_qmesh below) -- at an incommensurate q
    a finite direct sum has leakage even for a perfectly periodic,
    impurity-free LDOS (the finite-size "form factor" of the sampled
    atoms), which would otherwise be mistaken for scattering signal."""
    r = np.array(r)
    q = np.array(q)
    ldos_r = np.array(ldos_r,dtype=np.float64)
    return _ft_jit(r[:,0],r[:,1],r[:,2],ldos_r,q[:,0],q[:,1],q[:,2])


@jit(nopython=True)
def _ft_jit(rx,ry,rz,f,qx,qy,qz):
    n = len(f)
    nq = len(qx)
    out = np.zeros(nq,dtype=np.complex128)
    for iq in range(nq):
        sr = 0.0
        si = 0.0
        for i in range(n):
            phase = rx[i]*qx[iq] + ry[i]*qy[iq] + rz[i]*qz[iq]
            sr += f[i]*np.cos(phase)
            si -= f[i]*np.sin(phase)
        out[iq] = sr + 1j*si
    return out


def commensurate_qmesh(g,nsuper):
    """q-mesh spanning one primitive Brillouin zone of the primitive
    geometry g (i.e. before building the supercell), at exactly the
    resolution a supercell of size nsuper supports: q = 2*pi*(m1/n1*b1 +
    m2/n2*b2), m1=0..n1-1, m2=0..n2-1 (recentered on Gamma), using g's
    primitive reciprocal vectors b1,b2 (a_i.b_j=delta_ij convention, see
    geometry.get_reciprocal -- the 2*pi here is what turns that into an
    actual Cartesian wavevector consistent with the exp(2*pi*i*k.n)
    Bloch phase convention used everywhere else in this package, e.g.
    geometrytk/bloch.py, and hence with real-space positions g.r).

    These are exactly the q at which a direct real-space Fourier sum
    over the supercell's atoms is free of finite-size leakage (see
    ldos_fourier_transform) -- this is also why no band-unfolding step
    is needed: the mesh already covers the full primitive Brillouin
    zone, just at a resolution set by nsuper, not a folded/reduced one
    that would need projecting back."""
    try: n1,n2 = int(nsuper[0]),int(nsuper[1])
    except (TypeError,IndexError): n1 = n2 = int(nsuper)
    b1,b2 = g.b1,g.b2
    m1s = np.arange(n1) - n1//2
    m2s = np.arange(n2) - n2//2
    return np.array([2.*np.pi*(m1/n1*b1 + m2/n2*b2) for m1 in m1s for m2 in m2s])
