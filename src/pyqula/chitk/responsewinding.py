"""Chern number of the collective modes of the site-basis spin RPA, from
the winding of its kernel over a window of frequencies
(topologytk/greenwinding.py), with the response playing the role of the
Green's function.

The dressed response of chitk.rpa.chi_ops_RPA is

    chi(q,z) = chi0 (1 - V(q) chi0)^-1 = (chi0(q,z)^-1 - V(q))^-1 ,

so its inverse, the kernel K(q,z) = chi0^-1 - V, takes the place of G^-1:
the collective modes are the zeros of K, its null vector at a mode is the
residue vector of chi there, and the winding of K over a window
[w_a, w_b] times the Brillouin zone is the sum of the Chern numbers of the
mode bands inside the window. Both ends of the window must lie in gaps of
the mode spectrum at every q and below the continuum, the spin-flip
continuum of a magnet, where chi0 has its poles.

The bubble at a complex frequency. localspinchi.spin_chi with energies=x
and delta=y sums fac X X^dag/(e1 - e2 - x + i y), with fac the real
difference of the occupations, so it is the analytic function

    chi0(q,z) = 1/Nk sum_k sum_{pairs} fac X X^dag/(e1 - e2 - z)

at z = x - i y, in the lower half plane; its poles lie on the real axis at
the transition energies, the spin flips at positive frequency. Measured
against a direct sum on the magnet of tests/chi/test_response_winding.py:
the two agree to 1e-17, and the call differs from chi0(x + iy) by 3e-3.
Since fac is real, chi0(conj z) = chi0(z)^dag, so the upper half plane is
the Hermitian conjugate of the call at delta = Im z, and a frequency and
its mirror image share one evaluation. One call takes every real part that
shares the same imaginary part, which is why the contour of greenwinding
is a rectangle.

The temperature. T is the one of the occupations of localspinchi, whose
Fermi function divides by T, so it is positive here, 1e-3 by default,
while the kernels of route 2 (chitk/kernelchern.py) default to T=0, sharp
occupations. It has to be small against the gap: the degenerate pairs of
spin_chi carry a term i delta f'(e), which is not analytic in z and
vanishes only where f' does, in a gapped state with T well below the gap.

The operators, channel=. They are those of route 2, one of
chitk.kernelchern.get_channel_names(): "full", (Sx,Sy,Sz)/2 on every
site; "raising", S^+ = Sx + i Sy, the operator that creates the magnon of
a magnet polarized along -z (add_zeeman([0,0,B]) with B > 0); and
"lowering", S^-, for a magnet polarized along +z. The block of a channel
is built as LocalSpinKernel builds it: with Tc the matrix of the channel
and T = Tc (x) 1_N in the layout a*N + i, chi0 becomes T chi0 T^dag and
the vertex T^-dag V T^-1, and the leading block is kept, the N operators
of S^+ or of S^-. When the rest couples to that block, in chi0 or in V,
which happens when Sz is not conserved, the channel is not a block of the
response and a ValueError names channel="full".

The two spin sectors. With a conserved Sz the transverse response is the
sum of two sectors that do not mix, S^+ and S^-. A saturated magnet has
modes in one of them only. A magnet with both spin species occupied, a
ferrimagnet or an antiferromagnet, has bands in both, the excitation
branch of the TDHF in one and the mirror of its de-excitation branch in
the other, and they overlap in energy, so a window that isolates a band
of one sector can have its end inside a band of the other. The channel of
the sector is then what isolates it. Measured on the Lieb ferrimagnet of
tests/magnon/test_magnon_chern.py (U=4, Zeeman 0.3, nk=6): the raising
channel is the sector of the excitation branch, with bands at 0.487 to
0.908 and 1.099 to 1.270 on the shifted mesh with nq=6, and with the
windows (0.2437, 1.0037) and (1.0037, 1.64) it gives +0.9992 and -0.9993
at nq=8, nz=8, the +1 and -1 of the TDHF eigenvectors and of route 2 in
the same channel. The lowering channel has one band, 0.324 to 1.185,
across the end 1.0037. The full channel there gives 1.0021 and -0.9964 at
nz=4, close only because that band has a trivial bundle (a winding of
2e-6 measured by the review on a window around it), and this is not a
control: its window end lies inside a band.

The kernels (kernel=), one of get_kernel_names():
- "dyson", the default: 1 - chi0 V, which needs no inverse of chi0, so it
  takes a magnet along any direction in any channel (a saturated magnet
  along z has an exactly zero zz block of chi0, and one along x a zero xx
  block), and has no pole inside a window below the continuum, so its
  winding counts the modes and not the zeros of chi0. It is not Hermitian
  on the real axis, so both halves of the contour are evaluated, from one
  bubble per frequency of the upper half.
- "response": K = chi0^-1 - V, the inverse of the dressed response, which
  needs chi0 invertible on the contour, and raises otherwise.
- "bare": chi0^-1 alone. Its winding over the window is minus the Chern
  number of the zeros of chi0 inside it, the term by which the two above
  differ, since K = chi0^-1 (1 - chi0 V) and the winding is additive over
  products. This is the trap of reading the response as a Green's
  function: a zero of chi0 in the window is a pole of K and adds to its
  winding; "bare" measures it.

The bundle that is wound, and its sign. The residue of chi at a mode of
positive frequency is a X X^dag, with X_A = <m,k|A|n,k+q> the matrix
element of an operator between an empty state at k and an occupied state
at k+q. So in pyqula's layout chi(q) describes the modes of momentum -q,
and its residue vector is <mode|A_q|0>, the complex conjugate of the
amplitude <0|A_q^dag|mode> that is linear in the mode. The winding of K is
the Chern number of that conjugate amplitude, which is minus the Chern
number of the mode itself; the inversion q -> -q does not change a Chern
number in two dimensions. So the winding of K is minus the Chern number of
the magnon band as the TDHF eigenvectors give it (bsetk/spinflip.py, lifted
to the orbital pair basis), and spin_response_chern returns minus the
winding, the Chern number of the mode, the same thing as winding the
transpose of K. This is measured, not only derived, on the magnet of
tests/chi/test_response_winding.py: the null vector of K at -Q and at the
mode energy has an overlap 1 - 2e-13 (an angle of about 6e-7) with the
complex conjugate of the TDHF amplitude on the local spin flips at Q, and
1e-16 with that amplitude itself; at +Q the overlap with the conjugate is
only 0.98. The mode energies of the two agree to 7e-6 (4.6438375 against
4.6438308 at Q=(0.13,0.27)), and this difference is the k-mesh: the bubble
at -Q pairs an empty state at k with an occupied one at k-Q, with k on
the mesh, while the TDHF pairs an occupied state at k with an empty one
at k+Q, so the occupied states of the two are sampled on meshes that
differ by a shift of Q. For a Q of the mesh the two sums are the same
and the energies agree to 1e-14 (Q=(1/8,2/8) at nk=8), and for
Q=(0.13,0.27) the difference falls to 7e-9 at nk=12 and 2e-11 at nk=16
(the bubble and the TDHF at that nk, on the mean field of nk=8).
The plaquette Chern number of those null vectors is -1 for the lower band
and +1 for the upper one, that of the TDHF amplitudes +1 and -1, and the
winding of K agrees with the null vectors, sign included.


The measured numbers on that magnet (honeycomb, tij=[1.0,0.3], Kane-Mele
0.15, Zeeman 3, U=8 ferro at half filling, nk=8), with kernel="response"
on the transverse components (Sx,Sy), which for this saturated magnet hold
the raising channel and nothing else at positive frequency: the two magnon
bands are 4.547 to 4.832 and 5.096 to 5.946 and the continuum starts at
7.43. With the window [4.0, 4.964], spin_response_chern gives 1.16,
1.0036, 0.987, 1.0031, 0.99956 and 0.99994 at nq=6, 8, 10, 12, 16 and 20
(nz=6 to 8, converged in nz to 1e-7 at nz=8), the error falling
exponentially once nq resolves the band, and -3e-7 with the window
[4.0, 6.687] around both bands. A mean field at nk=12 gives 1.0036 at
nq=8, and add_kane_mele(-0.15) gives -1.0036. kernel="dyson" gives
1.0036 and kernel="bare" 4e-16, so the zeros of chi0 inside the window,
if there are any, carry no net Chern number. That there is none is seen
directly: the transverse block of chi0 is negative definite at 28 real
frequencies across the two windows on the 8 x 8 mesh of q, with its
largest eigenvalue at -0.024. The approach to the integer is not monotonic on
coarse meshes: at nz=4 the lower window gives 1.166, 0.927, 1.0024, 1.023,
0.986, 1.0012, 1.0030 and 0.9971 at nq=6 to 13, and 0.9951 at nq=8 with
shift=(0.03,0.07), so one mesh below nq of about 12 says which integer it
is and not how converged it is. The default, kernel="dyson" in the full
channel, gives 1.0036 at nq=8, nz=8 in 10.7 s, the time of
kernel="response" in the raising channel on the upper half of the contour
alone (10.7 s); before the two halves shared their bubble it took 13.0 s."""

import numpy as np

from . import localspinchi


def bare_spin_chi_complex(h, q, zs, nk=8, T=1e-3):
    """Return chi0(q,z) of the spin operators (Sx,Sy,Sz)/2 of every site at
    the complex frequencies zs, one 3N x 3N matrix per frequency, in the
    layout of localspinchi.bare_spin_chi (component a of site i at index
    a*N + i), as the analytic continuation of the bubble described in the
    module docstring. zs may lie in either half plane, and each pair
    (Re z, |Im z|) is evaluated once, since chi0(conj z) = chi0(z)^dag. A
    frequency on the real axis must avoid the energy difference of every
    pair of states that contributes, where chi0 has a pole and a result
    that is not finite raises; the pairs of two states with the same
    occupation, which do not contribute, are left out by localspinchi.

    h: a spinful Hamiltonian without Nambu, usually a mean field.
    q: the momentum in units of the reciprocal lattice vectors.
    nk: the k-mesh of the bubble, which should be the one the mean field
        was converged on.
    T: the temperature of the occupations, positive and small against the
        gap (module docstring)."""
    localspinchi._require_normal_spinful(h)
    if q is None:
        raise ValueError("the kernel of a collective mode is a quantity of "
                "one momentum, q=None was given; pass q in units of the "
                "reciprocal lattice vectors")
    zs = np.atleast_1d(np.asarray(zs, dtype=np.complex128))
    if not T > 0.:
        raise ValueError("T is the temperature of the occupations and must "
                "be positive, got "+str(T))
    n = len(h.geometry.r)
    out = np.zeros((len(zs), 3*n, 3*n), dtype=np.complex128)
    ys = np.abs(zs.imag)
    for y in np.unique(ys): # one call per imaginary part
        sel = np.where(ys == y)[0]
        # a frequency and its mirror image share one evaluation
        xs, inverse = np.unique(zs[sel].real, return_inverse=True)
        es, chi = localspinchi.bare_spin_chi(h, q=q, energies=xs, nk=nk,
                                             delta=y, T=T)
        for (i, c) in zip(sel, chi[np.ravel(inverse)]):
            # c is chi0 at Re z - i |Im z|
            out[i] = np.conj(c.T) if zs[i].imag > 0. else c
    finite = np.all(np.isfinite(out), axis=(1, 2))
    if not np.all(finite):
        raise ValueError("chi0 is not finite at z=%s, q=%s: a frequency on "
                "the real axis coincides with the energy difference of a "
                "pair of states of the k-mesh; move the frequency or give "
                "it an imaginary part" % (zs[int(np.argmin(finite))], q))
    return out


def _kernel_response(chi0, V):
    return np.linalg.inv(chi0) - V


def _kernel_dyson(chi0, V):
    return np.identity(V.shape[0], dtype=np.complex128) - chi0@V


def _kernel_bare(chi0, V):
    return np.linalg.inv(chi0)


# name -> (kernel of chi0 and V, hermitian on the real axis, inverts chi0)
_KERNELS = {"response": (_kernel_response, True, True),
            "dyson": (_kernel_dyson, False, False),
            "bare": (_kernel_bare, True, True)}


def get_kernel_names():
    """The accepted values of kernel="""
    return sorted(_KERNELS)


def _check_invertible(chi, zs, q, channel, tol=1e-10):
    s = np.linalg.svd(chi, compute_uv=False) # (nz, D), decreasing
    ratio = s[:, -1]/np.maximum(s[:, 0], 1e-300)
    i = int(np.argmin(ratio))
    if ratio[i] < tol:
        raise ValueError("chi0 is singular at z=%s, q=%s (smallest over "
            "largest singular value %.1e) in the channel %r: one of its "
            "operators couples to no transition, as Sz in a magnet "
            "collinear along z. Use kernel='dyson', which does not invert "
            "chi0, or the channel of the magnet, 'raising' or 'lowering'"
            % (zs[i], q, ratio[i], channel))


def _block(m, D, channel, what, q, tol):
    """The leading D x D block of the matrices m (..., 3N, 3N), refused
    when the rest couples to it"""
    if D == m.shape[-1]: return m
    off = max(np.max(np.abs(m[..., :D, D:])), np.max(np.abs(m[..., D:, :D])))
    if off > tol*max(np.max(np.abs(m)), 1e-300):
        raise ValueError("%s couples the channel %r to the other spin "
                "components at q=%s (largest element %.2e), so the channel "
                "is not a block of the response; use channel='full'"
                % (what, channel, q, off))
    return m[..., :D, :D]


def spin_kernel(h, V=None, nk=8, T=1e-3, channel="full", kernel="dyson"):
    """Return (ginv, hermitian): the kernel of the site-basis spin RPA as a
    callable ginv(zs, q) -> (len(zs), D, D) array at the complex
    frequencies zs and the reduced momentum q, in the form
    topologytk.greenwinding takes with batched=True, and whether it obeys
    ginv(conj z) = ginv(z)^dag.

    h: the mean-field Hamiltonian, spinful, with its interaction h.V.
    V: the real-space vertex {(n1,n2,n3): 3N x 3N matrix} in the layout of
        localspinchi.spin_chi, by default chitk.spinchi._full_spin_U(h),
        which takes an onsite interaction only.
    nk, T: those of bare_spin_chi_complex.
    channel: the spin operators, one of
        chitk.kernelchern.get_channel_names(), see the module docstring; D
        is 3N for "full" and N for "raising" and "lowering".
    kernel: one of get_kernel_names(), see the module docstring."""
    from .rpa import interaction_at_q
    from .kernelchern import _channel
    localspinchi._require_normal_spinful(h)
    if kernel not in _KERNELS:
        raise ValueError("unknown kernel "+repr(kernel)+"; the accepted "
                "ones are "+", ".join(get_kernel_names()))
    f, hermitian, inverts = _KERNELS[kernel]
    Tc, nc = _channel(channel)
    if V is None:
        from .spinchi import _full_spin_U
        V = _full_spin_U(h)
        if V is None:
            raise ValueError("the Hamiltonian has no mean-field interaction "
                    "(h.V) to build the vertex from; pass V or use the "
                    "output of get_mean_field_hamiltonian")
    n = len(h.geometry.r)
    Tm = np.kron(Tc, np.identity(n)) # T (x) 1_N in the layout a*N + i
    Ti = np.linalg.inv(Tm)
    D = nc*n
    def ginv(zs, q):
        zs = np.atleast_1d(np.asarray(zs, dtype=np.complex128))
        chi = bare_spin_chi_complex(h, q, zs, nk=nk, T=T)
        Vq = np.asarray(interaction_at_q(V, h, q), dtype=np.complex128)
        if D < 3*n: # T chi0 T^dag and T^-dag V T^-1, the leading block
            chi = _block(Tm@chi@Tm.conj().T, D, channel, "chi0", q, 1e-8)
            Vq = _block(Ti.conj().T@Vq@Ti, D, channel, "the vertex", q,
                        1e-10)
        if inverts: _check_invertible(chi, zs, q, channel)
        return np.array([f(c, Vq) for c in chi])
    return ginv, hermitian


def spin_response_chern(h, window, nq=8, nz=8, nk=8, T=1e-3, V=None,
                        channel="full", kernel="dyson", height=None,
                        derivative="spectral", shift=(0., 0.)):
    """Sum of the Chern numbers of the collective modes of the site-basis
    spin RPA inside the window [w_a, w_b] of real frequencies, from the
    winding of its kernel (module docstring), as a float that converges to
    an integer as nq and nz grow. The sign is the one of the mode bundle,
    so minus the winding of K in pyqula's layout of chi, see the module
    docstring.

    window: both ends in gaps of the mode spectrum of the channel at every
        q, and below the continuum; with kernel="response" also with no
        zero of chi0 in between.
    nq, nz, height, derivative, shift: those of
        greenwinding.window_chern_density; the cost is about nq^2 (nz+1)
        calls of the bubble on the nk x nk mesh, the same for the two
        halves of the contour of kernel="dyson".
    nk, T, V, channel, kernel: those of spin_kernel."""
    from ..topologytk import greenwinding, plaquette
    plaquette.require_2d(h, "the collective modes of the spin RPA")
    ginv, hermitian = spin_kernel(h, V=V, nk=nk, T=T, channel=channel,
                                  kernel=kernel)
    w = greenwinding.window_chern(ginv, window, nq=nq, nz=nz, height=height,
                                  derivative=derivative, hermitian=hermitian,
                                  batched=True, shift=shift)
    return -w
