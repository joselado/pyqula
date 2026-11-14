"""Chern number of a collective mode from the null vectors of the kernel of
a response function.

The question is what plays the role of the Bloch state of a collective mode
(a magnon of the random phase approximation, the spin exciton of the
moments of a Kondo lattice) when there is no eigenvalue problem behind it,
only a response function between local operators. The answer is the
residue of the pole. In the random phase approximation the response is

    chi(q,w) = chi0(q,w) (1 - V(q) chi0(q,w))^-1 = K(q,w)^-1 ,
    K(q,w) = chi0(q,w)^-1 - V(q) ,

so a pole of chi is a zero of K. Near a mode of energy w_n(q) the kernel
has one eigenvalue that goes as c (w - w_n) with c > 0, and chi goes as
v v^dag / (c (w - w_n)), with v the null vector of K at the pole. That
vector lives in the space of the local operators, which does not depend on
q, so the plaquette sum of topologytk.plaquette takes it with no lift.

The sign convention of the bare response is the one of chitk.chiAB: each
pair of states a and b contributes (f_a - f_b) <a|A_i|b><b|A_j^dag|a> /
(e_a - e_b - w + i delta). Below the continuum the broadening is not
needed, and the bubble here is evaluated with delta = 0, as

    chi0_ij(q,w) = 1/Nk sum_k sum_ab (f_a - f_b) X_i X_j^* / (e_a - e_b - w) ,
    X_i = <a,k+q| O_i |b,k> ,

with a a state at k+q and b a state at k, k on the mesh. This is
chitk.chiAB.chiAB_q at the momentum -q on the mesh shifted by q, and it is
the pair set of bsetk.pairbasis.PairBasis at the momentum q (an electron at
k+q and a hole at k), so for an onsite U the zeros of K are exactly the
energies of time-dependent Hartree-Fock (bsetk.spinflip.magnon_matrix) at
every q, and not only on the points of the mesh. Every term of chi0 is
negative between the highest pole at negative frequency and the lowest one
at positive frequency, since there f_a - f_b and e_a - e_b - w have
opposite signs. So below the continuum chi0 is Hermitian and negative
definite on the operators that couple to some transition, and K is a
smooth Hermitian matrix there.

So the modes below a frequency are counted with the inertia of K: an
eigenvalue of a Hermitian matrix that depends continuously on w can change
sign only through zero, and at a mode of positive frequency and positive
weight it goes from negative to positive. The number of modes in a window
[w_a,w_b] is then the number of negative eigenvalues of K at w_a minus the
number at w_b (Sylvester's law of inertia), which gives a bisection for
every mode that needs no frequency grid and does not miss a mode close to
another one. When all the transitions have the same sign of f_a - f_b, as
in the raising channel of a saturated ferromagnet, d chi0/dw is negative
definite, so dK/dw = -chi0^-1 (d chi0/dw) chi0^-1 is positive definite and
every eigenvalue of K increases with w. With both signs, which is the case
of an antiferromagnet, of a Kondo lattice and of the full spin response,
dK/dw is indefinite, and what still holds is that a zero at positive
frequency of a stable mean field is crossed upwards, since its residue is
the spectral weight of the mode (derived, not proven here for every
case). The routines below check what they rely on instead, at every
frequency the search visits: the count must not increase inside the
window, and a root of an eigenvalue must be a zero of it, not a jump
through infinity, which is what a pole of K inside the window would give
(a level of a bath, or a zero of det chi0). A pole of K and a zero that
fall between two visited frequencies cancel in the count and are not
seen, which is why the kernels of the adaptors carry their continuum.

What the null vector is, and so the sign of the Chern number. The residue
of chi_ij is <n|A_i|0><0|A_j^dag|n>, so v_i = <n|A_i|0>, with
A_i = sum_k c^dag_{k+q} O_i c_k the operator that creates the mode at
momentum q. For a mode of time-dependent Hartree-Fock,
|n,q> = sum_k sum_cv X_cv(k) c^dag_{c,k+q} c_{v,k}|0>, this is

    v_i = conj( sum_k sum_cv X_cv(k) conj(<c,k+q|O_i|v,k>) ) ,

the complex conjugate of the site-diagonal part of the two-body
wavefunction Phi_q(k;alpha,beta) = sum_cv X_cv(k) u_{c,k+q}(alpha)
u*_{v,k}(beta) that future_development/topological_invariants.md uses for
route 1 (for O_i = S^+_i, the element alpha = (i,up), beta = (i,down)
summed over k). The kernels built here are therefore returned as the
complex conjugate of K, whose null vector is

    psi_i(q) = conj(v_i(q)) = <0| A_i(q)^dag |n,q> ,

the amplitude of the mode on the local operators, in the same way as the
amplitude u_i(k) = <0|c_{k,i}|n,k> of a Bloch state. With it the Berry
connection is A = i<psi|grad psi> and the sign is the one of h.get_chern()
and of the eigenvectors of route 1; the raw null vector of chi0^-1 - V
gives the opposite integer. tests/chi/test_kernel_chern.py checks the
overlap of psi with the site-diagonal part of the eigenvectors of
magnon_matrix and the integer of both. When the Casida matrix has a
de-excitation half the mode is (X,Y), and psi is the site-diagonal part
of the sum of the two lifts, X on the pairs (empty at k+q, occupied at k)
and Y on the pairs (occupied at k+q, empty at k). Measured on a honeycomb
antiferromagnet in a staggered field, in the lowering channel, with |Y|
up to 0.27 |X|: the zero is the Casida energy to 1e-9 and the overlap is
1 to 1e-12, against 0.87 with the opposite sign of Y and 0.97 with X
alone; its magnon band has Chern number 0 in both routes.

The operators of a spin response, channel= of the adaptors, one of
get_channel_names():

- "raising": S^+_i = c^dag_{i,up} c_{i,down} on every site, the N x N block
  in which S^+ creates the mode. It needs a mean field that conserves Sz,
  and it holds the magnons of a magnet polarized along -z, which is what
  add_zeeman([0,0,B]) with B > 0 gives (get_vev("sz") negative), and one
  copy of every spin triplet of a state with spin rotation symmetry.
- "lowering": S^-_i, the block in which S^- creates the mode, for a magnet
  polarized along +z.
- "full": (Sx,Sy,Sz)/2 on every site, the 3N x 3N response, for any mean
  field. In a collinear state each mode appears once there too, in its
  own block. When some of the operators couple to no transition at all
  (Sz in a saturated ferromagnet, where the opposite circular component
  still couples, at negative frequency), chi0 is singular, and the kernel
  is written in the span of the operators that do couple; the null vector
  is returned in the full space of the operators. The same happens in the
  raising channel of a spin singlet at q=0, whose uniform S^+ annihilates
  the ground state.

In the raising block the vertex is the transverse part of the full one,
T^-dag V T^-1 with S^+ = Sx + i Sy, which for an onsite U is the -U of
chitk.spinchi.spinchi_ladder, and the RPA is the 1/(1/chi0 - U') of
kondotk.momentchi for the moments of a Kondo lattice.

Three limits. Route 2 works only below the continuum, and the window must
lie between zero and the lowest pole of chi0 at every q, which the
adaptors check. On a finite k-mesh that pole is one discrete transition,
chi0 diverges at it, and any attractive vertex binds a spurious zero just
below it, which moves into the continuum as nk grows (measured on a
chiral spinon state at q=(1/8,0): 0.38208 against a pole at 0.38214 with
nk=12, and no zero at nk=24). So w_b has to stay well below the
continuum, and a mode is a mode only when its energy is stable in nk.
With one moment per cell and a raising or lowering
channel the kernel is a number and the bundle of its null vector is
trivial, so its Chern number is zero by construction. And the residual
interaction U'(q) of a Kondo lattice is a parameter of the comparison
with matrix product states, not a result, so an invariant there is a
property of the model with that U'.

The mesh of q. The plaquette sum returns an integer on any mesh, and it
is the Chern number only when the Berry flux of every plaquette is below
pi in modulus. For a group of n modes the flux of a plaquette is the sum
of the fluxes of the n modes, and when they share one null vector, as
the three components of a spin triplet do, it is n times the one of a
single mode, so the mesh has to be finer. Measured on the spin
exciton of a honeycomb Kondo insulator (two moments per cell, a Haldane
term of 0.2 on the conduction electrons, U' = -4): the raising channel
gives 1 from nq=6 on, and the full channel, where the exciton is a
triplet with the same null vector in the three spin components, gives 0,
2, 1, 3, 3 at nq = 6, 8, 10, 12, 16, with no error raised. So an integer
is read only once it is stable in nq, and a spin triplet is cheaper in
the raising channel, which holds one copy of it; the largest flux of a
plaquette, max_flux of the record of kernel_chern, is the thing to watch.

A touching. A group degenerate with another zero at a point of the mesh
raises. With no Haldane term the two spin excitons of that Kondo lattice
touch at K and K' (1.53057, nk=8 and 10), and an unshifted mesh with nq=6
or 12, which holds K, refuses the lowest one, while nq=10 misses K and
returns an integer with no error, with a gap of 0.14 on its record: a
touching between two points of the mesh is not seen, not even by the
gap of the record, so a second nq is the check. This is why the default
is nq=12 (topologytk/plaquette.py).

The temperature. The kernels default to T=0, sharp occupations, as the
pair basis of route 1 has, so there are no thermal pairs. On the gapped
magnet of the tests the zeros at T=0 and at T=1e-3 are the same to the
last digit, and the TDHF energies to 8e-14 (nk=6 and 8)."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from .. import algebra
from .. import check
from .chiAB import DEGENERATE_TOL
from ..topologytk import plaquette


# ------------------------------------------------------------------------
# The generic engine: a Hermitian kernel K(q,w) and its zeros in a window
# ------------------------------------------------------------------------


def _check_window(window):
    w = np.array(window, dtype=np.float64).reshape(-1)
    if w.shape[0] != 2 or not w[0] < w[1]:
        raise ValueError("window is the frequency interval (w_a, w_b) in "
                "which the modes are searched, two numbers with w_a < w_b, "
                "got "+str(window))
    return float(w[0]), float(w[1])


def _as_q(q):
    q = np.array(q, dtype=np.float64).reshape(-1)
    if q.shape[0] < 3: q = np.concatenate([q, np.zeros(3 - q.shape[0])])
    return q


class _Eigen():
    """The kernel at one q, evaluated on demand and checked to be
    Hermitian"""
    def __init__(self, kernel, q, herm_tol=1e-8):
        self.kernel, self.q, self.herm_tol = kernel, q, herm_tol
        self.norms = [] # largest |eigenvalue| at the frequencies counted
    def eigh(self, w):
        m = np.array(self.kernel(self.q, w), dtype=np.complex128)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("the kernel must return a square matrix, got "
                    "shape "+str(m.shape)+" at q="+str(self.q)+", w="+str(w))
        if not np.all(np.isfinite(m)):
            raise ValueError("the kernel is not finite at q=%s, w=%.10g: it "
                "has a pole there (a pole of the bare response, a level of "
                "a bath), so the window is not inside the region where it "
                "is smooth. Move the window" % (self.q, w))
        scale = max(1., np.max(np.abs(m)))
        anti = np.max(np.abs(m - m.conj().T))
        if anti > self.herm_tol*scale:
            raise ValueError("the kernel is not Hermitian at q=%s, w=%.6g "
                "(largest element of K - K^dag %.2e); its zeros are the "
                "modes only at real frequency below the continuum, where "
                "the broadening is not needed" % (self.q, w, anti))
        return np.linalg.eigh((m + m.conj().T)/2.)
    def count(self, w):
        """Number of negative eigenvalues of K(q,w)"""
        ev = self.eigh(w)[0]
        self.norms.append(np.max(np.abs(ev)))
        return int(np.sum(ev < 0.))
    def scale(self):
        """The size of K away from its poles: the smallest, over the
        frequencies at which the eigenvalues were counted, of its largest
        eigenvalue in modulus, and at least 1. The largest eigenvalue at a
        root is not a scale, since at a pole of K it diverges and any
        eigenvalue is small next to it"""
        return max(1., min(self.norms))


def _non_increasing(n1, w1, n2, w2, q):
    if n2 > n1:
        raise ValueError("the number of negative eigenvalues of the kernel "
            "increases from %d at w=%.6g to %d at w=%.6g (q=%s). Inside a "
            "window below the continuum it can only decrease, one unit at "
            "each mode, so the kernel has a pole there (a pole of the bare "
            "response, a level of a bath, a zero of det chi0) or a mode of "
            "negative weight, the mirror of a mode at negative frequency or "
            "an instability of the mean field. Move the window, or keep it "
            "at positive frequency" % (n1, w1, n2, w2, q))


def _root(E, wl, wr, index, xtol):
    """The zero of the eigenvalue number index (in increasing order) of K,
    which is negative at wl and not negative at wr"""
    f = lambda w: E.eigh(w)[0][index]
    w0 = brentq(f, wl, wr, xtol=xtol, rtol=4*np.finfo(float).eps)
    return w0


def _poles(E, wa, wb, xtol, degenerate_tol, nscan):
    """The zeros of K in [wa,wb], as a list of (energy, first index,
    multiplicity): the null vectors are the eigenvectors of K(energy) with
    indices first to first+multiplicity-1"""
    ws = np.linspace(wa, wb, nscan + 1)
    ns = [E.count(w) for w in ws]
    for i in range(nscan): _non_increasing(ns[i], ws[i], ns[i+1], ws[i+1], E.q)
    out = []
    stack = [(ws[i], ns[i], ws[i+1], ns[i+1]) for i in range(nscan)]
    while len(stack) > 0:
        wl, nl, wr, nr = stack.pop()
        if nl == nr: continue
        if nl - nr == 1: # one mode inside, refined on its eigenvalue
            out.append((_root(E, wl, wr, nl - 1, xtol), nr, 1))
        elif wr - wl < degenerate_tol: # a group of degenerate modes
            out.append(((wl + wr)/2., nr, nl - nr))
        else:
            wm = (wl + wr)/2.
            nm = E.count(wm)
            _non_increasing(nl, wl, nm, wm, E.q)
            _non_increasing(nm, wm, nr, wr, E.q)
            stack += [(wl, nl, wm, nm), (wm, nm, wr, nr)]
    out.sort(key=lambda p: p[0])
    return out


def _check_continuum(continuum, q, wa, wb):
    if continuum is None: return
    lo, hi = continuum(q)
    if not (lo < wa and wb < hi):
        raise ValueError("the window (%.6g, %.6g) is not inside the gap "
            "(%.6g, %.6g) of the continuum at q=%s, where the bare "
            "response has poles. The kernel describes the modes only "
            "below the continuum, so lower w_b below the lowest "
            "transition at every q (or raise w_a above zero and above "
            "the highest transition at negative frequency). With T > 0 in "
            "the kernel of a spin response a thermal pair, two states of "
            "nearly the same occupation, can lower the edge, and a lower T "
            "is then the remedy" % (wa, wb, lo, hi, q))


def kernel_modes(kernel, q, window, continuum=None, basis=None, xtol=1e-12,
                 degenerate_tol=1e-9, nscan=16, null_tol=1e-6):
    """Return (energies, vectors): the zeros of the Hermitian kernel K(q,w)
    in the frequency window, in increasing order, and their null vectors as
    the rows of an (n, D) array. A group of degenerate zeros gives one row
    per null vector, with the same energy.

    kernel: callable kernel(q, w) returning K(q,w), a Hermitian matrix at
        real w, whose eigenvalues cross zero upwards at the modes (see the
        module docstring). If it also has the methods continuum(q) and
        basis(q), as the kernels of spin_rpa_kernel and moment_rpa_kernel
        do, they are used when continuum and basis are not given.
    q: momentum, in units of the reciprocal lattice vectors.
    window: (w_a, w_b), the frequencies between which the modes are
        searched. Each end must be a frequency where K has no zero, and
        the whole window must be inside the gap of the continuum.
    continuum: callable q -> (lower, upper), the interval in which K is
        smooth; a ValueError is raised when the window is not inside it.
    basis: callable q -> (D, r) matrix whose columns span the space in
        which K(q,w) (r x r) is written; the null vectors are returned as
        basis(q) @ u, in the space of dimension D. None: the null vectors
        are those of K itself.
    xtol: tolerance of the energy of a zero.
    degenerate_tol: two zeros closer than this are a degenerate group.
    nscan: number of intervals of the first scan of the window, at whose
        ends the count of negative eigenvalues is checked not to increase.
    null_tol: largest eigenvalue accepted at a zero, relative to the size
        of K away from its poles (the smallest, over the frequencies of
        the scan, of its largest eigenvalue in modulus, and at least 1);
        above it the root is a jump through infinity, a pole of K, and a
        ValueError is raised."""
    wa, wb = _check_window(window)
    q = _as_q(q)
    if continuum is None: continuum = getattr(kernel, "continuum", None)
    if basis is None: basis = getattr(kernel, "basis", None)
    _check_continuum(continuum, q, wa, wb)
    E = _Eigen(kernel, q)
    poles = _poles(E, wa, wb, xtol, degenerate_tol, nscan)
    es, vs = [], []
    for (w0, first, mult) in poles:
        ev, evec = E.eigh(w0)
        lam = ev[first:first+mult]
        if np.max(np.abs(lam)) > null_tol*E.scale():
            raise ValueError("the kernel changes sign at w=%.10g (q=%s) "
                "through infinity, not through zero (eigenvalue %.3e "
                "there): it has a pole inside the window, a level of a "
                "bath or a zero of det chi0, so its count of negative "
                "eigenvalues does not count modes there. Move the window"
                % (w0, q, np.max(np.abs(lam))))
        for i in range(first, first + mult):
            es.append(w0)
            vs.append(evec[:, i])
    if len(vs) == 0: return np.zeros(0), np.zeros((0, 0), dtype=np.complex128)
    vs = np.array(vs)
    if basis is not None:
        B = basis(q)
        if B is not None: vs = (np.asarray(B)@vs.T).T
    return np.array(es), vs


def _groups(es, degenerate_tol):
    """Index of the degenerate group of every zero, in increasing order"""
    out, g = [], 0
    for i in range(len(es)):
        if i > 0 and es[i] - es[i-1] >= degenerate_tol: g += 1
        out.append(g)
    return out


def kernel_chern(kernel, window, bands=None, nq=plaquette.NQ,
                 shift=(0., 0.), tol=1e-6, full_output=False, continuum=None,
                 basis=None, degenerate_tol=1e-9, **kwargs):
    """Chern number of a group of zeros of a Hermitian kernel K(q,w), from
    the plaquette product of its null vectors on the nq x nq mesh of the
    reduced Brillouin zone of topologytk.plaquette.

    kernel, window, continuum and basis are those of kernel_modes, and the
    remaining keyword arguments (xtol, nscan, null_tol) are passed to it.
    bands: indices of the zeros inside the window, counted from the lowest
        one, whose bundle is computed, a contiguous group as in route 1
        (bsetk.topology.mode_chern): [0] for the lowest band and [0,1] for
        the two lowest together. None, the default, takes every zero in
        the window. A group of degenerate zeros has to be taken whole.
    nq, shift: the mesh, topologytk.plaquette.mesh. The default nq=12 holds
        the points of a hexagonal or square zone where a symmetry can make
        two modes touch, and a touching on a point of the mesh is refused.
    tol: the smallest link overlap accepted, below which the mesh does not
        resolve the bundle.
    full_output: True returns a topologytk.plaquette.ModeChern record: the
        fluxes and the largest one, the zeros of the group at every point
        of the mesh as energies (nq,nq,n), gap, the smallest distance of
        the group to another zero of the window or to an end of the window,
        and bound, the smallest distance of the top of the group below the
        upper edge of the continuum, None for a kernel with no continuum.

    The kernel is solved once per point of the mesh. A group degenerate
    with another zero at a point of the mesh raises, and so does a number
    of zeros in the window that changes over the zone, since a band has a
    Chern number only when it is separated from the others and from the
    continuum over the whole zone. A touching between two points of the
    mesh is not seen; the gap of the record and a second nq are the check.
    Returns a float that is an integer up to roundoff, on any mesh: it is
    the Chern number only once it is stable in nq, and a group of
    degenerate modes needs a finer mesh than a single one (see "The mesh of
    q" in the module docstring). With the kernels of spin_rpa_kernel and
    moment_rpa_kernel the sign is the one of h.get_chern() for the
    amplitude of the mode on the local operators (see the module
    docstring)."""
    from .. import parallel
    wa, wb = _check_window(window)
    if continuum is None: continuum = getattr(kernel, "continuum", None)
    qs = plaquette.mesh(nq, shift)
    nq = qs.shape[0]
    flat = list(qs.reshape(-1, 3))
    def at(q): # the zeros, their null vectors and the edge, at one q
        es, vs = kernel_modes(kernel, q, window, continuum=continuum,
                basis=basis, degenerate_tol=degenerate_tol, **kwargs)
        # the kernels of this module keep the last q, so the edge is read
        # here and not after the whole mesh is solved
        edge = None if continuum is None else continuum(q)[1]
        return es, vs, edge
    sols = parallel.pcall(at, flat)
    n0 = len(sols[0][0])
    if n0 == 0:
        raise ValueError("the kernel has no zero in the window %s at q=%s, "
                "so there is no band to compute" % (window, flat[0]))
    if bands is None: bands = list(range(n0))
    bands = plaquette.check_bands(bands)
    if bands[-1] >= n0:
        raise ValueError("bands are indices of the %d zeros of the kernel "
                "in the window (0 to %d) at q=%s, got %s" % (n0, n0 - 1,
                flat[0], bands))
    # a group degenerate with another zero is looked for first, at every
    # point where the group exists, since it is the more specific reason:
    # a touching usually also changes the number of zeros in the window
    for q, (es, vs, edge) in zip(flat, sols):
        if len(es) <= bands[-1]: continue
        g = _groups(es, degenerate_tol)
        inside = set(g[i] for i in bands)
        for m in range(len(es)):
            if (g[m] in inside) != (m in bands):
                raise ValueError("the zero %d is degenerate with a zero of "
                    "the group %s at q=%s (energy %.10g), so the group is "
                    "not separated from the other modes there and has no "
                    "Chern number of its own; take the degenerate zeros "
                    "together, or move the mesh (shift=) if the touching "
                    "is only at that point" % (m, bands, q, es[m]))
    for q, (es, vs, edge) in zip(flat, sols):
        if len(es) != n0:
            raise ValueError("the number of zeros of the kernel in the "
                "window %s changes over the zone: %d at q=%s but %d at "
                "q=%s. A band has a Chern number only when it stays "
                "inside the window and away from its ends at every q, so "
                "choose ends that lie in gaps of the mode spectrum over "
                "the whole zone" % (window, n0, flat[0], len(es), q))
    V = [[sols[i*nq + j][1][bands] for j in range(nq)] for i in range(nq)]
    fluxes, worst = plaquette.mesh_berry_fluxes(V)
    plaquette.check_overlap(worst, tol, nq, what="groups of zeros")
    energies, gap, bound = [], np.inf, None
    for (es, vs, edge) in sols:
        eg = es[bands]
        below = eg[0] - (es[bands[0]-1] if bands[0] > 0 else wa)
        above = (es[bands[-1]+1] if bands[-1]+1 < n0 else wb) - eg[-1]
        gap = min(gap, below, above)
        if edge is not None:
            bound = edge - eg[-1] if bound is None else min(bound,
                                                            edge - eg[-1])
        energies.append(eg)
    out = plaquette.ModeChern(fluxes, worst, qs, energies, gap, bound)
    if full_output: return out
    return out.chern


# ------------------------------------------------------------------------
# The kernel of the response of local spin operators
# ------------------------------------------------------------------------

_SIGMA = [np.array([[0., 1.], [1., 0.]], dtype=np.complex128),
          np.array([[0., -1j], [1j, 0.]], dtype=np.complex128),
          np.array([[1., 0.], [0., -1.]], dtype=np.complex128)]

# channel name: (T, number of rows kept). The rows of T are the operators
# A_c = sum_a T_ca S^a, S^a = sigma^a/2; the first rows are the channel and
# the others complete an invertible change of basis, used to check that
# the vertex does not couple the channel to the rest.
_CHANNELS = {
    "raising": (np.array([[1., 1j, 0.], [1., -1j, 0.], [0., 0., 1.]]), 1),
    "lowering": (np.array([[1., -1j, 0.], [1., 1j, 0.], [0., 0., 1.]]), 1),
    "full": (np.identity(3, dtype=np.complex128), 3)}


def get_channel_names():
    """Names accepted by channel= of spin_rpa_kernel and moment_rpa_kernel"""
    return list(_CHANNELS)


def _channel(channel):
    if channel not in _CHANNELS:
        raise ValueError("unknown channel %r of the spin response; the "
                "accepted ones are %s" % (channel,
                ", ".join(repr(c) for c in get_channel_names())))
    T, nc = _CHANNELS[channel]
    return np.array(T, dtype=np.complex128), nc


def _conserves_sz(h, tol=1e-10):
    for m in h.get_multihopping().get_dict().values():
        m = np.asarray(algebra.todense(m))
        if np.max(np.abs(m[0::2, 1::2])) > tol: return False
        if np.max(np.abs(m[1::2, 0::2])) > tol: return False
    return True


class LocalSpinKernel():
    """The kernel K(q,w) = chi0(q,w)^-1 - V(q) of the response of the spin
    operators of some sites, as the complex conjugate whose null vector is
    the amplitude of the mode (module docstring). Built by spin_rpa_kernel
    and moment_rpa_kernel; the work that does not depend on the frequency,
    the diagonalization of the mesh and the matrix elements of every pair,
    is done once per q and kept for the last q asked for.

    h: spinful Hamiltonian without the Nambu degree of freedom, with the
        Fermi energy at zero.
    channel: one of get_channel_names().
    sites: indices of the sites whose spin operators are kept.
    vertex: callable q -> 3M x 3M matrix, the vertex at the momentum q of
        the mode for the operators (Sx,Sy,Sz)/2 of the M sites, component
        a of the i-th site at index a*M + i. None: the bare response.
    nk: k-mesh of the bubble along each periodic direction.
    T: temperature of the occupations, zero or a positive number. T=0 takes
        sharp occupations, f=1 below the Fermi energy, f=0 above and 1/2
        for a state within DEGENERATE_TOL of it, as the pair basis of
        bsetk.pairbasis does, and a Fermi surface is then a degenerate pair
        at the Fermi energy with a nonzero matrix element. T > 0 takes the
        Fermi function. It is not the temperature the mean field was
        converged at: a pair between two occupied (or two empty) states has
        the weight f_a - f_b of order exp(-gap/2T) for a gap centered at
        the Fermi energy, and above weight_tol it is a transition of low
        energy, which puts the continuum near zero. Measured on a chiral
        spinon state with a gap of 0.34: at T=0.01 the continuum came out
        at 0.02, at T=1e-3 at 0.43. So a positive T has to be below about
        gap/50, and at T=0 there are no thermal pairs.
    weight_tol: a pair whose weight |f_a - f_b| |X|^2 is below weight_tol
        times the largest one is dropped, from the bubble and from the
        continuum."""

    def __init__(self, h, channel, sites, vertex, nk, T, weight_tol=1e-10):
        check.require_spin(h, "the kernel of a spin response")
        if h.has_eh:
            raise NotImplementedError("the kernel of a spin response is not "
                    "implemented for a Hamiltonian with the electron-hole "
                    "(Nambu) degree of freedom")
        if T is None or not np.isscalar(T) or not T >= 0:
            raise ValueError("T is the temperature of the occupations of the "
                    "bubble and must be zero (sharp occupations) or a "
                    "positive number; got "+repr(T))
        Tc, nc = _channel(channel)
        if nc < 3 and not _conserves_sz(h):
            raise ValueError("channel=%r keeps one circular component of the "
                    "spin and needs a mean field that conserves Sz, and this "
                    "Hamiltonian has elements between the two spins; use "
                    "channel='full'" % (channel,))
        self.h, self.channel, self.T, self.nk = h, channel, float(T), nk
        self.sites = np.array(sites, dtype=int)
        self.vertex, self.weight_tol = vertex, weight_tol
        n = len(h.geometry.r)
        M = len(self.sites)
        # T acts on the components of every site, T (x) 1_M in the layout
        # a*M + i; the operators of the channel as 2x2 spin matrices
        self._T = np.kron(Tc, np.identity(M))
        self._nkeep = nc*M
        self._ops = np.array([sum(Tc[c, a]*_SIGMA[a]/2. for a in range(3))
                              for c in range(nc)])
        self._n = n
        self._hk = h.get_hk_gen()
        self._ks = np.array(h.geometry.get_kmesh(nk=nk), dtype=np.float64)
        # the states at k, on the mesh, do not depend on q
        self._mesh = [self._eigh(k) for k in self._ks]
        self._q, self._data = None, None

    def _eigh(self, k):
        e, u = np.linalg.eigh(np.array(algebra.todense(self._hk(k)),
                                       dtype=np.complex128))
        return e, np.asarray(u).T.reshape(2*self._n, self._n, 2)[:, self.sites]

    def _vertex(self, q):
        """The vertex in the operators of the channel, kept block only"""
        D = self._nkeep
        if self.vertex is None: return np.zeros((D, D), dtype=np.complex128)
        V = np.array(self.vertex(q), dtype=np.complex128)
        Ti = np.linalg.inv(self._T)
        Vc = Ti.conj().T@V@Ti
        if np.max(np.abs(Vc - Vc.conj().T)) > 1e-8*max(1., np.max(np.abs(Vc))):
            raise ValueError("the vertex at q=%s is not Hermitian" % (q,))
        off = Vc[:D, D:]
        if D < Vc.shape[0] and np.max(np.abs(off)) > 1e-10*max(1.,
                np.max(np.abs(Vc))):
            raise ValueError("the vertex couples the channel %r to the other "
                    "spin components (largest element %.2e), so the channel "
                    "is not a block of the response; use channel='full'"
                    % (self.channel, np.max(np.abs(off))))
        return Vc[:D, :D]

    def _occupations(self, e):
        """The occupations of the states of energies e, and the weight of a
        degenerate pair made of a state of energy e at w -> 0, beta f(1-f)
        at T > 0, and at T = 0 an indicator of a state at the Fermi energy,
        whose weight is a delta function"""
        if self.T == 0.:
            fermi = np.abs(e) < DEGENERATE_TOL
            f = np.where(fermi, 0.5, np.where(e < 0., 1., 0.))
            return f, fermi.astype(np.float64)
        beta = 1./self.T
        f = expit(-beta*e)
        return f, beta*f*(1. - f)

    def _pairs(self, q):
        """Weights, energies and matrix elements of the pairs at q"""
        gs, ds, Xs, dw, xmax = [], [], [], [], 0.
        for k, (e2, b) in zip(self._ks, self._mesh):
            e1, a = self._eigh(k + q)
            a = np.conj(a) # (state at k+q, site, spin)
            X = np.einsum("mis,cst,nit->mnci", a, self._ops, b)
            X = X.reshape(len(e1), len(e2), -1) # component c of site i at c*M+i
            (f1, fp), (f2, _) = self._occupations(e1), self._occupations(e2)
            g = f1[:, None] - f2[None, :]
            d = e1[:, None] - e2[None, :]
            degenerate = np.abs(d) < DEGENERATE_TOL
            x2 = np.sum(np.abs(X)**2, axis=2)
            xmax = max(xmax, np.max(x2))
            # a degenerate pair contributes only at w=0 when delta -> 0, but
            # with a Fermi surface it means a continuum down to zero
            dw.append(np.max(np.where(degenerate, fp[:, None]*x2, 0.)))
            keep = ~degenerate
            gs.append(g[keep]); ds.append(d[keep]); Xs.append(X[keep])
        nk = len(self._ks)
        g, d, X = np.concatenate(gs)/nk, np.concatenate(ds), np.concatenate(Xs)
        w = np.abs(g)*np.sum(np.abs(X)**2, axis=1)
        wpair = np.max(w) if len(w) > 0 else 0.
        if self.T == 0.: # a weight delta(e): any matrix element is a metal
            metal = max(dw) > self.weight_tol*xmax
            wmax = wpair
        else:
            wmax = max(wpair, max(dw)/nk)
            metal = max(dw)/nk > self.weight_tol*wmax
        if wmax == 0. and not metal:
            raise ValueError("the spin operators of the channel %r couple to "
                    "no transition of this Hamiltonian at q=%s, so there is "
                    "no response and no mode" % (self.channel, q))
        keep = w > self.weight_tol*wmax
        return g[keep], d[keep], X[keep], metal

    def _at(self, q):
        q = _as_q(q)
        if self._q is not None and np.array_equal(q, self._q): return self._data
        g, d, X, metal = self._pairs(q)
        # the operators that couple to some transition, the range of chi0
        G = (X.T*np.abs(g))@np.conj(X)
        ev, U = np.linalg.eigh((G + G.conj().T)/2.)
        r = ev > 1e-10*np.max(ev)
        B = None if np.all(r) else U[:, r]
        XB = X if B is None else X@np.conj(B) # components on the basis B
        V = self._vertex(q)
        VB = V if B is None else B.conj().T@V@B
        # the modes are counted at positive frequency only, so the window
        # starts above zero whatever the poles at negative frequency
        hi = 0. if metal else (np.min(d[d > 0.]) if np.any(d > 0.) else np.inf)
        self._q, self._data = q, (g, d, X, XB, B, V, VB, (0., hi))
        return self._data

    def bare(self, q, w):
        """chi0(q,w) of the operators of the channel, in the convention of
        chitk.chiAB at the momentum -q (the first operator creates the
        mode at q), with delta = 0"""
        g, d, X = self._at(q)[0:3]
        return (X.T*(g/(d - w)))@np.conj(X)

    def response(self, q, w):
        """The RPA response chi0 (1 - V chi0)^-1 of the operators of the
        channel, in the convention of bare"""
        V = self._at(q)[5]
        c = self.bare(q, w)
        return c@np.linalg.inv(np.identity(c.shape[0]) - V@c)

    def __call__(self, q, w):
        """conj(K(q,w)), K = chi0^-1 - V on the operators that couple to
        some transition, whose null vector is the amplitude of the mode"""
        g, d, X, XB, B, V, VB, edge = self._at(q)
        c = (XB.T*(g/(d - w)))@np.conj(XB)
        K = np.linalg.inv((c + c.conj().T)/2.) - VB
        return np.conj(K)

    def continuum(self, q):
        """(0, upper): the kernel is smooth up to the lowest transition at
        positive frequency, upper, and the modes are counted above zero.
        upper is 0 when there is a Fermi surface, a continuum down to zero"""
        return self._at(q)[7]

    def basis(self, q):
        """None when every operator of the channel couples to some
        transition, otherwise the (D, r) matrix whose columns span the
        operators that do, conjugated as the kernel"""
        B = self._at(q)[4]
        return None if B is None else np.conj(B)


def spin_rpa_kernel(h, nk=10, T=0., channel="raising"):
    """Return the kernel of the site-basis spin RPA of a mean field, a
    LocalSpinKernel to pass to kernel_modes and kernel_chern.

    The vertex is the one of chitk.spinchi.spinchi_full,
    chitk.spinchi._full_spin_U, Fourier transformed at the momentum of the
    response. This covers the two mean fields whose spin response is
    dressed with a site-separable vertex: a mean field with an onsite
    interaction (a Hubbard U), where the site-basis RPA of the transverse
    response is exactly time-dependent Hartree-Fock, and a spinon mean
    field converged with decoupling="bond" (spinon.SpinonHamiltonian),
    whose vertex is the exchange J(q) of arXiv:2503.13831. An interaction
    between different sites of a Hartree-Fock mean field has a Fock rung
    that this vertex has no place for, and is refused; its magnons are the
    eigenvectors of the pair basis (bsetk.spinflip.magnon_matrix).

    h: the mean-field Hamiltonian, with the interaction it was converged
        with (h.V, h.Vchannels).
    nk: k-mesh of the bubble along each periodic direction, independent of
        the mesh of q of the Chern number.
    T: temperature of the occupations of the bubble, 0 (sharp occupations,
        the default) or a positive number, see LocalSpinKernel.
    channel: the spin operators, one of get_channel_names(); see the module
        docstring for which one holds the modes."""
    from . import spinchi
    from .rpa import interaction_at_q
    check.require_spin(h, "the spin RPA kernel")
    if spinchi._use_pair_basis(h):
        raise NotImplementedError("the interaction of this mean field couples "
                "different sites, and its Fock rung is not part of the "
                "site-basis vertex, so the site-basis RPA does not give its "
                "modes; use the eigenvectors of the pair basis "
                "(bsetk.spinflip.magnon_matrix)")
    V = spinchi._full_spin_U(h)
    if V is None:
        raise ValueError("the spin RPA kernel needs the interaction of the "
                "mean field, and this Hamiltonian carries none (h.V is "
                "None); build it with get_mean_field_hamiltonian")
    sites = list(range(len(h.geometry.r)))
    def vertex(q): # the response at -q in the convention of chitk
        return interaction_at_q(V, h, -np.array(q))
    return LocalSpinKernel(h, channel, sites, vertex, nk, T)


def moment_rpa_kernel(h, U=None, nk=10, T=0., channel="raising",
                      hybridization_scale=1.0):
    """Return the kernel of the spin susceptibility of the local moments of
    a Kondo lattice mean field with the residual interaction U'(q) of
    arXiv:2302.09701, the one of kondotk.momentchi.moment_spinchi, as a
    LocalSpinKernel to pass to kernel_modes and kernel_chern.

    The bubble is the one of the spin operators of the f sites, with the
    states of the whole Hamiltonian, and the vertex is the exchange 2U'(q)
    between the three spin components of the moments, which in the raising
    channel is chi = 1/(1/chi0 - U'). With one moment per cell and a
    raising or lowering channel the kernel is a number, so the bundle of
    its null vector is trivial and its Chern number zero by construction;
    a Chern number needs at least two moments per cell.

    h: Hamiltonian of a Kondo lattice mean field (spinful).
    U: residual interaction, a number, a matrix between the moments of the
        cell, or a function of q that returns one of them, as in
        moment_spinchi, where q is the momentum of moment_spinchi; the
        kernel at the momentum q of the mode evaluates it at -q (see the
        module docstring). None is the bare response, which has no mode
        below the continuum.
    nk, T, channel: those of spin_rpa_kernel.
    hybridization_scale: factor of the hybridization, as in moment_spinchi."""
    from ..kondotk import pairs as pairstk
    from ..kondotk.momentchi import _residual_at_q
    what = "the kernel of the spin susceptibility of the local moments"
    pairs = pairstk.get_pairs(h, what)
    h = pairstk.scale_hybridization(h, hybridization_scale, pairs)
    fsites = [fi for (ci, fi) in pairs]
    nf = len(fsites)
    vertex = None
    if U is not None:
        def vertex(q): # U S_q.S_-q is the exchange 2U between the components
            return np.kron(np.identity(3),
                           2.*_residual_at_q(U, -np.array(q), nf))
    return LocalSpinKernel(h, channel, fsites, vertex, nk, T)


def moment_rpa_chern(h, window, U=None, nk=10, T=0., channel="raising",
                     hybridization_scale=1.0, **kwargs):
    """Chern number of a band of the spin exciton of the local moments of a
    two-dimensional Kondo lattice mean field, the magnon of the moments:
    kernel_chern of the kernel of moment_rpa_kernel.

    window: (w_a, w_b), the frequencies between which the zeros are
        counted, chosen by hand: both ends in gaps of the mode spectrum at
        every q, w_a above zero and w_b below the continuum.
    U, nk, T, channel, hybridization_scale: those of moment_rpa_kernel. A
        spin triplet is read in the raising channel, the default, which
        holds one copy of it, and one moment per cell gives zero there by
        construction.
    kwargs: those of kernel_chern (bands, nq, shift, tol, full_output, ...).

    The residual interaction U'(q) is a parameter of the model, so the
    integer is a property of the model with that U'."""
    plaquette.require_2d(h, "the magnon of the moments")
    K = moment_rpa_kernel(h, U=U, nk=nk, T=T, channel=channel,
                          hybridization_scale=hybridization_scale)
    return kernel_chern(K, window, **kwargs)
