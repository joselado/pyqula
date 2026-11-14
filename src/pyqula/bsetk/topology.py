"""Chern numbers of collective modes: the excitons of the Bethe-Salpeter
equation (bsetk/solve.py) and the magnons of time-dependent Hartree-Fock
(bsetk/spinflip.py).

The question is what plays the role of the Bloch state. For an electron
band it is an eigenvector of H(k) in a fixed orbital basis. A collective
mode at center-of-mass momentum Q is an eigenvector of a matrix written in
a pair basis that is rebuilt at every Q: PairBasis diagonalizes H at k and
at k+Q, and eigh leaves an arbitrary phase on every Bloch state, so the
amplitudes X_vc(k) of two neighbouring Q cannot be overlapped as they are.
The way we fix this is by lifting the amplitudes to the orbital pair
basis, a space that does not depend on Q,

  Phi_Q(k; a, b) = sum_vc X_vc(k) u_{c,k+Q}(a) conj(u_{v,k}(b)),

meaning that Phi_Q is the amplitude of the operator c^dag_{a,k+Q} c_{b,k}
in the mode, an electron in orbital a at k+Q and a hole in orbital b at k.
The rows of PairBasis.el and .ho are the eigenvectors u themselves (not
their conjugates), so this is sum_m X_m el_m(a) conj(ho_m(b)). Under
u_n -> exp(i phi_n) u_n the Casida matrix is conjugated by a diagonal
phase matrix and X_m picks up exp(-i phi_c(k+Q) + i phi_v(k)), which
cancels against the two u, so Phi_Q does not depend on the Bloch gauge.

The antiresonant amplitude Y lives on the pairs elA, hoA, the
de-excitations c^dag_{v,k+Q} c_{c,k}: the hole band v at k+Q and the
electron band c at k. The Casida matrix is [[A,B],[-B^dag,-conj(Abar)]],
and the same phase argument applied to its lower block row gives
Y_m -> exp(+i phi_c(k) - i phi_v(k+Q)) Y_m, the conjugate of the phases of
the pair (v,k+Q),(c,k). The invariant lift is therefore

  Psi_Q(k; a, b) = sum_vc Y_vc(k) u_{v,k+Q}(a) conj(u_{c,k}(b)),

which is the amplitude of the same operator c^dag_{a,k+Q} c_{b,k}, now
taken between an occupied state at k+Q and an empty one at k, up to a
sign: the lower block row of the Casida matrix carries a minus, so a
one-body operator enters the eigenproblem as (X,-Y) (goldstone_vector of
bsetk/spinflip.py), and Psi of an eigenvector is minus that amplitude.
The sign of a whole block changes no overlap between two Q, in either
metric, so it changes neither the fluxes nor the integer. The two
lifts are kept as two blocks of one vector of 2*nk*norb^2 numbers, so the
Casida metric diag(1,-1) of the eigenproblem is diag(1,-1) on the blocks
too. The lift of each block is an isometry, so the Euclidean norm and the
Casida norm X^dag X - Y^dag Y survive it unchanged.

The exact check of the lift is a one-body operator O at momentum transfer
Q: written in the pair basis, as goldstone_vector writes the spin
generator, it must lift to P_emp(k+Q) O P_occ(k) and P_occ(k+Q) O P_emp(k),
and it does to 1e-15, with and without random phases on the Bloch states
(tests/bse/test_exciton_chern.py). The integer alone is a much weaker
check of the Y block. On the Lieb ferrimagnet of
tests/magnon/test_magnon_chern.py, whose lowest magnon has |Y|/|X| up to
about 0.5, three wrong lifts of Y were tried: the electron and the hole
exchanged, sum Y elA conj(hoA), which is not gauge invariant; the
conjugate, sum conj(Y) conj(hoA) elA, which is gauge invariant but
antilinear in the mode; and Y dropped. Under random Bloch phases the
lifted state moves for the first two (overlap 0.965 and 0.949 with the
state in the raw gauge, against 1 for the right lift), and the integer
stays at +1 for all three (nq=6, nk=6). A reason for this is that
|X| > |Y| on the excitation branch, so the resonant block never vanishes
and alone spans a bundle isomorphic to the one of the mode (derived);
that a wrong Y block then leaves the integer alone is measured on this one
case only, and is not a guarantee. A missing lift is
different: the raw amplitudes of the pair basis carry the phases of the
Bloch states, and on the zero model of the exciton tests they give C = -5
under random phases where the lifted states give 0.

The convention of an electron at k+Q and a hole at k is the one PairBasis
uses, and it is the electron Wilson loop of Davenport, Knolle and
Schindler (arXiv:2507.22983, the conduction sector, alpha=0 in their
family of exciton Berry connections; arXiv:2608.07372 writes the same
Wilson line with the hole momentum held fixed). It attaches the position
of the mode to the electron. Since pyqula's Bloch Hamiltonian is periodic
in the reduced momentum (the cell gauge), Phi_{Q+G} = Phi_Q, so the
plaquettes close on the torus of Q. The split k+Q/2, k-Q/2 of the center
of mass (Kwan, Hu, Simon and Parameswaran, arXiv:2003.11560, who define
the exciton Chern number from the Fukui-Hatsugai-Suzuki plaquette on the
cell-periodic two-body state) is not periodic in Q on a lattice, and it is
not used here. The Chern number does not depend on this choice:
arXiv:2507.22983 writes the Chern number of its connection alpha as
alpha C_exc,v + (1-alpha) C_exc,c, with C_exc,c and C_exc,v the Chern
numbers of the exciton band in the electron and in the hole convention
(not those of the electron bands), and argues that it is an integer for
every alpha, which needs C_exc,c = C_exc,v. The same follows from the
electron and the center-of-mass connections differing by half the exciton
polarization, the mean electron-hole separation, a periodic function of Q
for a bound mode whose curl integrates to zero (our argument, not the
paper's). The Berry curvature of a single plaquette does depend on the
convention, and the fluxes returned here are those of the electron one.

Once lifted, a mode is a state in a fixed space, and the plaquette product
of topologytk/plaquette.py (Fukui, Hatsugai and Suzuki, cond-mat/0503172)
gives its Chern number, with the sign of h.get_chern(). The mesh of Q and
the internal k-mesh nk of the pair basis are independent. The integer is
the Chern number of the mode only when the mode is converged in nk and
isolated from the other modes at every Q; a mode inside the electron-hole
(or Stoner) continuum of a finite mesh is a discrete level among the
continuum states, and its integer means nothing. A touching point that
falls between two points of the mesh is not seen, and the integer then
comes back with no error. Measured on the Frenkel model of the exciton
tests with no flux, which is time-reversal symmetric and whose two
excitons touch at Q=0: the lowest one alone gives -1 on a shifted mesh
(nq=4, nk=4), where the smallest gap on the mesh is 0.0055, and the
group of the two gives 0. The gap of the full_output record is the check.

The momentum of a flux. The map Q -> -Q is a rotation of the plane, so it
keeps the orientation, and a routine that solved every point at -Q would
return the same integer (measured: with that change every integer of
the two test files stays). What it would change is which plaquette a
flux belongs to and the energies of the record, and those are pinned in
the tests on two models with
E(Q) != E(-Q): the Frenkel model with a flux against its oracle, and a
ferromagnet with a sublattice imbalance against magnon_energies.

The sign of the magnon numbers has a check that does not use this
package's two-particle code. At strong coupling the single magnon of a
saturated ferromagnet hops between the sites i and j with the amplitude
(2/U) t_ij(flipped spin) conj(t_ij(majority spin)), the superexchange of
second order in the hopping, which is a tight-binding model whose Chern
numbers the plaquette product gives directly. On the saturated honeycomb
ferromagnet of the magnon tests its splitting of the two magnon bands is
the one of time-dependent Hartree-Fock within 1.4 % at U=40 (nk=6 and
nk=8) and within 30 % at U=8 (nk=8), and its Chern numbers are those of
magnon_chern at U=40 and at U=8, where both signs of the Kane-Mele coupling and both
signs of the field were compared; a field of the other sign polarizes
the magnet the other way and goes through the other spin channel
(tests/magnon/test_magnon_chern.py). The model is derived here and not
taken from a paper; what supports it is the agreement of the splittings
at large U.

The Casida metric. A mode of the full BSE or of a magnon problem with a
de-excitation half has X^dag X - Y^dag Y > 0 on the excitation branch of a
stable reference, so the Casida metric is positive on the group of modes,
and the Chern number of the group, a property of the bundle the modes
span, is the same with the Euclidean overlap and with metric="casida"
(derived). Measured: the same integer with both on the full BSE of the
Frenkel model of the exciton tests (C = -1, |Y|/|X| = 0.10) and on the two
bound magnons of the Lieb ferrimagnet (+1 and -1). The bosonic Berry
curvature of the magnon thermal Hall effect (arXiv:1106.1987,
arXiv:1204.3349) is written in the metric diag(1,-1), which is why the
fluxes take metric=; whether the fluxes of the electron convention are the
curvature that enters a thermal Hall conductivity is not settled here.

An unstable reference. When the interaction binds an exciton below zero
energy, the resonant block A has a negative eigenvalue, and the positive
branch of the full BSE holds its mirror, a mode with X = 0 and a negative
norm. Measured on the zero model of the exciton tests with the flat level
at -1: an exciton at -0.715 in A, and the mirror at +0.715 as the third
mode of the positive branch. The modes counted from the lowest are then
not the excitations of the reference, so a negative energy or a
nonpositive norm on the excitation branch raises instead of returning an
integer. So does a complex energy anywhere on the branch, and a positive
branch of the full BSE whose number of modes is not the number of pairs.
The norm of a mode of complex energy vanishes, so its sign is roundoff,
and BSE.solve keeps or drops a purely imaginary pair by the roundoff of
its real part. Measured on a honeycomb Neel state converged at U=4 with
the kernel of U=4.5 (nk=6, Q=0): the unstable magnon has the energy
0.84i and the norm +1.5e-15, the full BSE returns 142 modes for 144
pairs, and before these two checks a mode above it came back from both
with no error.

The Goldstone point. With a de-excitation half and no net moment (an
antiferromagnet with spin rotation symmetry) the zero mode at Q=0 is
defective, its norm vanishes, and its energy comes out as the square root
of the residual of the mean field, real or imaginary (2.9e-5 i on the
Neel state above). The imaginary side raises as a complex energy, for
every group of bands, since the test is on the whole branch; the real
side is not caught, so a mesh of an ungapped antiferromagnet has to
avoid Q=0 with shift=. A ferrimagnet and a ferromagnet have a regular
zero mode, of norm proportional to the net moment (measured: the Lieb
ferrimagnet with no field returns a state at Q=0)."""

import numpy as np

from .. import parallel
from ..topologytk import plaquette


# metric= of the Chern number, as the diagonal of the inner product on the
# two blocks (Phi, Psi) of a lifted mode
_metrics = {"euclidean": (1., 1.), "casida": (1., -1.)}


def get_metric_names():
    """Every value metric= accepts, besides None (which is "euclidean")"""
    return sorted(_metrics)


def _check_metric(metric):
    if metric is None: return "euclidean"
    if metric not in _metrics:
        raise ValueError("metric must be None or one of %s, got %r"%(
            ", ".join(repr(m) for m in get_metric_names()), metric))
    return metric


def lift_amplitudes(pb, X, Y=None, m1=None, m2=None):
    """Return the modes with amplitudes X (and Y) on the pair basis pb,
    lifted to the orbital pair basis, as an (n, 2*nk*norb^2) array, one
    mode per row: the resonant block Phi followed by the antiresonant
    block Psi, each indexed by (k, a, b) with k the hole momentum of a
    resonant pair (see the module docstring).

    pb: a PairBasis.
    X: (n, n1) resonant amplitudes, on the pairs selected by the mask m1
        (None: every pair of pb). A single mode can be given as a vector.
    Y: (n, n2) antiresonant amplitudes on the pairs of the mask m2, or None
        for a mode with no de-excitation part (the Tamm-Dancoff BSE, a
        saturated ferromagnet), whose Psi block is then zero.
    m1, m2: boolean masks of the pairs X and Y live on, as magnon_matrix
        returns them; for a metal they select a different number of pairs
        at every k-point, which the lift tolerates."""
    nk = len(pb.kpoints)
    norb = pb.ck.shape[1]
    X = np.array(X, dtype=np.complex128)
    if X.ndim == 1: X = X[None, :]
    if m1 is None: m1 = np.ones(pb.npair, dtype=bool)
    if X.shape[1] != int(np.sum(m1)):
        raise ValueError("X has %d amplitudes per mode but the mask selects "
            "%d pairs"%(X.shape[1], int(np.sum(m1))))
    Phi = _lift_block(X, pb.el[m1], pb.ho[m1], pb.kindex[m1], nk, norb)
    if Y is None:
        Psi = np.zeros(Phi.shape, dtype=np.complex128)
    else:
        Y = np.array(Y, dtype=np.complex128)
        if Y.ndim == 1: Y = Y[None, :]
        if m2 is None: m2 = np.ones(pb.npair, dtype=bool)
        if Y.shape != (X.shape[0], int(np.sum(m2))):
            raise ValueError("Y has shape %s but %d modes on %d pairs were "
                "expected"%(str(Y.shape), X.shape[0], int(np.sum(m2))))
        # hole (v,k+Q) in the first index, electron (c,k) in the second
        Psi = _lift_block(Y, pb.hoA[m2], pb.elA[m2], pb.kindex[m2], nk, norb)
    return np.concatenate([Phi, Psi], axis=1)


def _lift_block(X, left, right, kindex, nk, norb):
    """out[s, k, a, b] = sum over the pairs m at k of
    X[s,m] left[m,a] conj(right[m,b]), flattened to (n, nk*norb^2)"""
    out = np.zeros((X.shape[0], nk, norb, norb), dtype=np.complex128)
    pairs = left[:, :, None]*np.conj(right)[:, None, :] # (npair,norb,norb)
    for s in range(X.shape[0]):
        np.add.at(out[s], kindex, X[s][:, None, None]*pairs)
    return out.reshape(X.shape[0], -1)


def casida_metric(pb):
    """The diagonal of the Casida metric diag(1,-1) in the lifted space of
    lift_amplitudes: +1 on the resonant block and -1 on the antiresonant
    one"""
    n = len(pb.kpoints)*pb.ck.shape[1]**2
    return np.concatenate([np.ones(n), -np.ones(n)])


def _metric_vector(metric, dim):
    """The diagonal of the inner product of the lifted space, or None"""
    metric = _check_metric(metric)
    if metric == "euclidean": return None
    s1, s2 = _metrics[metric]
    return np.concatenate([np.full(dim//2, s1), np.full(dim//2, s2)])



class _Modes():
    """The lifted states of a group of modes at one Q, with what decides
    whether their Chern number means anything: the energies of the group,
    its distance to the modes below and above (inf when there are none),
    and the lowest electron-hole transition of the pair basis, the edge of
    the continuum on this mesh"""
    def __init__(self, states, energies, below, above, continuum):
        self.states = states
        self.energies = energies
        self.below, self.above = below, above
        self.continuum = continuum


def _group(es, bands, what, norms=None):
    """Energies of the group and its gaps to the rest, from the energies
    es of the whole excitation branch sorted by their real part, and the
    Casida norms X^dag X - Y^dag Y of its modes (None when there is no
    de-excitation half)"""
    if bands[-1] >= len(es):
        raise ValueError("bands %s asks for mode %d, but this %s has %d "
            "modes"%(bands, bands[-1], what, len(es)))
    eg = es[bands]
    scale = max(1., float(np.max(np.abs(es.real))))
    # Three signs that the reference is not a stable state: a complex
    # energy, a negative energy, and a mode of the positive branch with a
    # negative norm (the mirror of a negative energy). Each is looked for
    # on the whole excitation branch and not only in the group, since the
    # modes counted from the lowest are then not the excitations they are
    # taken for. The complex energies go first: their norm vanishes, so
    # its sign is roundoff and cannot be tested
    imag = float(np.max(np.abs(es.imag)))
    if imag > 1e-6*scale:
        raise ValueError("the %s has an excitation with a complex energy "
            "(largest imaginary part %g), so the mean-field reference is "
            "unstable against it, or the mesh of Q hits the Goldstone mode "
            "of a magnet with no net moment, whose eigenvector is not "
            "defined; a shift of the mesh keeps Q=0 out of it"%(what, imag))
    if np.min(es.real) < -1e-8*scale:
        raise ValueError("the %s has an excitation of negative energy "
            "(lowest %g), so the mean-field reference is unstable against "
            "it and the modes counted from the lowest are not its "
            "excitations; the Chern number of a group needs a stable "
            "reference"%(what, float(np.min(es.real))))
    if norms is not None and np.min(norms) <= 0.:
        raise ValueError("the %s has a mode of negative norm X^dag X - "
            "Y^dag Y on its excitation branch (lowest norm %g), the mirror "
            "of an excitation of negative energy, so the mean-field "
            "reference is unstable against it and the modes counted from "
            "the lowest are not its excitations; the Chern number of a "
            "group needs a stable reference"%(what, float(np.min(norms))))
    er = es.real
    below = er[bands[0]] - er[bands[0]-1] if bands[0] > 0 else np.inf
    above = er[bands[-1]+1] - er[bands[-1]] if bands[-1]+1 < len(er) \
            else np.inf
    return eg.real, below, above


def _exciton_modes(h, Q, bands, kwargs):
    """Solve the BSE at Q and lift the group of excitons"""
    from .solve import BSE
    solver = kwargs.get("solver", "dense")
    if solver != "dense":
        raise NotImplementedError("the exciton Chern number is built on "
            "solver='dense' only, got solver=%r: the iterative solver gives "
            "the neig lowest excitons under the Tamm-Dancoff approximation "
            "and the quantics one never builds the pair arrays the lift "
            "needs"%(solver,))
    b = BSE(h, Q=Q, **kwargs)
    es = np.array(b.energies, dtype=np.complex128)
    if not b.tda and len(es) != b.pairs.npair:
        # BSE.solve keeps the eigenvalues of positive real part, one per
        # pair for a stable reference; a purely imaginary pair is dropped
        # or kept by the roundoff of its real part, and when it is dropped
        # nothing else tells that the lowest mode kept is not the lowest
        # excitation
        raise ValueError("the positive branch of the BSE has %d modes for "
            "%d electron-hole pairs, so some excitation has an imaginary "
            "energy: the mean-field reference is unstable against it, and "
            "the modes counted from the lowest are not its excitations; "
            "the Chern number of a group needs a stable reference"%(
            len(es), b.pairs.npair))
    order = np.argsort(es.real, kind="stable") # already sorted, kept explicit
    es = es[order]
    norms = None
    if not b.tda:
        norms = (np.sum(np.abs(b.amplitudes)**2, axis=1)
                - np.sum(np.abs(b.amplitudesY)**2, axis=1))[order]
    eg, below, above = _group(es, bands, "BSE", norms=norms)
    idx = order[bands]
    Y = None if b.tda else b.amplitudesY[idx]
    states = lift_amplitudes(b.pairs, b.amplitudes[idx], Y)
    return _Modes(states, eg, below, above, float(np.min(b.pairs.dE)))


def _magnon_excitations(p):
    """Energies and eigenvectors (rows) of the excitation branch of a
    MagnonProblem p, sorted by energy, the branch magnon_energies keeps:
    the Hermitian resonant block when there is no de-excitation half, and
    otherwise the n1 modes of largest Casida norm X^dag X - Y^dag Y"""
    from .solve import solve_pseudo_hermitian
    from .spinflip import _solve_resonant
    if p.n2 == 0:
        es, ws = _solve_resonant(p.M)
        norm = None
    else:
        es, ws = solve_pseudo_hermitian(p.M, n1=p.n1)
        ws = ws/np.linalg.norm(ws, axis=1)[:, None]
        norm = (np.sum(np.abs(ws[:, 0:p.n1])**2, axis=1)
                - np.sum(np.abs(ws[:, p.n1:])**2, axis=1))
        keep = np.argsort(-norm)[0:p.n1] # the excitation branch
        es, ws, norm = es[keep], ws[keep], norm[keep]
    order = np.argsort(es.real, kind="stable")
    if norm is not None: norm = norm[order]
    return np.array(es[order], dtype=np.complex128), ws[order], norm


def _magnon_modes(h, Q, bands, kwargs):
    """Solve the magnon problem at Q and lift the group of magnons"""
    from .spinflip import magnon_matrix
    p = magnon_matrix(h, Q=Q, **kwargs)
    es, ws, norms = _magnon_excitations(p)
    eg, below, above = _group(es, bands, "magnon problem", norms=norms)
    X = ws[bands, 0:p.n1]
    Y = ws[bands, p.n1:] if p.n2 > 0 else None
    states = lift_amplitudes(p.pb, X, Y, m1=p.m1, m2=p.m2)
    dE = p.pb.dE[p.m1]
    continuum = float(np.min(dE)) if len(dE) > 0 else np.inf
    return _Modes(states, eg, below, above, continuum)




# mode= of the general functions: name -> (solver at one Q, what the
# messages call the band)
_MODES = {"exciton": (_exciton_modes, "an exciton band"),
          "magnon": (_magnon_modes, "a magnon band")}


def get_mode_names():
    """Every value mode= accepts"""
    return sorted(_MODES)


def _mode(mode):
    if mode not in _MODES:
        raise ValueError("unknown mode %r of a collective mode; the accepted "
            "ones are %s"%(mode, ", ".join(repr(m) for m in get_mode_names())))
    return _MODES[mode]


def mode_states(h, mode="exciton", Q=None, bands=(0,), **kwargs):
    """Return the lifted states of a group of collective modes at the
    momentum Q, an (n, 2*nk*norb^2) array with one mode per row, in the
    space of lift_amplitudes, which does not depend on Q.

    mode: "exciton" for the excitons of bsetk.solve.BSE, whose keywords
        (V, nk, nv, nc, kernel, tda, screening, gauge, max_memory, ...;
        only solver='dense') go in kwargs, or "magnon" for the magnons of
        the excitation branch that magnon_energies returns, with the
        keywords of bsetk.spinflip.magnon_matrix (nk, V, channel, nv, nc,
        check_su2, metal, max_memory, ...).
    bands: indices of the modes counted from the lowest (a contiguous
        group, so a degenerate multiplet goes in as one group)."""
    modes_at, what = _mode(mode)
    if Q is None: Q = [0., 0., 0.]
    return modes_at(h, Q, plaquette.check_bands(bands), kwargs).states


def _mesh(modes_at, nq, shift):
    """The points of plaquette.mesh and the modes solved at each of them,
    once per point"""
    qs = plaquette.mesh(nq, shift)
    return qs, parallel.pcall(modes_at, list(qs.reshape(-1, 3)))


def _fluxes(modes_at, nq, shift, metric):
    """Solve the mesh and return its ModeChern record"""
    qs, modes = _mesh(modes_at, nq, shift)
    nq = qs.shape[0]
    S = _metric_vector(metric, modes[0].states.shape[1])
    V = [[modes[i*nq + j].states for j in range(nq)] for i in range(nq)]
    fluxes, worst = plaquette.mesh_berry_fluxes(V, metric=S)
    return plaquette.ModeChern(fluxes, worst, qs,
            [m.energies for m in modes],
            min(min(m.below, m.above) for m in modes),
            min(m.continuum - np.max(m.energies) for m in modes))


# the record of full_output, defined with the engine and named here too
ModeChern = plaquette.ModeChern


def _chern(modes_at, nq, shift, metric, tol, full_output):
    out = _fluxes(modes_at, nq, shift, metric)
    scale = max(1., float(np.max(np.abs(out.energies))))
    if out.gap < 1e-8*scale:
        raise ValueError("the group of modes is degenerate with another "
            "mode at some point of the mesh (smallest gap %.2e), so it has "
            "no Chern number of its own; add the other mode to the group "
            "(bands=) or move the mesh (shift=)"%(out.gap,))
    if out.min_overlap < tol:
        raise ValueError("two neighbouring groups of modes of the mesh are "
            "almost orthogonal (smallest normalized overlap %.2e, below "
            "tol=%.1e), so the plaquette sum does not resolve their Chern "
            "number. Either the group is not separated by a gap from the "
            "other modes at some Q (smallest gap on the mesh %.3g), or "
            "nq=%d is too coarse"%(out.min_overlap, tol, out.gap,
            len(out.fluxes)))
    if full_output: return out
    return out.chern


def mode_berry_fluxes(h, mode="exciton", bands=(0,), nq=plaquette.NQ,
        shift=(0., 0.), metric=None, **kwargs):
    """Return (fluxes, energies, min_overlap) of a group of collective
    modes on the nq x nq mesh Q = (i/nq, j/nq, 0) + shift of the reduced
    Brillouin zone: the Berry flux of each plaquette as an (nq,nq) array,
    whose sum over 2 pi is the Chern number, the energies of the group as
    (nq,nq,n), and the smallest normalized overlap of two neighbouring
    groups. An unstable reference raises as in mode_chern, but a group
    that touches another mode at a mesh point and a link overlap below
    tol are not refused here; mode_chern makes those two checks.

    The fluxes are those of the electron convention (module docstring);
    metric="casida" gives them in the Casida metric diag(1,-1), the metric
    the bosonic Berry curvature of a thermal Hall conductivity is written
    in (whether these fluxes are that curvature is not settled, module
    docstring). The other arguments are those of mode_chern. The modes are
    solved once per point of the mesh."""
    modes_at, what = _mode(mode)
    plaquette.require_2d(h, what)
    _check_metric(metric)
    bands = plaquette.check_bands(bands)
    out = _fluxes(lambda q: modes_at(h, q, bands, kwargs), nq, shift, metric)
    return out.fluxes, out.energies, out.min_overlap


def mode_chern(h, mode="exciton", bands=(0,), nq=plaquette.NQ,
        shift=(0., 0.), metric=None, tol=1e-6, full_output=False, **kwargs):
    """Return the Chern number of a group of bands of collective modes on
    top of the mean field h: the excitons of the Bethe-Salpeter equation
    (mode="exciton") or the magnons of time-dependent Hartree-Fock
    (mode="magnon").

    The two-particle problem is solved at every point of the mesh
    Q = (i/nq, j/nq, 0) + shift, the modes of the group are lifted to the
    orbital pair basis (module docstring), and the plaquette product of
    topologytk/plaquette.py gives the integer, with the sign of
    h.get_chern(). Only the excitation branch enters: the excitons of
    positive energy, and for the magnons the whole resonant problem of a
    saturated ferromagnet and otherwise the modes of positive Casida norm
    X^dag X - Y^dag Y, as magnon_energies selects them.

    mode: one of get_mode_names().
    bands: indices of the bands counted from the lowest mode of the
        excitation branch, a contiguous group, so a degenerate multiplet
        (a spin triplet, a valley pair) goes in as one group. Default: the
        lowest band.
    nq: points of the mesh of Q along each reciprocal lattice vector. It is
        independent of the internal k-mesh nk of the two-particle problem,
        and the mode has to be converged in nk before the integer is read.
        The default 12 holds Gamma, M and K of a hexagonal zone and Gamma,
        X and M of a square one, where a touching is refused.
    shift: offset of the mesh of Q, two numbers in reduced coordinates. A
        magnet with no gap at Q=0 and no net moment needs shift=, since its
        Goldstone mode has no eigenvector there, and whether the mesh with
        Q=0 raises depends on the sign of the residual of the mean field.
    metric: None or "euclidean" for the Euclidean overlap of the lifted
        states, "casida" for the Casida metric diag(1,-1); the integer is
        the same with both for a stable reference (module docstring).
    tol: smallest normalized overlap of two neighbouring groups accepted;
        below it, or when the group touches another mode at a mesh point,
        a ValueError is raised instead of an integer.
    full_output: True returns a plaquette.ModeChern record with the fluxes,
        the largest flux, the energies, the gap of the group to the other
        modes and its distance below the electron-hole (or Stoner)
        continuum of the mesh, which is how to check that the mode is
        isolated and bound.
    kwargs: for mode="exciton" the keywords of bsetk.solve.BSE (V, nk, nv,
        nc, kernel, tda, screening, channel, gauge, max_memory, ...), with
        solver='dense' only; for mode="magnon" those of
        bsetk.spinflip.magnon_matrix (nk, V, channel, nv, nc, check_su2,
        metal, transverse, max_memory), where the mean field has to be
        converged on the same nk as the magnon problem.

    A mode inside the continuum, as a magnon of a metal, has no band
    topology, and a touching between two points of the mesh is not seen;
    the gap and bound of the record, and a second nq, are the check."""
    modes_at, what = _mode(mode)
    plaquette.require_2d(h, what)
    _check_metric(metric)
    bands = plaquette.check_bands(bands)
    f = lambda q: modes_at(h, q, bands, kwargs)
    return _chern(f, nq, shift, metric, tol, full_output)


def exciton_chern(h, **kwargs):
    """Return the Chern number of a group of exciton bands of the
    Bethe-Salpeter equation on top of the mean field h, mode_chern with
    mode="exciton".

    The keywords are those of mode_chern (bands, nq, shift, metric, tol,
    full_output) and of bsetk.solve.BSE (V, nk, nv, nc, kernel, tda,
    screening, channel, gauge, max_memory, ...), with solver='dense' only.
    bands counts the excitons from the lowest one of positive energy."""
    return mode_chern(h, mode="exciton", **kwargs)


def magnon_chern(h, **kwargs):
    """Return the Chern number of a group of magnon bands of the
    time-dependent Hartree-Fock problem of bsetk/spinflip.py on top of the
    magnetic mean field h, mode_chern with mode="magnon".

    The keywords are those of mode_chern (bands, nq, shift, metric, tol,
    full_output) and of bsetk.spinflip.magnon_matrix (nk, V, channel, nv,
    nc, check_su2, metal, transverse, max_memory). bands counts the magnons
    from the lowest of the excitation branch, and the mean field has to be
    converged on the same nk as the magnon problem.

    Two things to know before reading the integer. The acoustic magnon of
    a magnet with no net moment has no eigenvector at the Goldstone point
    (a defective zero mode, module docstring), so it needs a gap, from a
    field or an anisotropy, or shift= to keep Q=0 out of the mesh. And a
    magnon inside the Stoner continuum, as in a metal, has no band
    topology; full_output gives its distance below the continuum of the
    mesh."""
    return mode_chern(h, mode="magnon", **kwargs)
