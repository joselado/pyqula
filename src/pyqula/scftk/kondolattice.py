"""SCF engine behind KondoLatticeHamiltonian (see ../kondolattice.py for the
public class and the full physics writeup). Implements the large-N
Abrikosov-pseudofermion / Read-Newns mean-field theory of the Kondo lattice,
following P. Coleman, "Heavy Fermions: electrons at the edge of magnetism",
arXiv:cond-mat/0612006, Sec. II.C.2 ("Mean field theory of the Kondo
lattice") verbatim -- Eq. 65-99 there.

The Coqblin-Schrieffer Kondo interaction at site j, Eq. 73,

    H_I(j) = (J/N) S_ab(j) c^dagger_jb c_ja,   S_ab(j) = f^dagger_ja f_jb - (n_f/N) delta_ab

is Hubbard-Stratonovich decoupled (Eq. 76-78) into a hybridization field
V_j = -(J/N) <c^dagger_ja f_ja> (summed over a, the N=2 spin components for
a spin-1/2 local moment; the stationary point in conj(V_j) of the mean-field
energy below, where V_j multiplies f^dagger_j c_j) coupling the localized
f-pseudofermion to the
conduction electron at the same site, plus a Lagrange multiplier lam_j
enforcing the local constraint <f^dagger_ja f_ja> = Q (=1 for N=2), giving
the mean-field Hamiltonian (Eq. 81/83)

    H_MFT = sum_k eps_k c^dagger_k c_k
          + sum_j [ V_j^* c^dagger_j f_j + V_j f^dagger_j c_j + lam_j f^dagger_j f_j ]
          + sum_j [ N|V_j|^2/J - lam_j Q ]

Both V_j and lam_j are kept per-site (arrays) rather than a single global
scalar, so a non-uniform (e.g. supercell) system self-consistently converges
to site-dependent values -- for a translationally invariant lattice they
converge to the same value on every site.

The coupling does not have to be onsite. A Kondo bond b = (c, f, R, J_b)
couples the spin of the conduction electron at site c of the cell R to the
moment f of the cell 0, with its own coupling J_b, so the interaction is
sum_b (J_b/N) S_ab(f) c^dagger_{c,R,b} c_{c,R,a} summed over the
translations of the lattice. Each bond is decoupled on its own, with its
own field

    V_b = -(J_b/N) sum_a <c^dagger_{c,R,a} f_{f,0,a}> ,

the term V_b f^dagger_{f,0} c_{c,R} + h.c. in the Hamiltonian (a hopping
of direction R from c to f, and its conjugate in direction -R) and the
constant N|V_b|^2/J_b, while lam_j stays one per moment. The onsite model
is the case of one bond per moment with c the site of the moment and
R = 0. A bond between different sites of the same cell has R = 0 too, and
a finite (0D) system has only those. This is the large-N decoupling of a
coupling J_ij s_i.S_j between every pair of sites, one channel per pair
(not the model with a hybridization form factor, in which one field per
moment multiplies a fixed pattern of neighbors).

This is deliberately NOT routed through scftk.spinspin's
VJinteraction/_run_anisotropic_scf (the engine SpinonHamiltonian reuses for
the analogous Heisenberg-model RVB decoupling): that machinery's per-site
"array filling" branch enforces a local occupation constraint on EVERY site
in the Hamiltonian, with no path for a subset of sites (only the localized
f-sites here) to be constrained while the rest (the conduction sites) are
left at a free, globally-set filling -- see kondolattice.py's module
docstring for why a plain, fixed global chemical potential plus a per-
f-site Lagrange multiplier is both simpler and matches Coleman's own
construction (Eq. 83 is a grand-canonical Hamiltonian at fixed mu, with the
total electron density left to float -- and indeed expand, Eq. 91-92 -- as
hybridization turns on; lam_j alone enforces the local f-constraint)."""
import numpy as np

from ..multihopping import MultiHopping

N_SPIN = 2 # N in the large-N Coqblin-Schrieffer coupling J/N: the number of
           # spin components of a spin-1/2 (Q=1) local moment

FIELD_PRECISION = 1e-10 # precision of a field relative to the largest one:
           # a field is a sum over the k-mesh and the states, so it is known
           # to roundoff times the largest field at best, and a field that
           # vanishes by symmetry sits there (see field_residual). 1e-10
           # leaves a wide margin over the roundoff of a large mesh, and a
           # tolerance at or below it keeps the residual relative to the
           # largest field


def field_residual(V, target, maxerror):
    """Largest change of the fields V towards their targets, each one
    relative to its own size max(|V_b|, |target_b|), as the field of a
    single moment converges, so that a small but genuine field converges
    to maxerror relative to itself and not to maxerror times the largest
    field.

    The size of a field is taken as at least FIELD_PRECISION/maxerror
    times the largest field: a field that vanishes by symmetry (the
    first-neighbor bond of a particle-hole symmetric chain with an onsite
    V) sits at roundoff, where its relative change is of order one at
    every iteration, so it counts as converged once it moves by less than
    FIELD_PRECISION times the largest field. The floor follows the largest
    field, so a trajectory where every V decays together towards V=0
    still has a constant residual and does not converge. With one field
    the floor is never reached and the residual is that of the field."""
    V = np.asarray(V)
    if len(V) == 0: return 0.0 # no bonds, nothing to converge
    size = np.maximum(np.abs(V), np.abs(target)) # of each field
    vmax = np.max(size)
    if vmax == 0.0: return 0.0 # all exactly zero
    if maxerror > FIELD_PRECISION: floor = vmax*(FIELD_PRECISION/maxerror)
    else: floor = vmax # a tolerance this tight is below the precision
    return np.max(np.abs(target - V)/np.maximum(size, floor))


def hs_constant(V, J):
    """Hubbard-Stratonovich constant of the Coqblin-Schrieffer decoupling.

    Decoupling -g X^dagger X (g = J/N, X = sum_a c^dagger_a f_a) as
    Vbar X + X^dagger V + |V|^2/g reproduces the interaction only with the
    constant |V|^2/g = N|V|^2/J -- the same g = J/N that the SCF update
    V <- -(J/N)<c^dagger f> below already uses, so the two have to carry
    the same N.

    V is one field per bond and J one coupling per bond, or one number for
    all of them. A bond with J=0 has no interaction and no constant."""
    V = np.atleast_1d(np.asarray(V))
    J = np.broadcast_to(np.asarray(J, dtype=np.float64), V.shape)
    keep = J != 0.0
    return N_SPIN*np.sum(np.abs(V[keep])**2/J[keep])


def band_grand_potential(es, T):
    """Grand potential of free fermions at the temperature T, with the
    energies es measured from the chemical potential,
    -T sum_n ln(1 + exp(-e_n/T)). Its derivative in an energy e_n is the
    Fermi-Dirac occupation of that level, and at T=0 it is the sum of
    the energies below zero, its limit."""
    es = np.asarray(es, dtype=np.float64)
    if T <= 0.0: return np.sum(np.minimum(es, 0.0))
    return -T*np.sum(np.logaddexp(0.0, -es/T))


def onsite_bonds(pairs, J):
    """The bonds (c, m, R, J_b) of the onsite model: the moment m of every
    Kondo pair coupled with J to the conduction electron of its own site"""
    return [(ci, m, (0, 0, 0), float(J)) for m, (ci, fi) in enumerate(pairs)]


def function_bonds(pairs, r, cells, J, rcut=None):
    """The bonds (c, m, R, J_b) of a coupling J(rc, rm) that depends on the
    position rc of the conduction site, in the cell R, and on the position
    rm of the site that holds the moment m, in the cell 0.

    pairs: the Kondo pairs (c_m, f_m), one per moment.
    r: positions of the conduction sites (the geometry of the conduction
        electrons, where the moment m sits on the site c_m).
    cells: the lattice vectors of the periodic directions (an empty list
        for a finite system).
    rcut: largest distance |rc - rm| of a bond (a bond at exactly rcut is
        kept). None keeps every pair of a finite system and means 5.0 for
        a periodic one, as for the pair interaction Vr of the electronic
        mean field.

    The two ends of a bond are different particles, so J(rc, rm) does not
    have to be the same as J(rm, rc). A bond with J_b=0 is dropped. They
    are sorted by moment, then by distance, so the onsite bond of a moment
    comes first."""
    if not callable(J):
        raise TypeError("J must be a number or a function J(rc, rm) of two "
                "positions; got "+type(J).__name__)
    r = np.array(r, dtype=np.float64)
    nd = len(cells)
    tol = 1e-6 # a shell sitting exactly at rcut is kept whole
    if rcut is None and nd > 0: rcut = 5.0
    if rcut is not None:
        rcut = float(rcut)
        if not rcut > 0.0:
            raise ValueError("rcut, the range of the Kondo coupling J, must "
                    "be positive; got "+str(rcut))
    if nd > 0: # every cell that can hold a bond within rcut
        A = np.array(cells, dtype=np.float64)
        B = np.linalg.pinv(A) # cell index of a vector R is R@B
        span = np.max(np.linalg.norm(r[:, None, :]-r[None, :, :], axis=2))
        ncells = int(np.ceil((rcut+span)*np.max(np.linalg.norm(B, axis=0))))+1
        grid = np.meshgrid(*[np.arange(-ncells, ncells+1)]*nd, indexing="ij")
        ns = np.array([g.ravel() for g in grid]).T
    else: ns = np.zeros((1, 0), dtype=int)
    rm = r[[cm for (cm, fm) in pairs]] # where the moments sit
    out = []
    for n in ns:
        R = tuple(int(x) for x in n) + (0,)*(3-nd)
        shift = np.zeros(3) if nd == 0 else n@np.array(cells, dtype=float)
        rc = r + shift
        dist = np.linalg.norm(rc[:, None, :] - rm[None, :, :], axis=2)
        keep = np.ones(dist.shape, dtype=bool) if rcut is None else \
                dist <= rcut + tol
        for c, m in zip(*np.nonzero(keep)): # J only where a bond can be
            Jb = J(rc[c], rm[m])
            if not np.isfinite(Jb) or np.imag(Jb) != 0.0:
                raise ValueError("the Kondo coupling J(rc, rm) must be a "
                        "real and finite number; got %s for rc=%s, rm=%s"
                        % (Jb, rc[c], rm[m]))
            Jb = float(np.real(Jb))
            if Jb != 0.0: out.append((dist[c, m], int(m), int(c), R, Jb))
    out.sort(key=lambda b: (b[1], round(b[0], 8), b[2], b[3]))
    return [(c, m, R, Jb) for (dist, m, c, R, Jb) in out]


def _minus(R): return tuple(-int(x) for x in R)


def _seed(mf, nb, npairs):
    """The starting V (one per bond) and lam (one per moment). A number is
    the same value on every bond or every moment."""
    if mf is None:
        return np.zeros(nb, dtype=np.complex128), np.zeros(npairs)
    V, lam = mf
    out = []
    for (x, n, name, what, dtype) in ((V, nb, "V", "bond", np.complex128),
            (lam, npairs, "lam", "moment", np.float64)):
        x = np.asarray(x, dtype=dtype)
        if x.ndim == 0: x = np.full(n, x, dtype=dtype)
        if x.shape != (n,):
            raise ValueError("the seed mf=(V, lam) needs one %s per %s, %d "
                    "of them here, or a number for all; got %s of shape %s"
                    % (name, what, n, name, x.shape))
        out.append(x.copy())
    return tuple(out)


def kondo_bonds(h0, pairs, J, rcut=None):
    """The Kondo bonds (c, m, R, J_b) of a coupling J: a number is the
    onsite coupling of every moment (onsite_bonds), and a function J(rc, rm)
    of the position of the conduction site and of the site that holds the
    moment gives every bond up to rcut (function_bonds). c is a conduction
    site of h0, m the index of a Kondo pair, R the cell of c."""
    if not callable(J):
        if rcut is not None:
            raise ValueError("rcut is the range of a Kondo coupling given as "
                    "a function J(rc, rm); with a number J the coupling is "
                    "onsite and has no range, so leave rcut out; got "
                    "rcut="+str(rcut))
        return onsite_bonds(pairs, J)
    g = h0.geometry
    fs = {fi for (ci, fi) in pairs}
    cs = [i for i in range(len(g.r)) if i not in fs] # the conduction sites
    where = {c: i for i, c in enumerate(cs)}
    local = [(where[ci], fi) for (ci, fi) in pairs]
    cells = [g.a1, g.a2, g.a3][:g.dimensionality]
    bonds = function_bonds(local, np.array(g.r)[cs], cells, J, rcut=rcut)
    return [(cs[c], m, R, Jb) for (c, m, R, Jb) in bonds]


class KondoLatticeSCF():
    """Result of kondo_lattice_mean_field: .hamiltonian, .converged,
    .total_energy (the grand potential per unit cell at the fixed mu and
    at the smearing T, see _pack), .hybridization (V_b, one per bond of
    .kondo_bonds),
    .kondo_bonds (the bonds (c, m, R, J_b)), .local_occupation (<n_f,j>,
    one per f-site), .constraint_lambda (lam_j, one per f-site)."""
    pass


def kondo_lattice_mean_field(h0, pairs, J=1.0, rcut=None, filling=0.5,
        mf=None, nk=8, mix=0.3, maxerror=1e-5, maxite=2000, T=2e-2):
    """Self-consistently solve the mean-field Hamiltonian above.

    h0: bare Hamiltonian (conduction dispersion + zero-hopping f-sites,
        has_spin=True, dense).
    pairs: list of (c_index, f_index) geometry-site pairs (0-indexed, not
        spin-orbital indices), one per moment: the f site and the
        conduction site it sits on.
    J: Coqblin-Schrieffer coupling (Eq. 73's J, entering as J/N with N=2 --
        NOT the same normalization as a bare J S_j.s_j Heisenberg-form
        Kondo coupling, see kondolattice.py's module docstring). A number
        is the onsite coupling of every moment. A function J(rc, rm) of the
        position rc of a conduction site and the position rm of the site
        that holds a moment is the coupling of that bond, for every bond
        up to the distance rcut, in this cell or in any other (see
        kondo_bonds and the module docstring).
    rcut: range of a J given as a function; None keeps every pair of a
        finite system and means 5.0 for a periodic one.
    mf: seed (V, lam), with one V per bond (in the order of kondo_bonds)
        and one lam per moment, or a number for all of them.
    filling: filling of the BARE (V=0, lam=0) conduction+f system
        (Hamiltonian.get_fermi4filling's convention: fraction of all
        spin-orbitals occupied), used ONCE to fix a lattice-wide chemical
        potential mu before the SCF loop starts. mu is then held fixed for
        every iteration -- NOT re-solved for the target filling at each
        step -- matching Coleman's own construction (Eq. 83 is a fixed-mu,
        grand-canonical Hamiltonian with no filling constraint at all; Eq.
        91-92 explicitly notes the Fermi *volume* expands to Q+n_c once
        hybridization turns on, i.e. the electron count is meant to float
        as V,lam converge, not be re-pinned to its bare-band value every
        iteration -- re-solving mu each step fights this expansion and
        produces a marginally-stable, weakly-damped oscillation in the
        coupled (V, lam, mu) system instead of converging). The converged
        electron count is generally not exactly `filling` any more, and
        the total energy is the grand potential at this mu and at the
        smearing T (see _pack), so two solutions at the same mu, for
        instance V=0 and a hybridized one, compare by it directly.

        Note that the bare (V=0) f-sector is a perfectly flat,
        macroscopically degenerate band at energy lam -- a filling target
        that lands mu inside that degenerate block (rather than in the
        dispersing conduction continuum) gives an ill-defined "which
        conduction states are filled" starting point; pick filling well
        away from the conduction band's own filling fraction if that
        happens (get_fermi4filling(0.0) vs get_fermi4filling(1.0) bracket
        the degenerate block, which is typically a wide plateau in filling
        space -- e.g. filling=0.5 is the flat block's midpoint for a single
        conduction orbital per f-site, and mu pins to lam=0 for any filling
        in between).

    T: Fermi-Dirac smearing for the density matrix AND for the lam
        (Lagrange multiplier) proportional update below -- deliberately
        NOT the tiny (~1e-7) smearing used elsewhere in this codebase's SCF
        loops. lam's local susceptibility d<n_f>/dlam scales like 1/(4T)
        for a resonance broadened only by T (V itself can be arbitrarily
        small deep in the weak-coupling Kondo regime this class exists to
        explore, so it cannot be relied on to provide that broadening); a
        naive lam += mix*(n_f-Q) step (as scftk.spinspin's
        per-site filling branch uses for the Heisenberg/RVB case, where the
        RVB bond order chi always provides an O(J) broadening) is then a
        proportional controller with gain mix/(4T), unstable for any
        mix>~8T -- observed directly as n_f oscillating between 0 and 2
        every iteration, never approaching 1, when T=1e-7 was tried here.
        Scaling the step by T below keeps the effective gain ~mix
        regardless of T, restoring the stability the plain update has in
        the RVB case. The converged Hamiltonian keeps T and nk, as
        h.temperature and h.nk like the other mean-field loops, so its
        density matrix and its filling are read at the T of the loop
        (densitymatrix.default_smearing) and n_f read from it is the
        constrained one.
    """
    Q = 1.0 # target f-occupancy for a spin-1/2 (N=2) local moment
    bonds = kondo_bonds(h0, pairs, J, rcut=rcut)
    cb = np.array([b[0] for b in bonds], dtype=int) # conduction site
    fb = np.array([pairs[b[1]][1] for b in bonds], dtype=int) # its f site
    Rb = [tuple(int(x) for x in b[2]) for b in bonds] # cell of the c site
    Jg = np.array([b[3] for b in bonds], dtype=np.float64)/2.0 # J/N, N=2
    h1 = h0.get_dense()
    h1.nk = nk # the mesh and the smearing of the loop, which the
    h1.temperature = T # observables of the converged h read it at
    hop0 = h1.get_dict()
    mu = h1.get_fermi4filling(filling, nk=nk) # fixed once, from the bare bands
    V, lam = _seed(mf, len(bonds), len(pairs))
    # the density matrix of direction -R holds the bond of the cell R
    ds = [(0, 0, 0)] + sorted({_minus(R) for R in Rb} - {(0, 0, 0)})

    def build_extra(V, lam):
        out = {d: np.zeros(h1.intra.shape, dtype=np.complex128)
                for d in set(ds) | set(Rb)}
        for idx, (ci, fi) in enumerate(pairs):
            for s in (0, 1): # the constraint, on the two spin components
                out[(0, 0, 0)][2*fi+s, 2*fi+s] += lam[idx]
        # V_b f^dagger_{f,0} c_{c,R} is the element [f, c] of the hopping
        # of direction R (a matrix m[d] of the Hamiltonian couples the
        # orbital i of the cell 0 to the orbital j of the cell d, as in
        # the Fock term of densitydensity.get_mf_normal), and its
        # conjugate c^dagger_{c,R} f_{f,0} = c^dagger_{c,0} f_{f,-R} the
        # element [c, f] of direction -R. For R=0 both land on the
        # intracell matrix
        for b in range(len(bonds)):
            for s in (0, 1): # spin components
                cc, ff = 2*cb[b] + s, 2*fb[b] + s
                out[Rb[b]][ff, cc] += V[b]
                out[_minus(Rb[b])][cc, ff] += np.conjugate(V[b])
        return out

    def evaluate(V, lam):
        hop = MultiHopping(hop0) + MultiHopping(build_extra(V, lam))
        h = h1.copy()
        h.set_multihopping(hop)
        h.fermi = mu
        h.shift_fermi(-mu)
        dms = h.get_density_matrix(nk=nk, T=T, ds=ds)
        dm = dms[(0, 0, 0)]
        nfocc = np.array([(dm[2*fi, 2*fi] + dm[2*fi+1, 2*fi+1]).real
                for (ci, fi) in pairs])
        # sum_s <c^dagger_{c,R,s} f_{f,0,s}>: get_density_matrix returns
        # the transpose of the density matrix, dm[d][i,j] =
        # <c^dagger_{i,0} c_{j,d}>, and the bond is
        # <c^dagger_{c,0} f_{f,-R}> by translation, the element [c,f] of
        # the direction -R (the Fock term of densitydensity.get_mf_normal
        # reads the bond of v[d][i,j] from dm[-d][j,i] the same way). With
        # [f,c] the target is conj(V), which is the same for a real V,
        # relaxes a complex seed to a real one, and is not a solution when
        # the phases of V cannot be removed (a complex hopping between the
        # moments)
        A = np.array([dms[_minus(Rb[b])][2*cb[b], 2*fb[b]]
                + dms[_minus(Rb[b])][2*cb[b]+1, 2*fb[b]+1]
                for b in range(len(bonds))], dtype=np.complex128)
        return h, A, nfocc

    def residual(V, A, nfocc):
        # V's residual is RELATIVE (to max(|V|,|target|), not an absolute
        # tolerance: V=0 is always a self-consistent solution (Eq. 77, like
        # the trivial BCS gap-equation root), so a seed that is subcritical
        # for this J decays geometrically toward it, V_target=c*V with
        # |c|<1 -- an ABSOLUTE |V_target-V|<maxerror check then declares
        # false convergence as soon as the still-decaying |V| itself
        # happens to be small (observed directly: at J values just below a
        # sharp hybridization threshold, this reported a converged but
        # meaningless V~1e-6, changing with maxerror/maxite rather than
        # reflecting any actual fixed point). The relative version's
        # residual |c*V-V|/max(|V|,|c*V|) = |1-c|/max(1,|c|) stays constant
        # under this decay (V cancels out), so it only reports converged
        # when V has genuinely stopped moving relative to its own size --
        # a decaying-to-zero trajectory instead correctly exhausts maxite
        # and reports non-convergence (the caller can re-run unseeded to
        # get the exact V=0 answer directly, or increase maxite).
        #
        # Each field is relative to its own size, down to a floor that
        # follows the largest field, for a field that vanishes by symmetry
        # (see field_residual)
        v_resid = field_residual(V, -Jg*A, maxerror)
        return max(v_resid, np.max(np.abs(nfocc - Q)))

    def pack(h, V, lam, nfocc, converged):
        scf = _pack(h, V, lam, nfocc, 2.0*Jg, Q, nk, T, converged)
        scf.kondo_bonds = list(bonds)
        return scf

    ite = 0
    while True:
        h, A, nfocc = evaluate(V, lam)
        diff = residual(V, A, nfocc)
        Vnew = (1 - mix)*V + mix*(-Jg*A)
        lamnew = lam + 4*mix*T*(nfocc - Q) # see T's docstring above
        if diff < maxerror:
            V, lam = Vnew, lamnew
            h, A, nfocc = evaluate(V, lam)
            err = residual(V, A, nfocc)
            scf = pack(h, V, lam, nfocc, err < maxerror)
            scf.iterations, scf.residual, scf.tol = ite, err, maxerror
            return scf
        V, lam = Vnew, lamnew
        if maxite is not None and ite >= maxite:
            scf = pack(h, V, lam, nfocc, False)
            scf.iterations, scf.residual, scf.tol = ite, diff, maxerror
            return scf
        ite += 1


def _pack(h, V, lam, nfocc, J, Q, nk, T, converged):
    scf = KondoLatticeSCF()
    scf.hamiltonian = h
    scf.converged = converged
    scf.hybridization = V
    scf.local_occupation = nfocc
    scf.constraint_lambda = lam
    # The total energy is the grand potential per unit cell at the fixed
    # mu and at the smearing T of the loop,
    #
    #   Omega = -T (1/Nk) sum_kn ln(1 + exp(-(e_kn - mu)/T))
    #           + sum_b N|V_b|^2/J_b - sum_j lam_j Q ,
    #
    # with e_kn the bands of the mean-field Hamiltonian (h is already
    # shifted by -mu). The loop occupies the states with the Fermi-Dirac
    # weight at T, and the derivative of the band term in V_b is that
    # same weighted <dH/dV_b>, so Omega is stationary in every V_b at the
    # fixed point, and in lam_j once n_f = Q. As a result its derivative
    # in mu along the converged solutions is minus the electron count,
    # and two solutions at the same mu and T compare by it. The constants
    # are those of Eq. 83: N|V|^2/J from the Hubbard-Stratonovich
    # transform, Eq. 78 (see hs_constant for why the N is there), and
    # -lam Q from lam (n_f - Q), Eq. 81. They multiply the identity, so
    # the one-body matrix does not contain them. No other double-counting
    # term is needed: unlike a generic V/U density-density interaction,
    # the Coqblin-Schrieffer coupling only enters through this one
    # Hubbard-Stratonovich channel.
    from ..htk.eigenvectors import peigvalsh_bloch
    ks = np.array(h.geometry.get_kmesh(nk=nk)) # the mesh of the loop
    es = peigvalsh_bloch(h.get_hk_gen(), ks) # bands measured from mu
    etot = band_grand_potential(es, T)/len(ks)
    etot += hs_constant(V, J) - np.sum(lam)*Q
    scf.total_energy = float(etot)
    return scf
