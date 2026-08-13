"""SCF engine behind KondoLatticeHamiltonian (see ../kondolattice.py for the
public class and the full physics writeup). Implements the large-N
Abrikosov-pseudofermion / Read-Newns mean-field theory of the Kondo lattice,
following P. Coleman, "Heavy Fermions: electrons at the edge of magnetism",
arXiv:cond-mat/0612006, Sec. III.C.2 ("Mean field theory of the Kondo
lattice") verbatim -- Eq. 65-99 there.

The Coqblin-Schrieffer Kondo interaction at site j, Eq. 73,

    H_I(j) = (J/N) S_ab(j) c^dagger_jb c_ja,   S_ab(j) = f^dagger_ja f_jb - (n_f/N) delta_ab

is Hubbard-Stratonovich decoupled (Eq. 76-78) into a hybridization field
V_j = -(J/N) <f^dagger_ja c_ja> (summed over a, the N=2 spin components for
a spin-1/2 local moment) coupling the localized f-pseudofermion to the
conduction electron at the same site, plus a Lagrange multiplier lam_j
enforcing the local constraint <f^dagger_ja f_ja> = Q (=1 for N=2), giving
the mean-field Hamiltonian (Eq. 81/83)

    H_MFT = sum_k eps_k c^dagger_k c_k
          + sum_j [ V_j^* c^dagger_j f_j + V_j f^dagger_j c_j + lam_j f^dagger_j f_j ]
          + sum_j [ |V_j|^2/J - lam_j Q ]

Both V_j and lam_j are kept per-site (arrays) rather than a single global
scalar, so a non-uniform (e.g. supercell) system self-consistently converges
to site-dependent values -- for a translationally invariant lattice they
converge to the same value on every site.

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


class KondoLatticeSCF():
    """Result of kondo_lattice_mean_field: .hamiltonian, .converged,
    .total_energy, .hybridization (V_j, one per f-site), .local_occupation
    (<n_f,j>, one per f-site), .constraint_lambda (lam_j, one per f-site)."""
    pass


def kondo_lattice_mean_field(h0, pairs, J=1.0, filling=0.5, mf=None, nk=8,
        mix=0.3, maxerror=1e-5, maxite=2000, T=2e-2):
    """Self-consistently solve the mean-field Hamiltonian above.

    h0: bare Hamiltonian (conduction dispersion + zero-hopping f-sites,
        has_spin=True, dense).
    pairs: list of (c_index, f_index) geometry-site pairs (0-indexed, not
        spin-orbital indices) -- the sites the Kondo coupling connects.
    J: Coqblin-Schrieffer coupling (Eq. 73's J, entering as J/N with N=2 --
        NOT the same normalization as a bare J S_j.s_j Heisenberg-form
        Kondo coupling, see kondolattice.py's module docstring).
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
        coupled (V, lam, mu) system instead of converging). The actual
        converged electron count (generally not exactly `filling` any
        more) is what determines the total energy's un-shift, see _pack.

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
        the RVB case.
    """
    Q = 1.0 # target f-occupancy for a spin-1/2 (N=2) local moment
    Jg = J/2.0 # J/N, N=2
    nf = len(pairs)
    h1 = h0.get_dense()
    hop0 = h1.get_dict()
    mu = h1.get_fermi4filling(filling, nk=nk) # fixed once, from the bare bands
    if mf is None:
        V = np.zeros(nf, dtype=np.complex128)
        lam = np.zeros(nf)
    else:
        V, lam = mf
        V = np.asarray(V, dtype=np.complex128)
        lam = np.asarray(lam, dtype=np.float64)

    def build_extra(V, lam):
        m = np.zeros(h1.intra.shape, dtype=np.complex128)
        for idx, (ci, fi) in enumerate(pairs):
            for s in (0, 1): # spin components
                cc, ff = 2*ci + s, 2*fi + s
                m[ff, ff] += lam[idx]
                m[cc, ff] += np.conjugate(V[idx])
                m[ff, cc] += V[idx]
        return {(0, 0, 0): m}

    def evaluate(V, lam):
        hop = MultiHopping(hop0) + MultiHopping(build_extra(V, lam))
        h = h1.copy()
        h.set_multihopping(hop)
        h.fermi = mu
        h.shift_fermi(-mu)
        dm = h.get_density_matrix(nk=nk, T=T, ds=[(0, 0, 0)])[(0, 0, 0)]
        nfocc = np.array([(dm[2*fi, 2*fi] + dm[2*fi+1, 2*fi+1]).real
                for (ci, fi) in pairs])
        A = np.array([dm[2*fi, 2*ci] + dm[2*fi+1, 2*ci+1]
                for (ci, fi) in pairs])
        return h, dm, A, nfocc

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
        target = -Jg*A
        vscale = np.maximum(np.abs(V), np.abs(target))
        vscale = np.where(vscale == 0.0, 1.0, vscale) # both exactly zero
        v_resid = np.max(np.abs(target - V)/vscale)
        return max(v_resid, np.max(np.abs(nfocc - Q)))

    ite = 0
    while True:
        h, dm, A, nfocc = evaluate(V, lam)
        diff = residual(V, A, nfocc)
        Vnew = (1 - mix)*V + mix*(-Jg*A)
        lamnew = lam + 4*mix*T*(nfocc - Q) # see T's docstring above
        if diff < maxerror:
            V, lam = Vnew, lamnew
            h, dm, A, nfocc = evaluate(V, lam)
            converged = residual(V, A, nfocc) < maxerror
            return _pack(h, dm, V, lam, nfocc, J, Q, nk, converged)
        V, lam = Vnew, lamnew
        if maxite is not None and ite >= maxite:
            return _pack(h, dm, V, lam, nfocc, J, Q, nk, False)
        ite += 1


def _pack(h, dm, V, lam, nfocc, J, Q, nk, converged):
    scf = KondoLatticeSCF()
    scf.hamiltonian = h
    scf.converged = converged
    scf.hybridization = V
    scf.local_occupation = nfocc
    scf.constraint_lambda = lam
    # Eq. 83's constant terms (V-bar*V/J from the Hubbard-Stratonovich
    # transform, Eq. 78, minus lam*Q from expanding lam*(n_f-Q), Eq. 81),
    # which the one-body matrix above does not contain (they multiply the
    # identity, not any operator) but which the free energy needs -- see
    # this module's docstring. No further double-counting correction is
    # needed beyond this: unlike a generic V/U density-density interaction,
    # the Coqblin-Schrieffer coupling here only ever enters through this one
    # Hubbard-Stratonovich channel.
    #
    # mu is now held fixed through the SCF loop (see kondo_lattice_mean_
    # field's docstring), so the converged electron count is NOT
    # n_orbitals*filling any more (that was only the bare-band count used
    # to pick mu in the first place) -- un-shift using the actual electron
    # count read off the converged (intracell) density matrix instead.
    n_electrons = np.trace(dm).real
    etot = h.get_total_energy(nk=nk) + h.fermi*n_electrons
    hs_term = np.sum(np.abs(V)**2)/J if J != 0.0 else 0.0
    etot += hs_term - np.sum(lam)*Q
    scf.total_energy = etot.real if hasattr(etot, "real") else etot
    return scf
