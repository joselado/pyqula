"""KondoLatticeHamiltonian: Abrikosov-pseudofermion (Read-Newns) mean-field
theory for the Kondo lattice / periodic Anderson model -- the standard
minimal model of heavy-fermion compounds.

Each localized moment is represented as S_j = 1/2 f_j^dagger sigma f_j
(Abrikosov pseudofermions), subject to the hard local constraint
f_j^dagger f_j = 1, and exchange-coupled to a conduction electron at the
same site through the Coqblin-Schrieffer Kondo interaction. Following
P. Coleman, "Heavy Fermions: electrons at the edge of magnetism",
arXiv:cond-mat/0612006, Sec. III.C ("The Large N Kondo Lattice", Eq.
65-99), this is decoupled in the large-N (here N=2, spin-1/2) saddle-point
approximation into a self-consistent hybridization field
V_j = -(J/2) <f_j^dagger c_j> (a composite fermion, "half electron, half
spin-flip", Eq. 100-102) plus a Lagrange multiplier lam_j enforcing
<f_j^dagger f_j> = 1 -- see scftk/kondolattice.py for the SCF
loop and the exact equations it implements.

IMPORTANT normalization caveat: J here is Coleman's Coqblin-Schrieffer
coupling (Eq. 73, entering the interaction as J/N with N=2), not the
coefficient of a bare Heisenberg-form J S_j.s_j Kondo coupling -- the two
differ by an N-dependent numerical factor that Eq. 73-78 already fixes
unambiguously, so this class follows the paper's own convention exactly
rather than converting to a different one and risking a silent
normalization error.

Physically, the T=0 large-N theory predicts the textbook heavy-fermion band
structure: an indirect hybridization gap of width ~ T_K = D*exp(-1/(J*rho))
(Eq. 88-89) separating a renormalized/"heavy" quasiparticle band (effective
mass m*/m ~ 1 + q/(rho*T_K), Eq. 99) from the rest of the conduction band,
with the Kondo scale T_K exponentially sensitive to J*rho (rho = conduction
band density of states at the Fermi level) -- i.e. a hybridization that, in
principle, turns on continuously (if exponentially slowly) as soon as J>0.
This SCF loop, however, necessarily runs at a finite Fermi-Dirac smearing
T>0 (its `T` kwarg -- needed for numerical stability of the local
constraint's feedback, see scftk/kondolattice.py's docstring),
and observably (see tests/kondolattice/) shows a genuine finite-temperature
Kondo crossover instead of a continuously-vanishing gap: for J*rho too
small that T_K(J) < T, thermal smearing washes out the hybridization
entirely and V=0 is the ONLY self-consistent solution; the threshold J at
which a nonzero V first appears moves to smaller J as T is lowered
(consistent with, not contradicting, the T=0 exponential-onset picture
above). Right at and above that threshold, V jumps directly to an O(1)
value rather than growing continuously from zero -- e.g. for the 1D chain
in tests/kondolattice/, V=0 is the only solution up to J=1.0 (at T=0.02),
then a second, distinct solution appears abruptly at V~0.14 by J=1.05 --
so within this finite-T mean field the onset is first-order-like, with
V=0 and V!=0 coexisting as self-consistent solutions on either side of
(and, likely, exactly at) the threshold, rather than the T=0 theory's
single, continuously-connected branch.

V=0 is always itself a self-consistent solution of the saddle-point
equation (Eq. 77), exactly like the trivial root of the BCS gap equation --
an unseeded SCF run (the default, mf=None) starts there and stays there
even for a J that also supports a genuine hybridized state; a seed away
from zero (mf=(V, lam), see get_mean_field_hamiltonian below) is needed to
find that other state. Where both coexist, the hybridized state is the
true (lower-energy) ground state -- see
tests/kondolattice/test_kondolattice.py::
test_kondo_branch_has_lower_energy_than_trivial_branch, which also
exercises return_total_energy=True's Eq. 83 constant terms end to end.

Only the U(1) (hybridization-only) large-N saddle point is implemented --
this is the same approximation level Coleman's review presents as the
standard Kondo-lattice mean field theory (the Read-Newns path integral,
Eq. 80-81); it does not capture magnetism or superconductivity (both
explicitly listed among the large-N approach's known limitations in the
review's Sec. III.D), nor the combined Kondo-Heisenberg (RKKY + Kondo,
Eq. 133-134) model, which needs a Nambu-doubled SU(2) gauge theory and is
out of scope here."""
import numpy as np

from .hamiltonians import Hamiltonian, _mean_field_scf_result
from .htk import fusion
from .scftk.kondolattice import kondo_lattice_mean_field


class KondoLatticeHamiltonian(Hamiltonian):
    """A Hamiltonian for Abrikosov-pseudofermion mean-field theory of the
    Kondo lattice / periodic Anderson model, built from a conduction-
    electron Hamiltonian `hc`.

    A second, initially decoupled sublattice of localized f-sites (one per
    site of `hc`'s geometry, offset in z purely so the two sublattices do
    not spatially overlap) is fused on top of `hc` with zero bare hopping
    -- see the module docstring for the physics. The Kondo coupling is
    supplied entirely through get_mean_field_hamiltonian's `J` kwarg::

        from pyqula import geometry
        from pyqula.kondolattice import KondoLatticeHamiltonian
        gc = geometry.chain()
        hc = gc.get_hamiltonian(has_spin=True)   # conduction electrons
        h = KondoLatticeHamiltonian(hc)
        seed = ([0.3+0.0j], [0.0])   # (V, lam) -- see the module docstring
                                      # for why an unseeded run finds V=0
        h2 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=150, mf=seed)
        h2.local_occupation   # <n_f,j> per localized site, target 1.0
        h2.hybridization      # converged V_j per localized site
        h2.constraint_lambda  # converged per-site Lagrange multipliers

    `filling` sets a lattice-wide chemical potential ONCE, from the bare
    (V=0, lam=0) conduction+f bands, and holds it fixed through the SCF
    loop (Hamiltonian.get_fermi4filling's usual fraction-of-all-spin-
    orbitals convention, covering both the conduction and f sectors, but
    see scftk.kondolattice.kondo_lattice_mean_field's docstring
    for why it is not re-solved every iteration, and for its "avoid the
    bare f-sector's massively degenerate flat band" caveat); the local
    constraint on the f-sector is enforced separately and automatically,
    not through this kwarg. `mf=(V, lam)` restarts the SCF loop from a
    previous run's converged (or intermediate) `(h2.hybridization,
    h2.constraint_lambda)` pair instead of the default all-zero guess (see
    the module docstring for why a nonzero seed is generally needed at
    all). `nk=`, `mix=`, `maxerror=`, `maxite=`, `T=` are forwarded to the
    SCF loop (scftk.kondolattice.kondo_lattice_mean_field)
    unchanged.

    An external Zeeman/magnetic field couples EXACTLY to both fermion
    species here (the conduction electron and the localized moment
    S_j=1/2 f_j^dagger sigma f_j, already bilinear in f -- see the module
    docstring) -- it is a single-particle term, added via the inherited
    Hamiltonian.add_zeeman/add_exchange called on this instance BEFORE
    get_mean_field_hamiltonian, exactly like SpinonHamiltonian::

        h = KondoLatticeHamiltonian(hc)
        h.add_zeeman([0., 0., 0.05])  # or h.add_exchange([0.,0.,0.05])
        h2 = h.get_mean_field_hamiltonian(J=1.5, filling=0.15, nk=150, mf=seed)

    add_zeeman applies to EVERY site in this fused Hamiltonian's geometry,
    i.e. both the conduction and the f sublattice (offset in z, see
    __init__) -- pass a position-dependent callable instead of a constant
    vector to target only one of them. As with SpinonHamiltonian, the
    argument is the coefficient of sigma, not of S=sigma/2. The f-site
    local constraint (a total-occupation, not a spin, constraint) stays
    exact under a field. Genuine physics, not a bug: a field competes with
    the Kondo singlet, so |V| SHRINKS as the field grows at fixed J, and a
    strong enough field destroys the hybridized state entirely -- the SCF
    then correctly reports non-convergence (a `None` return), the same
    "decays toward the always-self-consistent V=0 branch" signal a
    subcritical J produces (see kondo_lattice_mean_field's `residual`
    docstring) -- not a smaller-but-nonzero V. See
    tests/kondolattice/test_kondolattice_zeeman.py."""

    def __init__(self, hc=None):
        super().__init__(None)
        if hc is None: return
        if hc.has_eh:
            raise ValueError("KondoLatticeHamiltonian does not support a "
                    "BdG (Nambu) conduction-electron Hamiltonian")
        hc = hc.copy()
        hc.turn_spinful()
        hc = hc.get_dense()
        nc = len(hc.geometry.r)
        gf = hc.geometry.copy()
        gf.r = gf.r.copy()
        gf.r[:, 2] += 1.0 # offset only so the two sublattices do not overlap
        gf.r2xyz()
        hf = gf.get_hamiltonian(has_spin=True, tij=[0.0]) # localized, no bare hopping
        h0 = fusion.hamiltonian_fusion(hc, hf)
        # re-run the base __init__ against the fused geometry -- the first
        # call above (geometry=None, before h0 existed) left
        # dimensionality/num_orbitals at their geometry=None defaults (0),
        # which silently collapses get_fermi4filling/eigenvalues etc. to a
        # single (Gamma-point-only) k-point for any dimensionality>0 system
        super().__init__(h0.geometry)
        self.set_multihopping(h0.get_multihopping())
        self.is_multicell = h0.is_multicell
        self._kondo_pairs = [(i, nc + i) for i in range(nc)]

    def get_mean_field_hamiltonian(self, return_total_energy=False, **kwargs):
        scf = kondo_lattice_mean_field(self, self._kondo_pairs, **kwargs)
        if scf.converged:
            scf.hamiltonian.local_occupation = scf.local_occupation
            scf.hamiltonian.hybridization = scf.hybridization
            scf.hamiltonian.constraint_lambda = scf.constraint_lambda
        return _mean_field_scf_result(scf, return_total_energy)
