"""SpinonHamiltonian: Abrikosov-pseudofermion (auxiliary-fermion) mean-field
theory for spin-1/2 Heisenberg models.

Represents each spin as S_i = 1/2 f_i^dagger sigma f_i (Abrikosov fermions),
subject to the hard local constraint f_i^dagger f_i = 1 (exactly one fermion
per site), and self-consistently decouples J S_i.S_j into an RVB bond order
parameter chi_ij = <f_i^dagger f_j> -- see Savary & Balents, "Quantum Spin
Liquids: a review", arXiv:1601.03742, Sec. 4 ("Partons") and Sec. 4.1 in
particular, whose Eq. 35/36/37/38 and surrounding text this class follows
directly. Physically this is exactly meanfield.VJinteraction's J-channel
(already the correct Fock/Hartree-Fock decoupling of a spin-spin exchange,
see selfconsistency/spinspin.py's module docstring) run on a Hamiltonian
with zero bare hopping (a pure spin model has no bare electron tunneling --
every "kinetic" term is the auxiliary mean field itself) and with the local
constraint enforced exactly, at every site, via the per-site `filling`
array VJinteraction now supports (selfconsistency.spinspin._run_anisotropic_
scf's array-filling branch) instead of only a single lattice-averaged Fermi
level.

Only the U(1) (RVB bond-only) ansatz is implemented -- a Z2 ansatz (also
allowing the pairing/anomalous channel J already induces on a BdG
Hamiltonian, see meanfield.VJinteraction's docstring) would need a
Nambu-doubled SpinonHamiltonian and a has_eh=True-aware local constraint,
neither of which exist yet (VJinteraction's array-filling path explicitly
raises NotImplementedError for has_eh=True -- see
selfconsistency.spinspin._run_anisotropic_scf)."""
import numpy as np

from .hamiltonians import Hamiltonian, _mean_field_scf_result
from .meanfield import VJinteraction


class SpinonHamiltonian(Hamiltonian):
    """A Hamiltonian for Abrikosov-pseudofermion mean-field theory of a
    spin-1/2 Heisenberg model on `geometry`.

    Built with zero bare hopping (see the module docstring) -- exchange
    couplings are supplied entirely through get_mean_field_hamiltonian's
    J1/J2/J3/Jr/J1x/J1y/J1z kwargs (meanfield.VJinteraction's own
    convention), e.g.::

        from pyqula import geometry
        from pyqula.spinon import SpinonHamiltonian
        g = geometry.triangular_lattice()
        h = SpinonHamiltonian(g)
        h2 = h.get_mean_field_hamiltonian(J1=1.0, nk=12)
        h2.local_occupation   # <n_i> per site, target is exactly 1.0
        h2.constraint_lambda  # converged per-site Lagrange multipliers

    `filling` cannot be overridden -- the auxiliary-fermion representation
    is only valid at exactly one fermion per site, everywhere, so this
    class always requests that (as the per-site array VJinteraction's
    array-filling path expects, one entry per site, in its 0-to-1
    fraction-of-2-orbital-capacity convention -- see
    densitymatrix.full_dm_accumulate_sparse_local_fermi's docstring), never
    a value the caller supplies. Any interaction/SCF kwarg
    get_mean_field_hamiltonian otherwise accepts (V1/V2/V3/U for an
    additional density-density term, mf=, nk=, mix=, maxerror=, maxite=,
    constrains=, ...) is forwarded unchanged; `scf.converged` (via
    get_mean_field_hamiltonian's usual None-on-non-convergence return, see
    hamiltonians._mean_field_scf_result) already implies the local
    constraint itself converged to within `maxerror`, since
    _run_anisotropic_scf folds the per-site occupation residual into the
    same convergence check as the mean field's own -- no separate
    tolerance to track.

    An external Zeeman/magnetic field couples EXACTLY (not via any
    mean-field decoupling) to the auxiliary fermion, since
    S_i = 1/2 f_i^dagger sigma f_i is itself already bilinear in f: physically
    it is just an extra single-particle term -h.S_i, i.e. -(h/2).sigma_i,
    added to the auxiliary Hamiltonian before the RVB exchange mean field is
    layered on top. Use the inherited Hamiltonian.add_zeeman/add_exchange
    (called on this instance BEFORE get_mean_field_hamiltonian, same as for
    any other Hamiltonian in this codebase) to add one::

        h = SpinonHamiltonian(g)
        h.add_zeeman([0., 0., 0.3])  # or h.add_exchange([0.,0.,0.3])
        h2 = h.get_mean_field_hamiltonian(J1=1.0, nk=12)

    Note add_zeeman/add_exchange's argument b is the coefficient of sigma
    (Pauli matrices, eigenvalues +-1), not of S=sigma/2 -- so the physical
    field h in H=-h.S_i is 2*b, matching this argument's convention
    everywhere else in pyqula (add_exchange/add_magnetism on an ordinary
    electronic Hamiltonian use the same sigma-not-S convention). The local
    constraint enforced by the array-filling machinery above (exactly one
    fermion per site, summed over spin) is unaffected by the field -- it is
    a total-occupation constraint, not a constraint on the spin
    polarization, so a finite field is free to induce a net <S_i> (up to
    full polarization at large field/J) while local_occupation stays
    exactly 1.0 (verified in tests/spinon/test_spinon_zeeman.py: an
    isotropic response to same-magnitude fields along different axes,
    saturation at large field, and constraint preservation throughout --
    see Savary & Balents, arXiv:1601.03742, Sec. 4.1, on the field response
    of a U(1) RVB spin liquid)."""

    def __init__(self, geometry=None):
        super().__init__(geometry)
        if geometry is not None:
            h0 = geometry.get_hamiltonian(tij=[0.0], has_spin=True)
            self.set_multihopping(h0.get_multihopping())
            self.is_multicell = h0.is_multicell

    def get_mean_field_hamiltonian(self, return_total_energy=False, **kwargs):
        if "filling" in kwargs:
            raise ValueError("SpinonHamiltonian enforces exactly one "
                    "Abrikosov fermion per site -- filling= is fixed by "
                    "the representation itself and cannot be overridden "
                    "(see the class docstring)")
        n = len(self.geometry.r)
        scf = VJinteraction(self, filling=np.full(n, 0.5), **kwargs)
        if scf.converged:
            # local_occupation in the <n_i> in [0,2] electron-count
            # convention (matching f_i^dagger f_i directly -- the target is
            # exactly 1.0), not VJinteraction's internal 0-to-1 fraction
            # one scf.local_occupation itself uses -- see
            # densitymatrix.full_dm_accumulate_sparse_local_fermi's
            # docstring for that distinction.
            scf.hamiltonian.local_occupation = 2.0*scf.local_occupation
            scf.hamiltonian.constraint_lambda = np.asarray(scf.lam)
        return _mean_field_scf_result(scf, return_total_energy)
