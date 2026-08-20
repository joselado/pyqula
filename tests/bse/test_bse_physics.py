"""Physical behaviour of the exciton spectrum, and the guards around it."""
import numpy as np
import pytest

from testutils import gapped_honeycomb
from pyqula import geometry
from pyqula.bsetk.interaction import density_interaction

NK = 6


def _coulomb(e2):
    """A soft-cutoff Coulomb tail, the interaction that actually binds an
    exciton (a Hubbard U alone is too short ranged in 2D to bind one)"""
    return lambda r1, r2: e2 / np.sqrt((r1 - r2).dot(r1 - r2) + 0.25)


def test_binding_energy_grows_with_the_interaction():
    h = gapped_honeycomb(mass=1.0)
    bindings = [h.get_exciton_binding_energies(
                    V=density_interaction(h, Vr=_coulomb(e2)), nk=NK)[0].real
                for e2 in (0., 0.2, 0.4, 0.8)]
    assert abs(bindings[0]) < 1e-10  # no interaction, no binding
    assert np.all(np.diff(bindings) > 0.01)  # and it grows monotonically


def test_exciton_lies_below_the_single_particle_gap():
    h = gapped_honeycomb(mass=1.0)
    W = density_interaction(h, Vr=_coulomb(0.8))
    b = h.get_bse(V=W, nk=NK)
    assert b.get_energies()[0].real < np.min(b.pairs.dE)


def test_tda_approaches_the_full_bse_at_weak_coupling():
    """The Tamm-Dancoff approximation drops the coupling block B, whose
    effect is second order in the interaction, so the two must agree ever
    more closely as the interaction is turned down."""
    h = gapped_honeycomb(mass=1.0)
    errs = []
    for e2 in (0.8, 0.2):
        W = density_interaction(h, Vr=_coulomb(e2))
        ef = h.get_exciton_energies(V=W, nk=NK, tda=False)[0].real
        et = h.get_exciton_energies(V=W, nk=NK, tda=True)[0].real
        errs.append(abs(ef - et))
    assert errs[1] < errs[0] / 4.  # at least as fast as the coupling squared


def test_amplitudes_are_normalized():
    """The conserved norm of the linear-response problem is X^dag X -
    Y^dag Y, not X^dag X + Y^dag Y"""
    h = gapped_honeycomb(mass=1.0)
    W = density_interaction(h, Vr=_coulomb(0.4))
    b = h.get_bse(V=W, nk=NK)
    norm = (np.sum(np.abs(b.amplitudes) ** 2, axis=1)
            - np.sum(np.abs(b.amplitudesY) ** 2, axis=1))
    assert np.max(np.abs(norm - 1.)) < 1e-8


def test_spectrum_is_real_for_a_stable_reference():
    h = gapped_honeycomb(mass=1.0)
    W = density_interaction(h, Vr=_coulomb(0.4))
    es = h.get_exciton_energies(V=W, nk=NK)
    assert np.max(np.abs(np.imag(es))) < 1e-10


def test_mean_field_hamiltonian_carries_its_own_interaction():
    """The end-to-end path: converge a mean field, then ask it for its
    excitons without naming an interaction at all"""
    h0 = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h0.add_sublattice_imbalance(0.6)
    h = h0.get_mean_field_hamiltonian(U=1.5, filling=0.5, mf="antiferro",
                                      nk=6, maxerror=1e-6)
    es = h.get_exciton_energies(nk=4, n=4)
    assert len(es) == 4
    assert np.all(np.real(es) > 0.)


def test_nambu_hamiltonian_is_rejected():
    h = gapped_honeycomb(mass=1.0)
    h.add_swave(0.2)
    with pytest.raises(ValueError, match="Nambu"):
        h.get_bse(V={(0, 0, 0): np.zeros((h.intra.shape[0],) * 2,
                                         dtype=np.complex128)}, nk=2)


def test_metallic_reference_is_rejected():
    """A metal has no well-defined valence/conduction split: a half-filled
    chain has its band below the Fermi energy at some k-points and above it
    at others, so no electron-hole pair basis exists."""
    h = geometry.chain().get_hamiltonian(has_spin=False)
    h = h.get_multicell().get_dense()
    with pytest.raises(ValueError, match="not the same at every k-point"):
        h.get_bse(V={(0, 0, 0): np.zeros((h.intra.shape[0],) * 2,
                                         dtype=np.complex128)}, nk=8)


def test_oversized_calculation_is_refused_before_allocating():
    h = gapped_honeycomb(mass=1.0)
    zero = {(0, 0, 0): np.zeros((h.intra.shape[0],) * 2, dtype=np.complex128)}
    with pytest.raises(MemoryError, match="max_memory"):
        h.get_bse(V=zero, nk=60, max_memory=0.05)


def test_exchange_term_splits_singlet_from_triplet():
    """Spin is part of the orbital index, so a spinful calculation returns
    singlet and triplet excitons together. With a spin-rotation-invariant
    reference the lowest transition is four-fold degenerate without any
    interaction, and the full kernel must split it into a three-fold
    triplet and a single singlet pushed up above it by the (repulsive)
    exchange term.

    Neither kernel term does this on its own -- an onsite Hubbard U only
    couples opposite spins, so the direct term alone shifts two of the four
    and leaves the other two put, which is not an allowed spin multiplet at
    all. Recovering 3+1 is therefore a real check that the two terms are
    combined with the right relative weight and sign, not just that each is
    individually present."""
    g = geometry.chain().supercell(4)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    # a staggered onsite potential gaps the cluster without breaking
    # spin-rotation symmetry the way a Zeeman field would
    h.add_onsite(lambda r: 0.8 * (-1) ** int(round(r[0] - 1.5)))
    h = h.get_multicell().get_dense()
    W = density_interaction(h, U=1.5)

    bare = h.get_exciton_energies(V=W, nk=1, kernel="none").real
    assert np.max(np.abs(bare[:4] - bare[0])) < 1e-9  # four-fold to begin with
    assert bare[4] - bare[0] > 1e-2

    es = h.get_exciton_energies(V=W, nk=1, kernel="full").real
    triplet, singlet = es[:3], es[3]
    assert np.max(np.abs(triplet - triplet[0])) < 1e-9  # a clean three-fold
    assert singlet - triplet[0] > 1e-2                  # split off above it
    assert triplet[0] < bare[0]                         # and the triplet binds

    # guard: the direct term alone gives 2+2, not 3+1
    ed = h.get_exciton_energies(V=W, nk=1, kernel="direct").real
    assert abs(ed[2] - ed[0]) > 1e-2
