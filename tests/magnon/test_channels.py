"""The spin-flip restriction of the pair basis, and what it may not change.

Restricting the electron-hole pair basis to the magnon block is an
optimization -- it cuts the matrix dimension by about two and the dense
solve by about eight -- so every energy it returns must also be in the
spectrum of the unrestricted problem. It is also the part of the module
easiest to get subtly wrong, because the two halves of the Casida matrix
need DIFFERENT subsets of the pairs (see spinflip_masks), so it is checked
against the unrestricted answer on three states with different spin
structure rather than on one.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.bsetk import spinflip
from pyqula.meanfield import VJinteraction

NK = 6


def _assert_subset(h, Q, tol=1e-10):
    """Every restricted magnon energy must appear in the unrestricted
    spectrum, and there must be fewer of them"""
    a = np.sort(h.get_magnon_energies(nk=NK, Q=Q, channel="auto").real)
    b = np.sort(h.get_magnon_energies(nk=NK, Q=Q, channel="all").real)
    assert len(a) <= len(b)
    worst = max(np.min(np.abs(b - x)) for x in a)
    assert worst < tol, f"restricted energy missing from the full spectrum: {worst}"
    return a, b


def _neel_honeycomb():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    return h.get_mean_field_hamiltonian(U=3.0, filling=0.5, mf="antiferro",
                                        nk=NK, maxerror=1e-10)


def test_restricted_spectrum_is_a_subset_for_an_antiferromagnet():
    a, b = _assert_subset(_neel_honeycomb(), [0.1, 0., 0.])
    assert 4*len(a) == len(b)  # one of the four spin sectors of the pair basis


def test_restricted_spectrum_is_a_subset_with_a_neighbor_shell_interaction():
    g = geometry.honeycomb_lattice()
    scf = VJinteraction(g.get_hamiltonian(), U=3.0, V1=0.5, filling=0.5,
                         mf="antiferro", nk=NK, maxerror=1e-10, mix=0.3,
                         maxite=2000)
    _assert_subset(scf.hamiltonian, [0.1, 0., 0.])


def test_restricted_and_full_agree_for_a_saturated_ferromagnet():
    """The case that catches a wrong branch selection. Here the two halves
    of the Casida matrix are not mirror images of each other -- the
    opposite spin channel is empty -- so the excitation and de-excitation
    branches interleave instead of separating at zero, and picking "the
    highest half of the spectrum" silently returns the wrong sign. The
    conserved norm ||X||^2-||Y||^2 is what separates them; this state has
    genuinely negative magnon energies (it is an unstable saddle) so the
    sign is not decoration."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    mf = h.copy()
    mf.add_exchange([0., 0., 3.0])
    hmf = h.get_mean_field_hamiltonian(U=10.0, filling=0.5, mf=mf, nk=NK,
                                       maxerror=1e-10)
    a, b = _assert_subset(hmf, [0.1, 0., 0.])
    assert len(a) == len(b)  # only one spin sector exists here
    assert np.min(a) < -1e-3  # unstable against a finite-Q spin wave


def test_spinflip_channel_is_refused_for_a_non_collinear_state():
    h = _neel_honeycomb()
    tilted = h.copy()
    tilted.global_spin_rotation(vector=[1., 0., 0.], angle=0.5)
    with pytest.raises(ValueError):
        spinflip.magnon_matrix(tilted, Q=[0., 0., 0.], nk=NK,
                               channel="spinflip")
    # ... while "auto" quietly keeps the whole basis instead of failing
    p = spinflip.magnon_matrix(tilted, Q=[0., 0., 0.], nk=NK, channel="auto")
    assert p.n1 == p.pb.npair


def test_degenerate_multiplets_do_not_hide_the_spin_flip_block():
    """A magnetic band structure has accidental degeneracies between an up
    and a down band, where the eigensolver returns arbitrary mixtures with
    no Sz character at all. Without rotating those multiplets into Sz
    eigenstates first, the restriction silently gives up and falls back to
    the whole basis -- correct, but eight times slower. nk=6 hits such a
    degeneracy on this state and nk=4 does not, which is exactly how
    subtle the effect is."""
    h = _neel_honeycomb()
    for nk in (4, 6):
        p = spinflip.magnon_matrix(h, Q=[0., 0., 0.], nk=nk)
        assert p.n1 == p.pb.npair//4, f"no spin-flip block found at nk={nk}"


def _crossing_spin_bands():
    """A gapped ionic chain whose two spin species hop differently and are
    split by a Zeeman term, so that the lowest band is up at k=0 and down
    at k=pi. The band index then does not carry the same spin at k and at
    k+Q, which a Neel or ferromagnetic test state never shows: there the
    spin of band n is the same everywhere in the Brillouin zone"""
    g = geometry.chain().supercell(2)
    h = g.get_hamiltonian()
    hop, onsite, zeeman = {0: 1.0, 1: 0.3}, [1.0, -1.0], 0.3
    h.intra = np.zeros((4, 4), dtype=np.complex128)
    h.inter = np.zeros((4, 4), dtype=np.complex128)
    for s in (0, 1): # up, down
        for i in (0, 1):
            h.intra[2*i+s, 2*i+s] = onsite[i] + zeeman*(1-2*s)
        h.intra[s, 2+s] = h.intra[2+s, s] = hop[s]
        h.inter[2+s, s] = hop[s] # site 1 to site 0 of the next cell
    return h


def test_restricted_spectrum_is_a_subset_when_band_spins_cross():
    """The de-excitation half of the Casida matrix needs the pairs whose
    OWN spin change is the opposite of the excitation one. That is the
    same set as reading it off the excitation pair only when every band
    keeps its spin between k and k+Q, so this case is what separates
    them; with the wrong set the restricted spectrum missed the exact one
    by 3e-4 here at Q=0.25"""
    from pyqula.bsetk.interaction import density_interaction
    h = _crossing_spin_bands()
    V = density_interaction(h, U=1.0)
    for Q in ([0.25, 0., 0.], [0.4, 0., 0.]):
        a = np.sort(h.get_magnon_energies(nk=8, Q=Q, V=V, channel="auto").real)
        b = np.sort(h.get_magnon_energies(nk=8, Q=Q, V=V, channel="all").real)
        assert len(a) < len(b)
        worst = max(np.min(np.abs(b - x)) for x in a)
        assert worst < 1e-10, f"restricted energy missing at Q={Q}: {worst}"
