"""`remove_nambu` branched on spinful_nambu / spinful / spinless and had no
spinless_nambu branch, so every spinless BdG Hamiltonian -- the ordinary
`g.get_hamiltonian(has_spin=False); h.add_swave(d)` -- fell through to a
NotImplementedError.  That raise was inherited by everything built on
`h0.remove_nambu()`: get_anomalous_hamiltonian, extract("density"/"onsite"),
remove_pairing and the transport superconducting dispatch.

The tests below pin the invariant rather than the fix: stripping the Nambu
degree of freedom from a BdG Hamiltonian must give back exactly the normal
Hamiltonian it was built from, because `sctk.spinless.nambu` puts that
matrix in the electron block untouched."""
import numpy as np
import pytest

from pyqula import geometry


def spinless_bdg(delta=0.3, **kwargs):
    """A spinless BdG chain and the normal Hamiltonian behind it"""
    g = geometry.chain()
    h0 = g.get_hamiltonian(has_spin=False, **kwargs)
    h = h0.copy()
    h.add_swave(delta)
    return h0, h


def test_removing_nambu_gives_back_the_normal_hamiltonian():
    h0, h = spinless_bdg()
    assert h.check_mode("spinless_nambu")
    h.remove_nambu()
    assert h.check_mode("spinless")
    assert h.intra.shape[0] == h0.intra.shape[0]
    # exact, not approximate: the electron block of the Nambu matrix is the
    # normal matrix itself
    assert h.same_hamiltonian(h0)


def test_removing_nambu_keeps_a_sparse_hamiltonian_sparse():
    """is_sparse is a public get_hamiltonian keyword and add_swave keeps it,
    so the branch has to survive scipy sparse matrices."""
    h0, h = spinless_bdg(is_sparse=True)
    h.remove_nambu()
    assert h.check_mode("spinless")
    assert h.intra.shape[0] == h0.intra.shape[0]
    assert h.same_hamiltonian(h0)


def test_the_anomalous_hamiltonian_carries_the_pairing_and_nothing_else():
    """h.get_anomalous_hamiltonian() is `h - (h with the pairing removed)`.
    Its electron-electron block must vanish and its pairing block must be
    the pairing of the original."""
    h0, h = spinless_bdg(delta=0.3)
    ha = h.get_anomalous_hamiltonian()
    ma = np.array(ha.intra.todense()) if hasattr(ha.intra, "todense") \
            else np.array(ha.intra)
    m = np.array(h.intra.todense()) if hasattr(h.intra, "todense") \
            else np.array(h.intra)
    # electron (even) and hole (odd) sectors are empty
    assert np.allclose(ma[::2, ::2], 0., atol=1e-10), ma[::2, ::2]
    assert np.allclose(ma[1::2, 1::2], 0., atol=1e-10), ma[1::2, 1::2]
    # the anomalous sector is untouched
    assert np.allclose(ma[::2, 1::2], m[::2, 1::2], atol=1e-10)


def test_the_onsite_term_of_a_spinless_bdg_is_extractable():
    """extract("onsite") strips the Nambu block first, so it inherited the
    raise. Once the block is stripped it must report the onsite energies of
    the normal Hamiltonian the BdG one was built from -- the second code
    path computing the same quantity."""
    g = geometry.chain().get_supercell(3)
    h0 = g.get_hamiltonian(has_spin=False)
    h0.add_onsite(lambda r: 0.4*r[0])  # a non-uniform onsite potential
    h = h0.copy()
    h.add_swave(0.3)
    o = h.extract("onsite")
    assert len(o) == len(h.geometry.r)
    assert np.allclose(o, h0.extract("onsite"), atol=1e-10), o


def test_remove_pairing_returns_the_normal_bdg_hamiltonian():
    """remove_pairing goes through remove_nambu too. The result must still
    be a Nambu Hamiltonian, but with no anomalous block at all."""
    h0, h = spinless_bdg(delta=0.3)
    h.remove_pairing()
    assert h.check_mode("spinless_nambu")
    m = np.array(h.intra.todense()) if hasattr(h.intra, "todense") \
            else np.array(h.intra)
    assert np.allclose(m[::2, 1::2], 0., atol=1e-10), m[::2, 1::2]
    # and the normal part survived
    assert np.allclose(m[::2, ::2], np.array(h0.intra), atol=1e-10)


def test_a_spinless_bdg_junction_conducts_half_of_the_spinful_one():
    """End to end: transporttk/didv.py asks every lead for its anomalous
    Hamiltonian, so a spinless BdG junction used to crash where the
    identical spinful one returned a number.  The two are the same model up
    to spin degeneracy, so the spinful conductance must be exactly twice the
    spinless one -- an invariant of the model, not a recorded value."""
    from pyqula import heterostructures
    g = geometry.chain()

    def junction(has_spin):
        h1 = g.get_hamiltonian(has_spin=has_spin)
        h1.add_swave(0.2)
        h2 = g.get_hamiltonian(has_spin=has_spin)
        h2.add_swave(0.2)
        return heterostructures.build(h1, h2)

    for energy in [0.05, 0.3]:
        gless = junction(False).didv(energy=energy)
        gful = junction(True).didv(energy=energy)
        assert gless > 1e-6, gless
        assert abs(gful - 2.*gless) < 1e-8*max(1., abs(gful)), (gless, gful)
