import numpy as np
import pytest

from pyqula import geometry, green

# green.green_operator -- the engine behind h.get_dos(mode="Green"/"RG")
# -- declared nk and then called bloch_selfenergy without it, so the
# k-mesh the caller asked for never reached the Brillouin-zone sum and
# h.get_dos(mode="Green",nk=4) and nk=80 returned byte-identical arrays.


def _honeycomb():
    return geometry.honeycomb_lattice().get_hamiltonian()


def test_green_mode_honours_nk(tmp_path, monkeypatch):
    """With gmode="full" the self-energy is an explicit nk x nk sum over
    the Brillouin zone, so the answer has to move with nk and has to
    converge to the exact-diagonalization DOS, which is the same integral
    evaluated by a different algorithm."""
    monkeypatch.chdir(tmp_path) # the Green branch always writes DOS.OUT
    h = _honeycomb()
    energies = np.linspace(-1., 1., 5)
    kw = dict(mode="Green", gmode="full", energies=energies, delta=0.2)
    (_, coarse) = h.get_dos(nk=2, **kw)
    (_, fine) = h.get_dos(nk=40, **kw)
    coarse, fine = np.array(coarse), np.array(fine)
    assert np.linalg.norm(fine - coarse)/np.linalg.norm(fine) > 0.1
    (_, ed) = h.get_dos(mode="ED", energies=energies, delta=0.2, nk=40,
                        write=False)
    ed = np.array(ed)
    assert np.linalg.norm(fine - ed)/np.linalg.norm(ed) < 0.05


def test_green_operator_forwards_nk_to_the_selfenergy():
    """The same statement one level down, without the DOS wrapper: the
    value green_operator returns must be the trace of the self-energy
    Green's function computed on the mesh it was given."""
    h = _honeycomb()
    for nk in [3, 25]:
        out = green.green_operator(h, e=0.3, delta=0.2, nk=nk, gmode="full")
        g = green.bloch_selfenergy(h.get_dense(), energy=0.3, delta=0.2,
                                   nk=nk, mode="full")[0]
        assert abs(out - (-np.trace(np.array(g)).imag)) < 1e-10


def test_unknown_dos_mode_lists_every_accepted_mode():
    """dos.get_dos_general accepts ED, KPM, adaptive, Green and RG, but
    its guard used to name only the first three -- so a user who typed
    'green' was told green is not a mode. The accepted list in the
    message is the elif chain right above it."""
    h = _honeycomb()
    with pytest.raises(ValueError) as info:
        h.get_dos(mode="bogus", energies=np.linspace(-1., 1., 4))
    message = str(info.value)
    for mode in ["ED", "KPM", "adaptive", "Green", "RG"]:
        assert mode in message, mode
