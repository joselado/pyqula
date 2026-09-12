import numpy as np
import pytest

from pyqula import geometry
from pyqula import ldos
from pyqula import algebra

# multi_ldos_tb accumulated the Lorentzian-weighted density over every
# eigenstate of every k-point and divided only by pi, never by the number
# of k-points, so every MULTILDOS/LDOS_*.OUT map was nk^dim times the
# physical LDOS; the DOS.OUT written into the same directory by the same
# call went through calculate_dos raw, so it was a further factor of pi
# off. The two quantities written side by side therefore disagreed with
# each other by exactly pi.
#
# Nothing here is a recorded number. Both assertions are against another
# routine of this library that computes the same quantity on the same
# k-mesh: h.get_ldos (which averages over k and divides by pi in
# ldoswaves.ldos_waves_jit) and h.get_dos (which does the same in
# dos.dos_kmesh).

E = 0.4
DELTA = 0.05
NK = 12


def _h():
    return geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)


def _written_map(e):
    return np.atleast_2d(np.genfromtxt("MULTILDOS/LDOS_"+str(e)+"_.OUT"))[:,2]


def test_the_written_map_is_the_ldos_at_that_energy(tmp_path, monkeypatch):
    """Same energy, same delta, same klist.kmesh -- so this is an exact
    identity, not a statistical one."""
    monkeypatch.chdir(tmp_path)
    h = _h()
    ref = np.array(h.get_ldos(e=E, delta=DELTA, mode="arpack", nk=NK,
                              nrep=1, write=False)[2])
    h.get_multildos(energies=np.array([E]), delta=DELTA, nrep=1, nk=NK)
    assert np.max(np.abs(_written_map(E)-ref)) < 1e-10*np.max(np.abs(ref))


def test_the_written_dos_is_the_dos(tmp_path, monkeypatch):
    """MULTILDOS/DOS.OUT is built from the same eigenvalues on the same
    mesh as h.get_dos(mode='ED'), on the refined grid multi_ldos_tb uses
    internally."""
    monkeypatch.chdir(tmp_path)
    h = _h()
    energies = np.linspace(-1.0, 1.0, 10)
    es2 = np.linspace(min(energies), max(energies), len(energies)*10)
    ref = h.get_dos(energies=es2, delta=DELTA, nk=NK, write=False)[1]
    h.get_multildos(energies=energies, delta=DELTA, nrep=1, nk=NK)
    out = np.genfromtxt("MULTILDOS/DOS.OUT").T
    assert np.max(np.abs(out[0]-es2)) < 1e-12  # same energy grid
    assert np.max(np.abs(out[1]-ref)) < 1e-10*np.max(np.abs(ref))


def test_the_map_and_the_dos_agree_with_each_other(tmp_path, monkeypatch):
    """The headline of the finding: summing the LDOS map over the sites of
    the unit cell is the total DOS, so the two files written into the same
    directory by one call have to agree where their grids meet. They were
    a factor of pi apart."""
    monkeypatch.chdir(tmp_path)
    h = _h()
    energies = np.linspace(-1.0, 1.0, 10)
    h.get_multildos(energies=energies, delta=DELTA, nrep=1, nk=NK)
    out = np.genfromtxt("MULTILDOS/DOS.OUT").T
    for e in energies:
        total = np.sum(_written_map(e))  # LDOS summed over the cell
        ie = np.argmin(np.abs(out[0]-e))
        assert abs(total-out[1][ie]) < 1e-8*abs(total)


def test_the_map_does_not_depend_on_the_k_mesh(tmp_path, monkeypatch):
    """A local density of states is an intensive quantity: refining the
    Brillouin-zone mesh has to leave it alone, up to the residual
    discretization of the Lorentzian. Without the 1/nk it grew by the
    ratio of the mesh sizes -- a factor of 4 between these two."""
    monkeypatch.chdir(tmp_path)
    h = _h()
    delta = 0.3  # broad enough that both meshes resolve the same profile
    h.get_multildos(energies=np.array([E]), delta=delta, nrep=1, nk=20)
    coarse = _written_map(E)
    h.get_multildos(energies=np.array([E]), delta=delta, nrep=1, nk=40)
    fine = _written_map(E)
    assert np.max(np.abs(coarse-fine)) < 0.02*np.max(np.abs(fine))


def test_the_operator_is_honoured(tmp_path, monkeypatch):
    """multi_ldos_tb named its operator argument `op` and carried a
    **kwargs that absorbed anything else, so get_multildos(operator=...)
    was silently identical to the plain call, while every other LDOS/DOS
    entry point in the library spells it `operator`."""
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_zeeman([0., 0., 0.4])
    args = dict(energies=np.array([E]), delta=DELTA, nrep=1, nk=6)
    h.get_multildos(**args)
    plain = _written_map(E)
    h.get_multildos(operator="sz", **args)
    sz = _written_map(E)
    assert np.max(np.abs(plain-sz)) > 1e-3
    # the sz-resolved LDOS is what get_ldos returns for the same operator
    ref = np.array(h.get_ldos(e=E, delta=DELTA, mode="arpack", nk=6,
                              nrep=1, write=False, operator="sz")[2])
    assert np.max(np.abs(sz-ref)) < 1e-10*np.max(np.abs(ref))


def test_the_old_op_spelling_is_still_accepted(tmp_path, monkeypatch):
    """`op` was the only spelling that worked, and examples use it."""
    monkeypatch.chdir(tmp_path)
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_zeeman([0., 0., 0.4])
    args = dict(energies=np.array([E]), delta=DELTA, nrep=1, nk=6)
    h.get_multildos(op="sz", **args)
    viaop = _written_map(E)
    h.get_multildos(operator="sz", **args)
    assert np.max(np.abs(viaop-_written_map(E))) < 1e-14
    with pytest.raises(TypeError):  # but not both at once
        h.get_multildos(op="sz", operator="sz", **args)


def test_a_typo_is_not_swallowed(tmp_path, monkeypatch):
    """The **kwargs that made the operator mismatch silent also swallowed
    every other misspelled argument."""
    monkeypatch.chdir(tmp_path)
    h = _h()
    with pytest.raises(TypeError):
        h.get_multildos(energies=np.array([E]), delats=DELTA)


def test_no_per_kpoint_diagonalization_call(tmp_path, monkeypatch):
    """The dense branch used to call algebra.eigh once per k-point, and
    dispatched the energy loop over a parallel.pcall process pool. Both
    are gone: the k-points go through the batched numba eigh that dos.py
    and ldosmap already use, and the per-eigenstate accumulation is one
    matmul."""
    monkeypatch.chdir(tmp_path)
    calls = []
    orig = algebra.eigh
    monkeypatch.setattr(algebra, "eigh", lambda *a, **kw: calls.append(1)
                        or orig(*a, **kw))
    h = _h()
    h.get_multildos(energies=np.array([E]), delta=DELTA, nrep=1, nk=NK)
    assert len(calls) == 0
