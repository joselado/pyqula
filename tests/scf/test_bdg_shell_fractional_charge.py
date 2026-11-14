"""The fill of the shell of a gapless Bogoliubov-de Gennes state
(scftk.superscf.ShellFilling) holds the requested charge also when the
vectors of the shell carry a fractional charge, as they do when the
pairing is still alive and the edge of the shell cuts an electron-hole
pair. The root equation of the fill once counted the charge of the
emptied shell twice: the density matrix then held N0 minus that charge,
and close to the bottom of the jump the bracket of the root never closed,
so the fill did nothing. The shells here are built by hand on one Nambu
site, whose electron components are the first two."""
import numpy as np
import pytest

from pyqula.scftk import superscf

TAU = np.array([1., 1., -1., -1.]) # electron and hole components


def _shell(charge0, levels, nkpoints, tol=4e-3):
    """an armed ShellFilling at the requested charge with the given levels,
    (k, energies, vectors as columns, occupations) per k-point"""
    sf = superscf.ShellFilling(charge0)
    sf.nkpoints, sf.tol = nkpoints, tol
    sf.ts = sf.tol/sf.width
    sf.tau = TAU
    sf.levels = [(np.array(k, dtype=float), np.array(e), np.array(v),
        np.array(o)) for (k, e, v, o) in levels]
    return sf


def _dm(levels, nkpoints):
    """the onsite density matrix of the levels with their occupations"""
    dm = np.zeros((4, 4), dtype=np.complex128)
    for (k, e, v, o) in levels:
        dm += np.einsum("in,n,jn->ij", np.conj(v), o, v)/nkpoints
    return {(0, 0, 0): dm}


def _two_halves():
    """two levels at zero energy with charges +1/2 and -1/2, both empty"""
    c, s = np.sqrt(0.75), 0.5
    V = np.array([[c, 0], [0, s], [s, 0], [0, c]], dtype=np.complex128)
    return [((0., 0., 0.), np.zeros(2), V, np.zeros(2))], 1


def _random_pairs():
    """two generic k-points (their Bloch phases are complex), each with two
    levels of a random unitary basis, so
    that their charges are generic fractions, at energies inside the
    shell and with occupations in [0,1]"""
    rng = np.random.default_rng(3)
    levels = []
    for k in ((0.13, 0., 0.), (0.37, 0.21, 0.)):
        a = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
        q, r = np.linalg.qr(a)
        levels.append((k, np.array([-2e-5, 3e-5]), q[:, :2],
            rng.random(2)))
    return levels, 2


@pytest.mark.parametrize("build", [_two_halves, _random_pairs])
def test_the_fill_holds_every_charge_the_shell_can_hold(build):
    """For every N0 strictly between the charges of the shell emptied and
    filled in charge, test() says the shell can hold it, the fill fires,
    and the filled density matrix holds exactly N0"""
    levels, nkpoints = build()
    probe = _shell(-1., levels, nkpoints) # not held, so test() runs
    probe.test(_dm(levels, nkpoints))
    c0, c1 = probe.c0, probe.c1
    assert c1-c0 > 0.1 # a shell that can hold some charge
    for x in (0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98):
        N0 = c0+x*(c1-c0)
        sf = _shell(N0, levels, nkpoints)
        dm = _dm(levels, nkpoints)
        dm = sf.fill(dm)
        assert sf.fired
        assert 0. < sf.fraction < 1.
        assert abs(superscf.get_charge_bdg(dm)-N0) < 1e-12
        assert np.max(np.abs(dm[(0, 0, 0)]-dm[(0, 0, 0)].conj().T)) < 1e-14


def test_the_fill_changes_every_direction_by_the_change_of_its_shell():
    """The fill adds to the density matrix of every lattice vector d the
    sum over the shell levels of e^{2 pi i k.d} (conj(w) f w - conj(v) o v)
    per k-point, with v and o the levels and occupations before the fill
    and w and f those at the chemical potential of the fill, as a direct
    einsum computes it"""
    levels, nkpoints = _random_pairs()
    probe = _shell(-1., levels, nkpoints)
    probe.test(_dm(levels, nkpoints))
    N0 = probe.c0+0.4*(probe.c1-probe.c0)
    sf = _shell(N0, levels, nkpoints)
    ds = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)]
    dm = {d: np.zeros((4, 4), dtype=np.complex128) for d in ds}
    dm[(0, 0, 0)] = _dm(levels, nkpoints)[(0, 0, 0)]
    before = {d: m.copy() for d, m in dm.items()}
    sf.fill(dm)
    assert sf.fired
    occupations = sf._occupations(sf.mu/sf.ts)[0]
    for d in ds:
        ref = np.zeros((4, 4), dtype=np.complex128)
        for (ks, es, em, vs, occ, Q), (R, on) in zip(sf.groups, occupations):
            ws = np.einsum("gim,gmn->gin", vs, R)
            ph = np.exp(2j*np.pi*(ks@np.array(d, dtype=float)))/nkpoints
            ref += np.einsum("gin,gn,gjn->ij", ws.conj(), on*ph[:, None], ws)
            ref -= np.einsum("gin,gn,gjn->ij", vs.conj(), occ*ph[:, None], vs)
        assert np.max(np.abs(dm[d]-before[d]-ref)) < 1e-14, d
    # the change of a lattice vector is the Hermitian conjugate of the one of
    # its opposite and differs from it, so the Bloch phase matters here
    delta = {d: dm[d]-before[d] for d in ds}
    assert np.max(np.abs(delta[(1, 0, 0)])) > 1e-3
    assert np.max(np.abs(delta[(1, 0, 0)]-delta[(-1, 0, 0)].conj().T)) < 1e-14
    assert np.max(np.abs(delta[(1, 0, 0)]-delta[(-1, 0, 0)])) > 1e-3
