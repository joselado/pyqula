import numpy as np
import pytest

from pyqula import geometry
from pyqula import superconductivity
from pyqula.sctk.pairing import pairing_generator, get_pairing_modes


def _geometries():
    g0 = geometry.honeycomb_lattice().get_supercell(3)
    g0.dimensionality = 0 # an island
    return {"honeycomb island": g0,
            "honeycomb": geometry.honeycomb_lattice(),
            "square": geometry.square_lattice().get_supercell(2),
            "triangular": geometry.triangular_lattice(),
            "chain": geometry.chain()}


def _blocks(g, df, rcut):
    """The electron-hole blocks add_pairing builds, inside the cell and
    towards every neighboring cell"""
    r = g.r
    out = [superconductivity.pairing_block(df, r1=r, r2=r, rcut=rcut)]
    if g.dimensionality > 0:
        for d in g.neighbor_directions():
            if d.dot(d) < 1e-4: continue
            r2 = g.replicas(d=d)
            out.append(superconductivity.pairing_block(df, r1=r, r2=r2,
                rcut=rcut))
    return out


@pytest.mark.parametrize("nn", [None, 2])
def test_every_registered_pairing_vanishes_beyond_its_range(nn):
    """pairing_block evaluates a registered pairing only on the pairs within
    the range pairing_generator reports for it, which is what makes
    add_pairing linear in the number of sites. That range is a statement
    about the weight functions, so it is checked here against the
    evaluation of every pair, for every registered mode: a new weight that
    reaches further would lose its distant pairs without this"""
    kw = {} if nn is None else {"nn": nn}
    built = 0
    for gname, g in _geometries().items():
        h = g.get_hamiltonian(has_spin=True)
        for mode in get_pairing_modes():
            try:
                df = pairing_generator(h, delta=0.3, mode=mode, **kw)
                full = _blocks(g, df, None)
            except (ValueError, TypeError): continue # not on this lattice
            assert df.rcut is not None, mode
            near = _blocks(g, df, df.rcut)
            for a, b in zip(full, near):
                assert np.max(np.abs((a - b).toarray()), initial=0.) < 1e-12, \
                        (gname, mode, nn)
            built += 1
    assert built > 20 # the comparison actually ran


def test_callable_pairing_keeps_every_pair():
    """A pairing given as a callable has no known range, so every pair is
    evaluated: a long-range one keeps its distant pairs"""
    g = geometry.square_lattice().get_supercell(3)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    f = lambda r1, r2: np.exp(-0.1*np.sum((r1 - r2)**2))*np.identity(2)
    df = pairing_generator(h, delta=0.2, mode=f)
    assert df.rcut is None
    m = superconductivity.pairing_block(df, r1=g.r, r2=g.r).toarray()
    assert np.all(np.abs(m[0::2, 0::2]) > 0.) # every pair of sites


@pytest.mark.parametrize("mode", ["dpid", "chiral_dwave"])
def test_singlet_pairing_of_second_neighbors(mode):
    """dpid and chiral_dwave with nn=2 used to stop with an AttributeError,
    since their builders do not pass the Hamiltonian that get_singlet read
    the shell distance from; the distance now comes from pairing_generator,
    and the pairing sits on the second-neighbor bonds and nowhere else"""
    g = geometry.triangular_lattice().get_supercell(3)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_pairing(delta=0.2, mode=mode, nn=2)
    d2 = g.neighbor_distances(n=2)[1]**2
    m = superconductivity.get_eh_sector(h.intra, i=0, j=1)
    m = np.array(m.todense()) if hasattr(m, "todense") else np.array(m)
    r = g.r
    for i in range(len(r)):
        for j in range(len(r)):
            dr2 = np.sum((r[i] - r[j])**2)
            block = m[2*i:2*i+2, 2*j:2*j+2]
            if abs(dr2 - d2) < 1e-4: assert np.max(np.abs(block)) > 1e-3
            else: assert np.max(np.abs(block)) < 1e-12
