import tracemalloc

from pyqula import geometry, meanfield
from pyqula.scftk import mfconstrains


def test_large_sparse_construction_has_no_quadratic_step():
    """Building a large sparse Hamiltonian for a mean-field calculation
    (exchange, Nambu doubling, s-wave and p-wave pairing, the neighbor
    shells, the random and Kekule guesses, the constraints) takes memory
    linear in the number of sites, about 0.05 GB here. Several of these
    steps used to build an n x n object (a list of blocks for scipy's bmat,
    an array of every pairwise distance), which at 10,000 sites is 0.8 GB
    on its own, so the bound catches any of them coming back, and the peak
    is checked after every step so that the first one to regress is the
    one named"""
    g = geometry.honeycomb_lattice().get_supercell(71) # 10,082 sites
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True, is_sparse=True)
    state = {}
    steps = [
        ("add_exchange", lambda: h.add_exchange([0.1, 0.2, 0.3])),
        ("add_rashba", lambda: h.add_rashba(0.2)),
        ("neighbor_distances", lambda: state.update(nd=g.neighbor_distances())),
        ("random guess", lambda: state.update(mf=meanfield.guess(h, mode="random"))),
        ("constraint", lambda: mfconstrains.enforce_constrains(state["mf"], h,
            ["no_inplane_magnetism"])),
        ("kekule guess", lambda: meanfield.guess(h, mode="kekule")),
        ("setup_nambu_spinor", lambda: h.setup_nambu_spinor()),
        ("add_swave", lambda: h.add_swave(0.1)),
        ("add_pairing", lambda: h.add_pairing(delta=0.1, mode="pwave")),
        ("pwave guess", lambda: state.update(mf=meanfield.guess(h, mode="pwave"))),
        ("Nambu constraint", lambda: mfconstrains.enforce_constrains(
            state["mf"], h, ["no_normal_term"])),
        ]
    tracemalloc.start()
    try:
        for name, step in steps:
            tracemalloc.reset_peak()
            step()
            _, peak = tracemalloc.get_traced_memory()
            assert peak < 0.4e9, (name, peak/1e9) # bytes
    finally:
        tracemalloc.stop()
    assert len(state["nd"]) == 4
    assert h.intra.shape == (4*len(g.r), 4*len(g.r))
