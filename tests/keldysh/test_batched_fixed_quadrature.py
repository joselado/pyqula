import numpy as np
import pytest

from pyqula import algebra
from pyqula import geometry
from pyqula import heterostructures
from pyqula.keldyshtk import current as keldysh_current
from pyqula.keldyshtk.current import (
    current_integrand,
    current_integrand_batch,
    dc_current,
)

# Correctness tests for keldyshtk.current's `quadrature="fixed"` batched
# chain solve (current_integrand_batch -> _floquet_green_functions_batch ->
# _assemble_chain_batch_jit/_rgf_chain_batch_jit), which replaced a plain
# per-node Python-callback loop over the same fixed node set. This repo's
# own history (documentation/keldysh_sideband_decimation_plan.md) has twice
# had a primitive validate to machine precision in isolation and then be
# wrong once wired into the full dc_current pipeline -- so these tests check
# both the batched primitive against its unbatched sibling AND dc_current's
# actual output against the independently-validated "adaptive" default.


def _sc_sc_junction(delta_sc, transparency, ht_delta=1e-4):
    """Two-lead SC-SC junction, same shape as the cases benchmarked in
    keldysh_sideband_decimation_plan.md's "fixed" quadrature updates."""
    h1 = geometry.chain().get_hamiltonian(); h1.shift_fermi(1.); h1.add_swave(delta_sc)
    h2 = geometry.chain().get_hamiltonian(); h2.shift_fermi(1.); h2.add_swave(delta_sc)
    HT = heterostructures.build(h1, h2)
    HT.set_coupling(transparency)
    HT.delta = ht_delta
    return HT


def _normal_normal_junction(transparency):
    h1 = geometry.chain().get_hamiltonian(); h1.turn_nambu()
    h2 = geometry.chain().get_hamiltonian(); h2.turn_nambu()
    HT = heterostructures.build(h1, h2)
    HT.set_coupling(transparency)
    HT.delta = 1e-3
    return HT


def _chain_pieces(ht, voltage, nmax_max):
    """Everything current_integrand/current_integrand_batch need, built the
    same way dc_current itself builds them (see dc_current's own body)."""
    ht = keldysh_current._prepare_bias_target(ht)
    keldysh_current._check_supported(ht)
    delta = ht.delta
    lead0 = ht.Hl
    tauz = algebra.todense(lead0.get_operator("tauz").get_matrix()).astype(np.complex128)
    system = keldysh_current._prepare_system(ht)
    chain_consts = keldysh_current._prepare_chain_consts(system)
    return ht, delta, tauz, system, chain_consts


@pytest.mark.parametrize("delta_sc,transparency,voltage,nmax", [
    (0.3, 0.5, 0.031, 20),   # deep-subgap representative case
    (0.1, 0.3, 0.15, 20),    # worst-case-accuracy point of the fixed-quad sweep
    (0.1, 0.3, 0.55, 20),    # hardest SC-SC case of the fixed-quad sweep
])
def test_batched_integrand_matches_per_node_loop(delta_sc, transparency, voltage, nmax):
    """current_integrand_batch's per-node values must exactly match calling
    current_integrand once per node in a Python loop -- both walk the same
    (block,sideband) RGF chain for each node, batching only changes how many
    Python/numba dispatches that costs, never the arithmetic."""
    ht = _sc_sc_junction(delta_sc, transparency)
    ht, delta, tauz, system, chain_consts = _chain_pieces(ht, voltage, nmax)
    nodes, _ = keldysh_current._fixed_quasienergy_nodes(voltage)

    cache_loop = {}
    vals_loop = np.array([
        current_integrand(ht, voltage, e, nmax, tauz, delta=delta,
                           temperature=0., cache=cache_loop, system=system,
                           chain_consts=chain_consts)
        for e in nodes
    ])

    cache_batch = {}
    vals_batch = current_integrand_batch(
        ht, voltage, nodes, nmax, tauz, delta=delta, temperature=0.,
        cache=cache_batch, system=system, chain_consts=chain_consts)

    assert np.array_equal(vals_loop, vals_batch)


def test_batched_integrand_independent_of_chunk_size():
    """current_integrand_batch chunks the node axis to bound peak memory
    (_BATCH_CHUNK_NODES) -- every chunking must give the identical per-node
    result, since each node's chain solve is independent of every other
    node's and chunking only changes how many are solved together."""
    ht = _sc_sc_junction(0.1, 0.3)
    ht, delta, tauz, system, chain_consts = _chain_pieces(ht, 0.55, 20)
    nodes, _ = keldysh_current._fixed_quasienergy_nodes(0.55)
    assert len(nodes) > 300  # make sure this actually exercises >1 chunk below

    ref = current_integrand_batch(
        ht, 0.55, nodes, 20, tauz, delta=delta, cache={}, system=system,
        chain_consts=chain_consts, chunk_size=keldysh_current._BATCH_CHUNK_NODES)
    for chunk_size in (1, 7, 64, 10_000):
        got = current_integrand_batch(
            ht, 0.55, nodes, 20, tauz, delta=delta, cache={}, system=system,
            chain_consts=chain_consts, chunk_size=chunk_size)
        assert np.array_equal(ref, got), f"chunk_size={chunk_size} disagrees"


@pytest.mark.parametrize("delta_sc,transparency,voltage", [
    (0.3, 0.5, 0.031),
    (0.1, 0.3, 0.15),
])
def test_dc_current_fixed_quadrature_matches_adaptive(delta_sc, transparency, voltage):
    """End-to-end: dc_current(quadrature="fixed") (now batched) must still
    agree with dc_current(quadrature="adaptive"), the independently-trusted
    default, to within the tolerance already established for the fixed
    quadrature's own accuracy (worst case 5.8e-4 over a 34-point sweep, see
    keldysh_sideband_decimation_plan.md) -- batching the chain solve must not
    change dc_current's returned value, only its cost."""
    ht = _sc_sc_junction(delta_sc, transparency)
    Ifixed = dc_current(ht, voltage, nmax_max=40, quadrature="fixed")
    Iadapt = dc_current(ht, voltage, nmax_max=40, quadrature="adaptive")
    rel = abs(Ifixed-Iadapt)/max(abs(Iadapt), 1e-12)
    assert rel < 2e-3


def test_dc_current_fixed_quadrature_matches_adaptive_at_finite_temperature():
    """The temperature!=0 branch of _assemble_chain_batch_jit is a verbatim
    copy of _assemble_chain_jit's, but was not covered by the bit-identical
    per-node checks above (those all use temperature=0.) -- check the full
    dc_current pipeline directly at finite temperature, since native
    temperature broadening is now the default path for finite_T_didv/kappa
    and so is reachable with quadrature="fixed" too."""
    ht = _sc_sc_junction(0.1, 0.5)
    Ifixed = dc_current(ht, 0.3, nmax_max=20, temperature=0.02, quadrature="fixed")
    Iadapt = dc_current(ht, 0.3, nmax_max=20, temperature=0.02, quadrature="adaptive")
    rel = abs(Ifixed-Iadapt)/max(abs(Iadapt), 1e-12)
    assert rel < 2e-3


def test_dc_current_fixed_quadrature_normal_junction_matches_adaptive():
    """Same cross-check on a normal-normal junction (no gap-edge
    singularity, the case the fixed quadrature's panel design is least
    suited to) -- accuracy should still hold even though this is the case
    where "fixed" pays the largest wall-clock overkill."""
    ht = _normal_normal_junction(0.6)
    Ifixed = dc_current(ht, 0.3, nmax_max=20, quadrature="fixed")
    Iadapt = dc_current(ht, 0.3, nmax_max=20, quadrature="adaptive")
    rel = abs(Ifixed-Iadapt)/max(abs(Iadapt), 1e-12)
    assert rel < 2e-3
