import numpy as np
import pytest

from pyqula import geometry
from pyqula.dmtk import fulldm
from testutils import assert_all_consistent, random_hermitian_hamiltonian, temporary_attr

NK = 3


def _compute(h, mode1, mode2, use_ds):
    with temporary_attr(fulldm, "mode", mode1):
        ds = [[i, 0, 0] for i in range(10)] if use_ds else None
        o = h.get_density_matrix(nk=NK, ds=ds, dm_mode=mode2)
    if use_ds:
        o = np.array([o[key] for key in o])
    return o


def test_density_matrix_modes_are_consistent():
    """explicit/vectorized and accumulate/simultaneous density-matrix
    implementations must all agree on the result, on the same Hamiltonian,
    for both the per-hopping (ds) and full-matrix output modes.

    fulldm.mode selects the kernel only on the `simultaneous` branch --
    `accumulate` always uses the batched kernels, which have no explicit
    counterpart -- so two of the four combinations here are deliberately
    the same code path, and the comparison that discriminates the explicit
    kernel from the vectorized one is the simultaneous one. It used to be
    no comparison at all in the ds case, where the switch reached nothing
    and both values ran full_dm_d_batch_vectorized (see
    test_the_kernel_switch_is_not_silently_ignored)."""
    h = random_hermitian_hamiltonian(geometry.honeycomb_lattice, supercell=4)
    modes = [(m1, m2)
             for m1 in ("explicit", "vectorized")
             for m2 in ("accumulate", "simultaneous")]
    for use_ds in (True, False):
        outs = [_compute(h, m1, m2, use_ds) for m1, m2 in modes]
        assert_all_consistent(outs, 1e-4, f"Density matrix modes (use_ds={use_ds})")


def test_density_matrix_index_convention_is_the_transposed_one():
    """full_dm returns dm[i,j] = sum_occ conj(psi_i) psi_j, the transpose
    of the usual density matrix. The mean-field machinery depends on that,
    so pin it: if it is ever normalized, every consumer that compensates
    (spectrum.ev, vev.get_dm_vev, magnetism.compute_magnetization,
    densitymatrix.restricted_dm) has to be updated in the same commit."""
    from pyqula import geometry
    h = geometry.chain().get_hamiltonian()
    h.add_exchange([0.3, 0.4, 0.5])   # generic, so sy matters
    dm = np.array(h.get_density_matrix(nk=30))
    hk = h.get_hk_gen()
    ref = np.zeros(dm.shape, dtype=complex)
    ks = np.linspace(0., 1., 30, endpoint=False)
    for k in ks:
        es, ws = np.linalg.eigh(np.array(hk([k, 0., 0.])))
        for i, e in enumerate(es):
            if e < 0.:
                psi = ws[:, i]
                ref += np.outer(np.conjugate(psi), psi)
    assert np.allclose(dm, ref/len(ks), atol=1e-8)
    # and the standard one is its transpose, which is what an expectation
    # value must be contracted against
    assert not np.allclose(dm, np.transpose(ref/len(ks)), atol=1e-8)


def test_the_kernel_switch_is_not_silently_ignored():
    """An option selected by a string must reject a typo rather than
    quietly running something else. For the per-hopping (ds) output
    fulldm.mode used to reach nothing at all, so ANY value -- including a
    misspelling -- ran the vectorized kernel."""
    h = random_hermitian_hamiltonian(geometry.honeycomb_lattice, supercell=2)
    ds = [[i, 0, 0] for i in range(3)]
    with temporary_attr(fulldm, "mode", "vectorised"):  # British spelling
        with pytest.raises(ValueError):
            h.get_density_matrix(nk=NK, ds=ds, dm_mode="simultaneous")
