"""Parity test for ``wannier90.api.setup(backend="python")`` (k-mesh +
projections parsing) against ``backend="fortran"``, on the GaAs reference
case -- exercises the whole ``setup()`` path (not just ``kmesh_get`` in
isolation, see ``test_engine_kmesh.py``), including the projections-block
parser (``_engine/projections.py``).

``run()``/``wannier_run`` isn't ported yet (phases 2-3), so there's no
end-to-end python-backend equivalent of ``test_gaas.py`` yet.
"""
import numpy as np

from conftest import GAAS_EXCLUDE_BANDS, GAAS_PROJECTIONS, GAAS_WIN_KEYWORDS


def test_setup_python_backend_matches_fortran(gaas_case):
    import wannier90

    fortran_result = wannier90.setup(
        "gaas", gaas_case.mp_grid, gaas_case.kpt_latt, gaas_case.real_lattice, gaas_case.num_bands_tot,
        gaas_case.symbols, gaas_case.atoms_cart, win_keywords=GAAS_WIN_KEYWORDS,
        exclude_bands=GAAS_EXCLUDE_BANDS, projections=GAAS_PROJECTIONS,
        gamma_only=gaas_case.gamma_only, spinors=gaas_case.spinors, backend="fortran",
    )
    python_result = wannier90.setup(
        "gaas", gaas_case.mp_grid, gaas_case.kpt_latt, gaas_case.real_lattice, gaas_case.num_bands_tot,
        gaas_case.symbols, gaas_case.atoms_cart, win_keywords=GAAS_WIN_KEYWORDS,
        exclude_bands=GAAS_EXCLUDE_BANDS, projections=GAAS_PROJECTIONS,
        gamma_only=gaas_case.gamma_only, spinors=gaas_case.spinors, backend="python",
    )

    assert python_result.num_bands == fortran_result.num_bands == 12
    assert python_result.num_wann == fortran_result.num_wann == 8
    assert python_result.nntot == fortran_result.nntot
    np.testing.assert_array_equal(python_result.nnlist, fortran_result.nnlist[:, :python_result.nntot])
    np.testing.assert_array_equal(python_result.nncell, fortran_result.nncell[:, :, :python_result.nntot])
    np.testing.assert_array_equal(python_result.exclude_bands, fortran_result.exclude_bands[:5])

    np.testing.assert_allclose(python_result.proj_site, fortran_result.proj_site[:, :8])
    np.testing.assert_array_equal(python_result.proj_l, fortran_result.proj_l[:8])
    np.testing.assert_array_equal(python_result.proj_m, fortran_result.proj_m[:8])
    np.testing.assert_array_equal(python_result.proj_radial, fortran_result.proj_radial[:8])
    np.testing.assert_allclose(python_result.proj_zona, fortran_result.proj_zona[:8])
    np.testing.assert_allclose(python_result.proj_z, fortran_result.proj_z[:, :8])
    np.testing.assert_allclose(python_result.proj_x, fortran_result.proj_x[:, :8])
