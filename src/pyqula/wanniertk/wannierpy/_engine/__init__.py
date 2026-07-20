"""Pure-Python engine for Wannier90's library-mode calculation
(``wannier_setup``/``wannier_run`` in src/wannier_lib.F90), mirroring that
file's own call graph one-to-one for auditability against the Fortran
source:

    wannier_setup: param_read -> kmesh_get -> (projections parsing)
    wannier_run:   param_read -> kmesh_get -> dis_main (if disentangling)
                   -> overlap_project -> wann_main

Every function here takes its inputs as plain arguments and returns plain
values -- no ``.win`` file, no module-global state (unlike the Fortran
engine, which both of those describe). See ``params.py``'s module docstring
for why: this is a deliberate design choice, not an oversight.

Ported incrementally -- see the project's phase plan. Implemented so far:
k-mesh/b-vector determination (``kmesh.py``). Not yet ported: projections
parsing, disentanglement, Wannierisation -- calling into those paths raises
``NotImplementedError`` rather than silently returning wrong data.
"""
from __future__ import annotations

import numpy as np

from .disentangle import dis_main
from .kmesh import kmesh_get
from .overlap import overlap_project
from .params import build_params, parse_slwf_centres
from .projections import apply_projection_selection, parse_projections
from .wannierise import wann_main


def wannier_setup(
    mp_grid,
    kpt_latt,
    real_lattice,
    recip_lattice,
    num_bands_tot: int,
    atom_symbols,
    atoms_cart,
    *,
    gamma_only: bool = False,
    spinors: bool = False,
    win_keywords: dict | None = None,
    exclude_bands=None,
    projections=None,
    slwf_centres=None,
):
    """Pure-Python equivalent of the Fortran ``wannier_setup`` subroutine.

    Returns a dict with the same keys as ``wannier90.api.SetupResult``'s
    fields (minus ``cwd``/``_setup_args``, which are Fortran-subprocess
    concerns that don't apply here).
    """
    params = build_params(win_keywords, exclude_bands, projections, slwf_centres)
    recip_lattice = np.asarray(recip_lattice, dtype=np.float64)

    kmesh = kmesh_get(
        np.asarray(kpt_latt, dtype=np.float64),
        recip_lattice,
        search_shells=params.search_shells,
        kmesh_tol=params.kmesh_tol,
        num_shells=params.num_shells,
        shell_list=params.shell_list or None,
        skip_b1_tests=params.skip_b1_tests,
        gamma_only=gamma_only,
    )

    num_bands = int(num_bands_tot) - len(params.exclude_bands)
    exclude_bands_arr = np.array(params.exclude_bands, dtype=np.int64)

    parsed_projections = None
    if params.projections:
        atoms_cart = np.asarray(atoms_cart, dtype=np.float64)
        atoms_frac = (recip_lattice @ atoms_cart) / (2.0 * np.pi)  # utility_cart_to_frac, per-column
        raw_projections = parse_projections(
            params.projections, atom_symbols, atoms_frac, recip_lattice, spinors=spinors
        )
        parsed_projections = apply_projection_selection(
            raw_projections, params.num_wann, params.select_projections
        )

    return {
        "nntot": kmesh.nntot,
        "nnlist": kmesh.nnlist,
        "nncell": kmesh.nncell,
        "num_bands": num_bands,
        "num_wann": params.num_wann,
        "exclude_bands": exclude_bands_arr,
        "projections": parsed_projections,
        "_kmesh": kmesh,  # kept for wannier_run to reuse without recomputing
        "_params": params,
    }


def wannier_run(
    *,
    mp_grid,
    kpt_latt,
    real_lattice,
    recip_lattice,
    atom_symbols,
    atoms_cart,
    gamma_only: bool,
    M_matrix,
    A_matrix,
    eigenvalues,
    engine_state: dict,
    sym=None,
):
    """Pure-Python equivalent of the Fortran ``wannier_run`` subroutine.

    ``engine_state`` is the dict returned by :func:`wannier_setup` (passed
    through via ``SetupResult._engine_state``) -- it carries the k-mesh and
    parsed ``.win`` params so they don't need recomputing.

    ``sym`` (a ``sitesym.SymmetryData``) enables ``lsitesymmetry`` -- not
    supported together with frozen-window disentanglement, matching
    upstream.

    Returns a dict with the same keys as ``wannier90.api.RunResult``'s
    fields. Validated end-to-end against upstream's own GaAs reference data
    (``tests/test_engine_wannierise.py``) for the disentangling path.

    ``gamma_only=True`` is deliberately rejected here rather than silently
    run through the general (complex-arithmetic) code path below: that was
    tried (feed the gamma-halved k-mesh straight into dis_main/wann_main)
    and empirically gives *wrong* answers, not just a slower version of the
    right one (checked against test-suite/tests/testw90_na_chain_gamma:
    spread 180.6 vs the fortran backend's 37.5, and it didn't even
    converge). The "_gamma" Fortran routines (dis_extract_gamma,
    wann_main_gamma, etc.) aren't merely a real-arithmetic optimization of
    the same search -- they constrain the unitary rotations to be real
    (time-reversal symmetry at Gamma), a genuinely different, more
    constrained optimization problem, and reproducing it needs a dedicated
    real-arithmetic port, not reuse of the complex one.
    """
    if gamma_only:
        raise NotImplementedError(
            "backend='python': gamma_only=True is not yet ported -- see the porting plan's "
            "gamma-point variants phase (dis_extract_gamma/wann_main_gamma need a dedicated "
            "real-arithmetic implementation, confirmed NOT equivalent to just reusing the "
            "general complex code path on the gamma-halved k-mesh)"
        )
    params: object = engine_state["_params"]
    kmesh = engine_state["_kmesh"]
    num_bands, num_kpts = eigenvalues.shape
    num_wann = params.num_wann
    disentanglement = num_bands > num_wann

    if disentanglement:
        dis_win_min = params.dis_win_min if params.dis_win_min is not None else float(eigenvalues.min())
        dis_win_max = params.dis_win_max if params.dis_win_max is not None else float(eigenvalues.max())
        dis_froz_min = params.dis_froz_min if params.dis_froz_min is not None else dis_win_min

        u_matrix_opt, u_matrix, lwindow, M_wann_gauge, dis_converged = dis_main(
            A_matrix, M_matrix, eigenvalues, kmesh.nnlist, kmesh.wb, num_wann,
            dis_win_min, dis_win_max, params.frozen_states, dis_froz_min, params.dis_froz_max,
            params.dis_num_iter, params.dis_mix_ratio, params.dis_conv_tol, params.dis_conv_window,
            sym=sym,
        )
        if not dis_converged:
            import warnings
            warnings.warn("backend='python': disentanglement did not converge within dis_num_iter")
    else:
        u_matrix, M_wann_gauge = overlap_project(A_matrix, M_matrix, kmesh.nnlist, sym=sym)
        u_matrix_opt = np.tile(np.eye(num_wann, dtype=complex)[:, :, None], (1, 1, num_kpts))
        lwindow = np.ones((num_bands, num_kpts), dtype=bool)

    real_lattice = np.asarray(real_lattice, dtype=np.float64)
    projections = engine_state.get("projections")

    proj_site_cart = None
    if params.guiding_centres:
        if not projections:
            raise ValueError("wannier_run: guiding_centres=True requires projections to be given")
        # utility_frac_to_cart per site: cart = frac @ real_lattice (real_lattice rows = lattice vectors).
        proj_site_cart = (np.array([p.site_frac for p in projections]) @ real_lattice).T

    ccentres_cart = None
    if params.selective_loc:
        overrides = parse_slwf_centres(params.slwf_centres)
        if projections:
            ccentres_frac = [p.site_frac for p in projections[:num_wann]]
        elif set(range(1, num_wann + 1)) <= overrides.keys():
            ccentres_frac = [None] * num_wann  # fully overridden below, defaults never read
        else:
            raise ValueError(
                "wannier_run: selective_loc requires either projections or a slwf_centres "
                "entry for every Wannier function to seed the constraint centres"
            )
        for j in range(num_wann):
            if (j + 1) in overrides:
                ccentres_frac[j] = overrides[j + 1]
        ccentres_cart = (np.array(ccentres_frac) @ real_lattice)  # (num_wann, 3), Cartesian

    conv_window = params.conv_window if params.conv_window > 0 else 5
    U_final, wann_centres, wann_spreads, om_tot, om_i, om_tilde, wann_converged = wann_main(
        u_matrix, M_wann_gauge, kmesh.nnlist, kmesh.bk, kmesh.wb,
        num_iter=params.num_iter, num_cg_steps=params.num_cg_steps, conv_tol=params.conv_tol,
        conv_window=conv_window, trial_step=params.trial_step, guiding_centres=params.guiding_centres,
        bka=kmesh.bka, neigh=kmesh.neigh, proj_site_cart=proj_site_cart,
        num_no_guide_iter=params.num_no_guide_iter, num_guide_cycles=params.num_guide_cycles,
        selective_loc=params.selective_loc, slwf_num=params.slwf_num, slwf_constrain=params.slwf_constrain,
        lambda_loc=params.slwf_lambda if params.slwf_constrain else 0.0, ccentres_cart=ccentres_cart,
        fixed_step=params.fixed_step, precond=params.precond, kpt_latt=np.asarray(kpt_latt, dtype=np.float64),
        real_lattice=real_lattice, mp_grid=np.asarray(mp_grid, dtype=np.int64), sym=sym,
    )
    if not wann_converged:
        import warnings
        warnings.warn("backend='python': Wannierisation did not converge within num_iter")

    return {
        "U_matrix": U_final,
        "U_matrix_opt": u_matrix_opt,
        "lwindow": lwindow,
        "wann_centres": wann_centres,
        "wann_spreads": wann_spreads,
        "spread_total": om_tot,
        "spread_invariant": om_i,
        "spread_tilde": om_tilde,
    }
