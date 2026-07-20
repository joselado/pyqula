"""In-memory equivalent of the subset of ``param_read`` (src/parameters.F90)
that ``wannier_setup``/``wannier_run`` actually consume.

Per the python backend's design: every algorithmic parameter comes in as a
plain Python argument (the same ``win_keywords``/``exclude_bands``/
``projections`` objects ``wannier90.api.setup``/``run`` already accept), not
by reading a ``<seedname>.win`` file from disk. Hand-authored ``.win`` files
on disk are only supported by ``backend="fortran"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def parse_range_vector(spec) -> list[int]:
    """Parse wannier90's range-vector syntax (``"1-5,8,10-12"``) or an
    iterable of (1-indexed) ints into a sorted list of unique ints. Use
    :func:`parse_range_vector_ordered` instead for keywords where position
    is meaningful (e.g. ``select_projections`` -- entry *j* names the
    projection for Wannier function *j*, not just membership in a set)."""
    return sorted(set(parse_range_vector_ordered(spec)))


def parse_range_vector_ordered(spec) -> list[int]:
    """Like :func:`parse_range_vector`, but preserves the given order and
    duplicates instead of sorting into a unique set -- ``param_get_keyword``
    reads these positionally in the Fortran source, e.g.
    ``select_projections = 8,1,2,3,4,5,6,7`` means "Wannier function 1 gets
    projection 8", not merely "projections {1..8} are used"."""
    if spec is None:
        return []
    if isinstance(spec, str):
        out = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                a, b = chunk.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(chunk))
        return out
    return [int(x) for x in spec]


def parse_slwf_centres(lines: list[str]) -> dict[int, list[float]]:
    """Parse a ``begin slwf_centres`` / ``end slwf_centres`` block's lines
    into ``{1-indexed Wannier function: [x, y, z] fractional}``. Matches
    ``param_get_centre_constraint_from_column``: each line is
    ``wann_index x y z [lagrange_multiplier]`` -- the optional 5th column
    (a per-centre Lagrange multiplier override) is accepted but ignored,
    same as upstream (parsed, never actually stored/used there either)."""
    out: dict[int, list[float]] = {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if len(parts) not in (4, 5):
            raise ValueError(f"slwf_centres: malformed line {line!r} (expected 'index x y z [lambda]')")
        out[int(parts[0])] = [float(x) for x in parts[1:4]]
    return out


@dataclass
class WinParams:
    num_wann: int
    exclude_bands: list = field(default_factory=list)
    projections: list = field(default_factory=list)
    search_shells: int = 36
    kmesh_tol: float = 1.0e-6
    num_shells: int = 0
    shell_list: list = field(default_factory=list)
    skip_b1_tests: bool = False
    dis_win_min: float | None = None  # None => defaults to min(eigval) once eigval is known
    dis_win_max: float | None = None  # None => defaults to max(eigval) once eigval is known
    dis_froz_min: float | None = None  # None => defaults to dis_win_min
    dis_froz_max: float = 0.0
    frozen_states: bool = False
    dis_num_iter: int = 200
    dis_mix_ratio: float = 0.5
    dis_conv_tol: float = 1.0e-10
    dis_conv_window: int = 3
    num_iter: int = 100
    num_cg_steps: int = 5
    conv_tol: float = 1.0e-10
    conv_window: int = -1
    trial_step: float = 2.0
    fixed_step: float | None = None
    guiding_centres: bool = False
    precond: bool = False
    num_no_guide_iter: int = 0
    num_guide_cycles: int = 1
    select_projections: list | None = None
    slwf_num: int = 0  # 0 is a placeholder; build_params always resolves this to num_wann by default
    selective_loc: bool = False
    slwf_constrain: bool = False
    slwf_lambda: float = 1.0
    slwf_centres: list = field(default_factory=list)  # raw "wann_index x y z" lines, fractional


def build_params(win_keywords: dict | None, exclude_bands=None, projections=None, slwf_centres=None) -> WinParams:
    if not win_keywords or "num_wann" not in win_keywords:
        raise ValueError(
            "backend='python' requires win_keywords={'num_wann': ..., ...} passed directly "
            "(hand-authored .win files on disk are only supported by backend='fortran')"
        )
    shell_list = parse_range_vector(win_keywords.get("shell_list"))
    frozen_states = "dis_froz_max" in win_keywords
    if "dis_froz_min" in win_keywords and not frozen_states:
        raise ValueError("build_params: found dis_froz_min but not dis_froz_max")
    dis_win_min = win_keywords.get("dis_win_min")
    dis_froz_min = win_keywords.get("dis_froz_min")
    num_wann = int(win_keywords["num_wann"])
    slwf_num = int(win_keywords.get("slwf_num", num_wann))
    if not (1 <= slwf_num <= num_wann):
        raise ValueError("slwf_num must be an integer between 1 and num_wann")
    selective_loc = slwf_num < num_wann
    slwf_constrain = bool(win_keywords.get("slwf_constrain", False)) and selective_loc
    return WinParams(
        num_wann=int(win_keywords["num_wann"]),
        exclude_bands=parse_range_vector(exclude_bands),
        projections=list(projections) if projections else [],
        search_shells=int(win_keywords.get("search_shells", 36)),
        kmesh_tol=float(win_keywords.get("kmesh_tol", 1.0e-6)),
        num_shells=len(shell_list),
        shell_list=shell_list,
        skip_b1_tests=bool(win_keywords.get("skip_b1_tests", False)),
        # dis_win_min/dis_win_max/dis_froz_min default to min/max(eigval)/dis_win_min
        # respectively when None -- only known once wannier_run has eigval, so resolved
        # there (see _engine.wannier_run), not here.
        dis_win_min=float(dis_win_min) if dis_win_min is not None else None,
        dis_win_max=float(win_keywords["dis_win_max"]) if "dis_win_max" in win_keywords else None,
        dis_froz_min=float(dis_froz_min) if dis_froz_min is not None else None,
        dis_froz_max=float(win_keywords.get("dis_froz_max", 0.0)),
        frozen_states=frozen_states,
        dis_num_iter=int(win_keywords.get("dis_num_iter", 200)),
        dis_mix_ratio=float(win_keywords.get("dis_mix_ratio", 0.5)),
        dis_conv_tol=float(win_keywords.get("dis_conv_tol", 1.0e-10)),
        dis_conv_window=int(win_keywords.get("dis_conv_window", 3)),
        num_iter=int(win_keywords.get("num_iter", 100)),
        num_cg_steps=int(win_keywords.get("num_cg_steps", 5)),
        conv_tol=float(win_keywords.get("conv_tol", 1.0e-10)),
        conv_window=int(win_keywords.get("conv_window", -1)),
        trial_step=float(win_keywords.get("trial_step", 2.0)),
        fixed_step=float(win_keywords["fixed_step"]) if win_keywords.get("fixed_step", -999.0) > 0.0 else None,
        guiding_centres=bool(win_keywords.get("guiding_centres", False)),
        precond=bool(win_keywords.get("precond", False)),
        num_no_guide_iter=int(win_keywords.get("num_no_guide_iter", 0)),
        num_guide_cycles=int(win_keywords.get("num_guide_cycles", 1)),
        select_projections=parse_range_vector_ordered(win_keywords.get("select_projections")) or None,
        slwf_num=slwf_num,
        selective_loc=selective_loc,
        slwf_constrain=slwf_constrain,
        slwf_lambda=float(win_keywords.get("slwf_lambda", 1.0)),
        slwf_centres=list(slwf_centres) if slwf_centres else [],
    )
