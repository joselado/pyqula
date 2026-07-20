"""Pure-Python port of the projections-block parsing subset of
``param_get_projections`` (src/parameters.F90) -- turns the raw projection
strings ``wannier90.api.setup`` already accepts (e.g.
``"f=0.25,0.25,0.25 : s"``) into the ``proj_site``/``proj_l``/``proj_m``/...
arrays ``SetupResult`` exposes.

Scope (matches what's exercised by common usage and the GaAs reference
case): fractional/cartesian/atom-label sites, shorthand orbital labels
(``s``, ``p``, ``px``, ``d``, ``sp3``, ...) and explicit ``l=``/``mr=``
syntax, optional ``z=``/``x=``/``zona=``/``r=`` modifiers, spinor
projections (``(u)``/``(d)``/``[qaxis]``, see ``_parse_spin_annotations``),
and ``select_projections`` (see ``apply_projection_selection``). Not yet
ported: ``random``/partial-random projections -- these raise
``NotImplementedError``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# label -> list of (l, mr) pairs, mr 1-indexed exactly as Fortran's
# ang_states table (param_get_projections, src/parameters.F90). Iteration
# order downstream is by ascending (l, mr), not by label-appearance order --
# see _expand_ang_states.
_LABELS: dict[str, list[tuple[int, int]]] = {
    "s": [(0, 1)],
    "p": [(1, 1), (1, 2), (1, 3)], "pz": [(1, 1)], "px": [(1, 2)], "py": [(1, 3)],
    "d": [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],
    "dz2": [(2, 1)], "dxz": [(2, 2)], "dyz": [(2, 3)], "dx2-y2": [(2, 4)], "dxy": [(2, 5)],
    "f": [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)],
    "fz3": [(3, 1)], "fxz2": [(3, 2)], "fyz2": [(3, 3)], "fxyz": [(3, 4)],
    "fz(x2-y2)": [(3, 5)], "fx(x2-3y2)": [(3, 6)], "fy(3x2-y2)": [(3, 7)],
    "sp": [(-1, 1), (-1, 2)], "sp-1": [(-1, 1)], "sp-2": [(-1, 2)],
    "sp2": [(-2, 1), (-2, 2), (-2, 3)], "sp2-1": [(-2, 1)], "sp2-2": [(-2, 2)], "sp2-3": [(-2, 3)],
    "sp3": [(-3, 1), (-3, 2), (-3, 3), (-3, 4)],
    "sp3-1": [(-3, 1)], "sp3-2": [(-3, 2)], "sp3-3": [(-3, 3)], "sp3-4": [(-3, 4)],
    "sp3d": [(-4, 1), (-4, 2), (-4, 3), (-4, 4), (-4, 5)],
    "sp3d-1": [(-4, 1)], "sp3d-2": [(-4, 2)], "sp3d-3": [(-4, 3)], "sp3d-4": [(-4, 4)], "sp3d-5": [(-4, 5)],
    "sp3d2": [(-5, 1), (-5, 2), (-5, 3), (-5, 4), (-5, 5), (-5, 6)],
    "sp3d2-1": [(-5, 1)], "sp3d2-2": [(-5, 2)], "sp3d2-3": [(-5, 3)],
    "sp3d2-4": [(-5, 4)], "sp3d2-5": [(-5, 5)], "sp3d2-6": [(-5, 6)],
}
_L_MR_COUNT = {0: 1, 1: 3, 2: 5, 3: 7, -1: 2, -2: 3, -3: 4, -4: 5, -5: 6}


@dataclass
class ParsedProjection:
    site_frac: np.ndarray  # (3,)
    l: int
    m: int  # mr, 1-indexed
    z_axis: np.ndarray  # (3,), normalized
    x_axis: np.ndarray  # (3,), normalized, orthogonal to z_axis
    radial: int
    zona: float
    spin: int | None = None  # +1 (up) / -1 (down); None unless spinors=True
    s_qaxis: np.ndarray | None = None  # (3,) spin quantisation axis; None unless spinors=True


def _parse_coord(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.split(",")])


def _species_frac_positions(atom_symbols, atoms_frac) -> dict[str, list[np.ndarray]]:
    species: dict[str, list[np.ndarray]] = {}
    for sym, pos in zip(atom_symbols, atoms_frac.T):
        species.setdefault(str(sym), []).append(pos)
    return species


def _expand_ang_states(state_specs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Given a set of (l, mr) pairs (possibly with repeats, in whatever
    order the user's labels/``l=``/``mr=`` syntax produced them), return
    them in ascending (l, mr) order -- Fortran assembles the final
    projection list via nested loops over l then mr, not label-appearance
    order (param_get_projections, ang_states loop)."""
    return sorted(set(state_specs))


def _parse_orbital_states(spec: str) -> list[tuple[int, int]]:
    """Parse the angular-momentum part of a projection line (before any
    z=/x=/zona=/r= modifiers), e.g. "s", "p;d", "l=2,mr=1,3"."""
    states: list[tuple[int, int]] = []
    for clause in spec.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        if clause.startswith("l="):
            rest = clause[2:]
            if "," in rest and "mr=" in rest:
                l_part, mr_part = rest.split(",", 1)
                l_tmp = int(l_part)
                if not mr_part.startswith("mr="):
                    raise ValueError(f"projections: malformed l=/mr= clause: {clause!r}")
                mrs = [int(x) for x in mr_part[3:].split(",")]
                states.extend((l_tmp, m) for m in mrs)
            else:
                l_tmp = int(rest)
                states.extend((l_tmp, m) for m in range(1, _L_MR_COUNT[l_tmp] + 1))
        else:
            for label in clause.split(","):
                label = label.strip()
                if label not in _LABELS:
                    raise ValueError(f"projections: unrecognised orbital label {label!r}")
                states.extend(_LABELS[label])
    return _expand_ang_states(states)


def _orthogonalize(z_axis: np.ndarray, x_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = z_axis / np.linalg.norm(z_axis)
    x = x_axis / np.linalg.norm(x_axis)
    cosphi = float(np.dot(z, x))
    if abs(cosphi) <= 1.0e-6:
        return z, x
    sinphi = np.sqrt(1.0 - cosphi * cosphi)
    x_new = (x - cosphi * z) / sinphi
    return z, x_new


def _parse_spin_annotations(rest: str, spinors: bool):
    """Strip and interpret the optional ``[qaxis]`` (spin quantisation
    direction) and ``(u)``/``(d)``/``(u d)`` (spin channel) annotations from
    the tail of a projection line's post-site text.

    Matches Fortran's actual (slightly surprising) behaviour: finding ``[``
    or ``(`` *truncates* ``rest`` there, discarding anything after the
    bracket -- so any ``z=``/``x=``/``zona=``/``r=`` modifiers must come
    *before* these annotations, not after (real .win files always write
    them this way; see param_get_projections).

    Returns (rest_without_annotations, spin_channels, s_qaxis) where
    spin_channels is a list of +1/-1 (one entry per spin sub-state to
    expand into) -- ``[None]`` when not spinors.
    """
    s_qaxis = np.array([0.0, 0.0, 1.0])
    pos1 = rest.find("[")
    if pos1 >= 0:
        if not spinors:
            raise ValueError("projections: spin quantisation axis given but spinors=False")
        pos2 = rest.find("]", pos1)
        if pos2 < 0:
            raise ValueError("projections: no closing ']' for spin quantisation direction")
        s_qaxis = _parse_coord(rest[pos1 + 1:pos2])
        rest = rest[:pos1]

    spin_channels: list[int | None]
    pos1 = rest.find("(")
    if pos1 >= 0:
        if not spinors:
            raise ValueError("projections: spin channel given but spinors=False")
        pos2 = rest.find(")", pos1)
        if pos2 < 0:
            raise ValueError("projections: no closing ')' for spin channel")
        channel = rest[pos1 + 1:pos2]
        up, down = "u" in channel, "d" in channel
        if not up and not down:
            raise ValueError("projections: found spin brackets but neither u nor d")
        spin_channels = [s for s, present in ((1, up), (-1, down)) if present]
        rest = rest[:pos1]
    elif spinors:
        spin_channels = [1, -1]  # default: both channels
    else:
        spin_channels = [None]

    return rest, spin_channels, s_qaxis


def _parse_line(line: str, atom_symbols, atoms_frac, recip_lattice, spinors: bool) -> list[ParsedProjection]:
    dummy = "".join(line.split())  # utility_strip: remove all whitespace
    pos1 = dummy.find(":")
    if pos1 < 0:
        raise ValueError(f"projections: malformed projection definition: {line!r}")
    site_spec = dummy[:pos1]
    rest = dummy[pos1 + 1:]

    if site_spec.startswith("f="):
        site_positions = [_parse_coord(site_spec[2:])]
    elif site_spec.startswith("c="):
        cart = _parse_coord(site_spec[2:])
        frac = (recip_lattice @ cart) / (2.0 * np.pi)  # utility_cart_to_frac
        site_positions = [frac]
    else:
        species = _species_frac_positions(atom_symbols, atoms_frac)
        if site_spec not in species:
            raise ValueError(f"projections: atom site not recognised {site_spec!r}")
        site_positions = species[site_spec]

    rest, spin_channels, s_qaxis = _parse_spin_annotations(rest, spinors)

    pos1 = rest.find(":")
    orbital_spec = rest[:pos1] if pos1 >= 0 else rest
    states = _parse_orbital_states(orbital_spec)

    z_axis, x_axis, zona, radial = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 1.0, 1
    if pos1 >= 0:
        modifiers = rest[pos1 + 1:]
        if "z=" in modifiers:
            z_axis = _parse_coord(modifiers.split("z=")[1].split(":")[0])
        if "x=" in modifiers:
            x_axis = _parse_coord(modifiers.split("x=")[1].split(":")[0])
        if "zona=" in modifiers:
            zona = float(modifiers.split("zona=")[1].split(":")[0])
        if "r=" in modifiers:
            radial = int(modifiers.split("r=")[1].split(":")[0])
    z_axis, x_axis = _orthogonalize(z_axis, x_axis)

    out = []
    for site_frac in site_positions:
        for l, m in states:
            for spin in spin_channels:
                out.append(ParsedProjection(site_frac, l, m, z_axis, x_axis, radial, zona,
                                             spin=spin, s_qaxis=s_qaxis.copy() if spinors else None))
    return out


def parse_projections(lines: list[str], atom_symbols, atoms_frac, recip_lattice,
                       spinors: bool = False) -> list[ParsedProjection]:
    """Parse a ``begin projections`` / ``end projections`` block's lines
    (without the ``begin``/``end`` markers) into candidate trial orbitals,
    in definition order -- the raw ``num_proj``-length list
    ``param_get_projections`` builds, *before* ``proj2wann_map`` narrows it
    down to ``num_wann`` (see :func:`apply_projection_selection`)."""
    stripped = [l.strip() for l in lines if l.strip()]
    if stripped and stripped[0].lower() in ("ang", "bohr"):
        raise NotImplementedError("projections: explicit ang/bohr unit line is not yet ported")
    if stripped and "random" in stripped[0].lower():
        raise NotImplementedError("projections: 'random'/partial-random projections are not yet ported")

    out: list[ParsedProjection] = []
    for line in stripped:
        out.extend(_parse_line(line, atom_symbols, atoms_frac, recip_lattice, spinors))
    return out


def apply_projection_selection(parsed: list[ParsedProjection], num_wann: int,
                                select_projections: list[int] | None) -> list[ParsedProjection]:
    """Narrow a raw parsed-projections list down to exactly ``num_wann``
    trial orbitals, per ``param_read``'s ``proj2wann_map`` construction.

    Without ``select_projections`` (the common case): the first
    ``num_wann`` projections in definition order are used, and any extras
    are silently dropped (matches Fortran's identity ``proj2wann_map`` --
    it's *not* an error to define more than ``num_wann`` projections and
    not select among them). With ``select_projections`` (1-indexed
    positions into ``parsed``, length ``num_wann``): wannier function ``j``
    takes ``parsed[select_projections[j] - 1]`` directly -- equivalent to,
    but simpler than, Fortran's index-map-then-invert.
    """
    if len(parsed) < num_wann:
        raise ValueError(
            f"projections: too few projection functions defined ({len(parsed)} < num_wann={num_wann})"
        )
    if select_projections is None:
        return parsed[:num_wann]

    if len(select_projections) != num_wann:
        raise ValueError(
            f"select_projections: expected {num_wann} entries (== num_wann), got {len(select_projections)}"
        )
    if any(s < 1 for s in select_projections):
        raise ValueError("select_projections must contain positive numbers")
    if max(select_projections) > len(parsed):
        raise ValueError("select_projections contains a number greater than the number of projections defined")
    return [parsed[s - 1] for s in select_projections]
