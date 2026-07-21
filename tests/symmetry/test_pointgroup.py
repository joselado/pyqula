import numpy as np

from pyqula import geometry
from pyqula.symmetrytk import pointgroup


def _names(ops):
    return sorted(c.op.name.split("(")[0] for c in ops)


def test_honeycomb_finds_c2_and_inversion_about_bond_midpoint():
    # honeycomb_lattice()'s two atoms straddle the origin symmetrically,
    # so the origin is a bond-midpoint centre with C2/inversion symmetry
    # (not the hexagon-centre C6/C3 axis, which this heuristic's default
    # origin-only centre search does not find -- see pointgroup.py)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    found = pointgroup.find_point_group(g, h=h)
    names = _names(found)
    assert "inversion" in names
    assert "C2" in names
    assert "C3" not in names


def test_square_lattice_finds_c4():
    g = geometry.square_lattice()
    h = g.get_hamiltonian(has_spin=False)
    found = pointgroup.find_point_group(g, h=h)
    names = _names(found)
    assert "C4" in names
    assert "inversion" in names


def test_hamiltonian_symmetries_are_a_subset_of_geometry_symmetries():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    geometry_only = pointgroup.find_point_group(g)
    hamiltonian_verified = pointgroup.find_point_group(g, h=h)
    assert len(hamiltonian_verified) <= len(geometry_only)
    geom_keys = {(o.name, tuple(o.center.tolist())) for o in geometry_only}
    for c in hamiltonian_verified:
        assert (c.op.name, tuple(c.op.center.tolist())) in geom_keys


def test_sublattice_imbalance_breaks_the_swap_symmetry():
    # onsite energy that differs between the two honeycomb sublattices
    # breaks any operation that swaps them (inversion, in-plane C2 about
    # the z axis), but must leave operations that fix each atom in place
    # (e.g. any operation about the x axis, since both atoms sit at y=z=0)
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.intra[0, 0] += 0.7
    found = pointgroup.find_point_group(g, h=h)
    for c in found:
        if "axis=[0.0, 0.0, 1.0]" in c.op.name and c.op.name.startswith("C2"):
            assert False, "sublattice-swapping C2(z) should not survive"
        if c.op.name == "inversion":
            assert False, "sublattice-swapping inversion should not survive"


def _matrix_key(M):
    # tuple of Python floats, not .tobytes(): Python floats compare/hash
    # -0.0 == 0.0, so this (unlike raw byte comparison) isn't fooled by
    # sign-of-zero differences between mathematically identical matrices
    return tuple(np.round(M, 6).flatten().tolist())


def test_close_group_produces_a_finite_group_containing_identity():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    found = pointgroup.find_point_group(g, h=h)
    group = pointgroup.close_group(h, found)
    assert any(c.op.name == "E" for c in group)
    keys = [_matrix_key(c.op.matrix) for c in group]
    assert len(keys) == len(set(keys))  # every element distinct
    # closed under composition: product of any two elements is in the group
    keyset = set(keys)
    for a in group:
        for b in group:
            prod = a.op.matrix @ b.op.matrix
            assert _matrix_key(prod) in keyset


def test_user_supplied_symmetry_not_present_is_rejected():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.intra[0, 0] += 0.7  # breaks C2(z)/inversion, see above
    c2z = pointgroup.SymmetryOperation(-np.eye(3), name="broken-inversion")
    assert pointgroup.compile_symmetry(h, c2z) is None
