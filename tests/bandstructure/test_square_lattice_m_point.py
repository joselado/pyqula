from pyqula import geometry


def test_square_lattice_path_through_m_reaches_both_band_edges():
    """On the square lattice "M" is the zone corner (1/2,1/2), where the
    first-neighbor band 2(cos kx + cos ky) has the opposite sign to the one
    at Gamma. It used to resolve to the hexagonal M, (1/2,0), the same point
    as X, so a G-X-M-G path never left the line ky=0 and stopped at zero
    energy instead of reaching the far band edge."""
    h = geometry.square_lattice().get_hamiltonian()
    (k, e) = h.get_bands(kpath=["G", "X", "M", "G"])
    assert abs(e.min() - (-4.)) < 1e-6, e.min()
    assert abs(e.max() - 4.) < 1e-6, e.max()


def test_hexagonal_m_point_is_unchanged():
    """On hexagonal lattices M stays the edge center (1/2,0)."""
    from pyqula.kpointstk import labels
    for g in [geometry.honeycomb_lattice(), geometry.triangular_lattice()]:
        assert list(labels.label2k(g, "M")) == [.5, 0., 0.]
