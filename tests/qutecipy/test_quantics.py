import itertools

import numpy as np
import pytest

from pyqula.qutecipytk.quantics import DiscretizedGrid, InherentDiscreteGrid, quantics_function


def test_1d_roundtrip_base2():
    R = 5
    g = InherentDiscreteGrid.from_resolutions(R, (0,))
    for gi in range(2 ** R):
        q = g.grididx_to_quantics(gi)
        assert len(q) == R
        back = g.quantics_to_grididx(q)
        assert back == (gi,)


def test_1d_quantics_matches_binary_digits():
    # bitnumber 0 = MSB, matching Julia's bitnumber=1-is-MSB convention shifted to 0-based.
    R = 3
    g = InherentDiscreteGrid.from_resolutions(R, (0,))
    for gi in range(8):
        q = g.grididx_to_quantics(gi)
        expected = [(gi >> (R - 1 - k)) & 1 for k in range(R)]
        assert q == expected


@pytest.mark.parametrize("scheme", ["fused", "interleaved", "grouped"])
def test_2d_roundtrip_all_unfolding_schemes(scheme):
    Rs = (2, 3)
    g = InherentDiscreteGrid.from_resolutions(Rs, (0, 0), unfoldingscheme=scheme)
    for i, j in itertools.product(range(2 ** Rs[0]), range(2 ** Rs[1])):
        q = g.grididx_to_quantics((i, j))
        back = g.quantics_to_grididx(q)
        assert back == (i, j)


def test_fused_dimension_order_matches_reference():
    # QuanticsGrids.jl's :fused adds dimensions in reverse order per site
    # (first dimension varies fastest) -- cross-checked against the actual
    # Julia output for InherentDiscreteGrid{2}((2,2),(0,0);unfoldingscheme=:fused).
    g = InherentDiscreteGrid.from_resolutions((2, 2), (0, 0), unfoldingscheme="fused")
    assert g.indextable == [[("1", 0), ("0", 0)], [("1", 1), ("0", 1)]]
    cases = {
        (0, 0): [0, 0], (0, 1): [0, 2], (0, 2): [2, 0], (0, 3): [2, 2],
        (1, 0): [0, 1], (2, 2): [3, 0], (3, 3): [3, 3],
    }
    for gi, q in cases.items():
        assert g.grididx_to_quantics(gi) == q


def test_mixed_base():
    g = InherentDiscreteGrid.from_resolutions((2, 2), (0, 0), base=(2, 6))
    for i, j in itertools.product(range(4), range(36)):
        q = g.grididx_to_quantics((i, j))
        assert g.quantics_to_grididx(q) == (i, j)


def test_custom_indextable():
    # Irregular, non-monotonic index table mixing variables per site.
    variablenames = ("a", "b")
    indextable = [[("a", 0), ("b", 1)], [("a", 1)], [("b", 0), ("a", 2)]]
    g = InherentDiscreteGrid.from_indextable(variablenames, indextable, base=2)
    assert g.Rs == (3, 2)
    for i, j in itertools.product(range(8), range(4)):
        q = g.grididx_to_quantics((i, j))
        assert g.quantics_to_grididx(q) == (i, j)


def test_rs_zero_dimension():
    g = InherentDiscreteGrid.from_resolutions((3, 0), (0, 0))
    assert g.grid_max()[1] == g.grid_min()[1]
    for gi in range(8):
        q = g.grididx_to_quantics((gi, 0))
        back = g.quantics_to_grididx(q)
        assert back == (gi, 0)
    with pytest.raises(ValueError):
        g.grididx_to_quantics((0, 1))  # dimension 1 has only index 0 valid


def test_step_and_origin():
    g = InherentDiscreteGrid.from_resolutions(3, (10,), step=(2,))
    assert g.grididx_to_origcoord(0) == (10,)
    assert g.grididx_to_origcoord(7) == (10 + 7 * 2,)
    assert g.origcoord_to_grididx((10,)) == (0,)
    assert g.origcoord_to_grididx((24,)) == (7,)


# -- DiscretizedGrid ---------------------------------------------------------


def test_discretized_grid_unit_interval():
    R = 5
    g = DiscretizedGrid.from_resolutions(("x",), (R,))
    assert np.isclose(g.grid_min()[0], 0.0)
    assert np.isclose(g.grid_max()[0], 1.0 - 1.0 / 2 ** R)
    assert np.isclose(g.grid_step()[0], 1.0 / 2 ** R)

    for x in [0.0, 0.1, 0.3, 0.5, 0.9999]:
        gi = g.origcoord_to_grididx((x,))
        back = g.grididx_to_origcoord(gi)
        assert 0 <= gi[0] < 2 ** R
        assert abs(back[0] - x) < g.grid_step()[0]


def test_discretized_grid_includeendpoint():
    R = 3
    g = DiscretizedGrid.from_resolutions(("x",), (R,), includeendpoint=True)
    assert np.isclose(g.grid_max()[0], 1.0)
    assert np.isclose(g.grididx_to_origcoord(0)[0], 0.0)
    assert np.isclose(g.grididx_to_origcoord(2 ** R - 1)[0], 1.0)

    with pytest.raises(ValueError):
        DiscretizedGrid.from_resolutions(("x",), (0,), includeendpoint=True)


def test_discretized_grid_nonunit_domain():
    R = 4
    g = DiscretizedGrid.from_resolutions(("x",), (R,), lower_bound=(-2.0,), upper_bound=(3.0,))
    assert np.isclose(g.grididx_to_origcoord(0)[0], -2.0)
    step = (3.0 - (-2.0)) / 2 ** R
    assert np.isclose(g.grididx_to_origcoord(1)[0], -2.0 + step)


def test_discretized_grid_out_of_bounds():
    g = DiscretizedGrid.from_resolutions(("x",), (4,))
    with pytest.raises(ValueError):
        g.origcoord_to_grididx((1.5,))
    with pytest.raises(ValueError):
        g.origcoord_to_grididx((-0.1,))


def test_localdimensions_and_sitedim():
    g = DiscretizedGrid.from_resolutions(("x", "y"), (2, 2), unfoldingscheme="fused")
    dims = g.localdimensions()
    assert len(dims) == len(g)
    for i, d in enumerate(dims):
        assert g.sitedim(i) == d


def test_quantics_function_adapter():
    g = DiscretizedGrid.from_resolutions(("x",), (6,))

    def fx(x):
        return np.exp(-x)

    qf = quantics_function(np.float64, g, fx)
    for x in [0.1, 0.4, 0.7]:
        q = g.origcoord_to_quantics((x,))
        got = qf(q)
        expect = fx(g.quantics_to_origcoord(q)[0])
        assert np.isclose(got, expect)


def test_float_edge_cases_small_step():
    # R large enough that step is tiny; round-trip should still be exact.
    R = 20
    g = DiscretizedGrid.from_resolutions(("x",), (R,), lower_bound=(0.0,), upper_bound=(1.0,))
    for gi in [0, 1, 2 ** R - 1, 2 ** (R - 1)]:
        x = g.grididx_to_origcoord(gi)
        back = g.origcoord_to_grididx(x)
        assert back == (gi,)


def test_float_edge_cases_large_offset():
    R = 10
    g = DiscretizedGrid.from_resolutions(("x",), (R,), lower_bound=(1e8,), upper_bound=(1e8 + 1.0,))
    for gi in [0, 5, 2 ** R - 1]:
        x = g.grididx_to_origcoord(gi)
        back = g.origcoord_to_grididx(x)
        assert back == (gi,)
