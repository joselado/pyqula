import numpy as np

from pyqula.qutecipytk.gausskronrod import kronrod

# QuadGK.jl's own hardcoded double-precision n=7 rule (src/gausskronrod.jl xd7/wd7/wgd7),
# the half-array (x <= 0) form, computed once in 100-bit arithmetic.
_XD7 = [
    -9.9145537112081263920685469752598e-01, -9.4910791234275852452618968404809e-01,
    -8.6486442335976907278971278864098e-01, -7.415311855993944398638647732811e-01,
    -5.8608723546769113029414483825842e-01, -4.0584515137739716690660641207707e-01,
    -2.0778495500789846760068940377309e-01, 0.0,
]
_WD7 = [
    2.2935322010529224963732008059913e-02, 6.3092092629978553290700663189093e-02,
    1.0479001032225018383987632254189e-01, 1.4065325971552591874518959051021e-01,
    1.6900472663926790282658342659795e-01, 1.9035057806478540991325640242055e-01,
    2.0443294007529889241416199923466e-01, 2.0948214108472782801299917489173e-01,
]
_WGD7 = [
    1.2948496616886969327061143267787e-01, 2.797053914892766679014677714229e-01,
    3.8183005050511894495036977548818e-01, 4.1795918367346938775510204081658e-01,
]


def test_n7_matches_quadgk_hardcoded_constants():
    x, w, wg = kronrod(7, -1, 1)
    n = 7
    # our x/w/wg are the full mirrored (2n+1)-point arrays; the reference is the half array.
    assert np.allclose(x[: n + 1], _XD7, atol=1e-13)
    assert np.allclose(w[: n + 1], _WD7, atol=1e-13)
    assert np.allclose(wg, _WGD7 + _WGD7[-2::-1], atol=1e-13)


def test_symmetry_and_normalization():
    for n in [3, 5, 7, 10, 15, 21]:
        x, w, wg = kronrod(n, -1, 1)
        assert len(x) == 2 * n + 1
        assert np.allclose(x, -x[::-1], atol=1e-14)
        assert np.allclose(w, w[::-1], atol=1e-14)
        # integral of the unit weight function over [-1,1] is 2.
        assert np.isclose(np.sum(w), 2.0, atol=1e-12)
        assert np.isclose(np.sum(wg), 2.0, atol=1e-12)


def test_exact_for_polynomials_up_to_degree():
    # A 2n+1-point Kronrod rule is exact for polynomials up to degree ~3n+1.
    n = 7
    x, w, _ = kronrod(n, -1, 1)
    for deg in range(0, 3 * n):
        exact = 0.0 if deg % 2 == 1 else 2.0 / (deg + 1)
        approx = np.sum(w * x ** deg)
        assert np.isclose(approx, exact, atol=1e-10), f"degree {deg} failed"


def test_rescale_to_arbitrary_interval():
    n = 7
    a, b = 2.0, 5.0
    x, w, _ = kronrod(n, a, b)
    assert np.isclose(np.sum(w), b - a)
    approx = np.sum(w * np.exp(x))
    exact = np.exp(b) - np.exp(a)
    assert np.isclose(approx, exact, atol=1e-10)
