import numpy as np

from pyqula.qutecipytk.integration import integrate

rng = np.random.default_rng(1234)


def test_integrate_real_polynomials():
    coefficients = [
        0.23637074801483304, 0.20661524945577847, 0.1850826417895819, 0.8433788714289417,
        0.5801482873508491, 0.20339438932656262, 0.21593267492457668, 0.8052490409622802,
        0.7189346124875339, 0.9400806688257749, 0.355210845205325, 0.5251561513473092,
        0.6819965273401778, 0.9221987248861162, 0.04166444723413998,
    ]

    def polynomial(x):
        return sum(c * x ** i for i, c in enumerate(coefficients))

    def polynomial_integral(x):
        return sum(c * x ** (i + 1) / (i + 1) for i, c in enumerate(coefficients))

    def f(x):
        p = 1.0
        for xi in x:
            p *= polynomial(xi)
        return p

    N = 5
    exactval = polynomial_integral(1.0) ** N
    got = integrate(np.float64, f, [0.0] * N, [1.0] * N)
    assert np.isclose(got, exactval)

    a = rng.random(N)
    b = rng.random(N)
    exactval = 1.0
    for ai, bi in zip(a, b):
        exactval *= polynomial_integral(bi) - polynomial_integral(ai)
    got = integrate(np.float64, f, a, b)
    assert np.isclose(got, exactval)


def test_integrate_complex_polynomials():
    coefficients = [
        0.3593310036882851 + 0.5576449076512986j, 0.21621263269455804 + 0.08800235200842366j,
        0.46808012645915487 + 0.1415909017229775j, 0.24698289079616775 + 0.9658935330334624j,
        0.4554497486419905 + 0.7984680137635275j, 0.6957883252881866 + 0.9570499781505035j,
        0.289419938556415 + 0.9984881496050377j, 0.9577173946390758 + 0.5442255738586897j,
        0.7034120251166891 + 0.3168670299990256j, 0.7305752395989986 + 0.14383153865593656j,
    ]

    def polynomial(x):
        return sum(c * x ** i for i, c in enumerate(coefficients))

    def polynomial_integral(x):
        return sum(c * x ** (i + 1) / (i + 1) for i, c in enumerate(coefficients))

    def f(x):
        p = 1.0
        for xi in x:
            p *= polynomial(xi)
        return p

    N = 5
    exactval = polynomial_integral(1.0) ** N
    got = integrate(np.complex128, f, [0.0] * N, [1.0] * N)
    assert np.isclose(got, exactval)

    a = rng.random(N)
    b = rng.random(N)
    exactval = 1.0
    for ai, bi in zip(a, b):
        exactval *= polynomial_integral(bi) - polynomial_integral(ai)
    got = integrate(np.complex128, f, a, b)
    assert np.isclose(got, exactval)


def test_integrate_10d_function():
    def f(x):
        x = np.asarray(x)
        return 1000 * np.cos(10 * np.sum(x ** 2)) * np.exp(-np.sum(x) ** 4 / 1000)

    I15 = integrate(np.float64, f, [-1.0] * 10, [1.0] * 10, GKorder=15, tolerance=1e-8)
    Iref = -5.4960415218049
    assert abs(I15 - Iref) < 1e-3
