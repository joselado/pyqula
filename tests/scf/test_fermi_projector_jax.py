import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pyqula.scftk.densitydensity_jax import fermi_projector


def _plain_projector(hks, mu, T):
    """The composition fermi_projector replaces: eigh, then V f(E) V^dagger,
    differentiated by jax through eigh's eigenvector tangent"""
    es, vs = jnp.linalg.eigh(hks)
    occ = jax.nn.sigmoid(-(es - mu) / T)
    return jnp.einsum('kie,ke,kje->kij', vs, occ, jnp.conj(vs)), es


def _random_batch(rng, nk, n):
    a = rng.standard_normal((nk, n, n)) + 1j*rng.standard_normal((nk, n, n))
    return a


def _degenerate_batch(rng, n=6):
    """Hamiltonians with exactly repeated eigenvalues: a diagonal matrix,
    which every eigensolver returns bit-identical levels for, and a random
    unitary rotation of a spin-doubled spectrum"""
    d = np.diag([-1., -1., -1., .3, .3, 2.]).astype(complex)
    q, _ = np.linalg.qr(_random_batch(rng, 1, n)[0])
    e = np.repeat(rng.standard_normal(n//2), 2)
    rotated = q @ np.diag(e) @ q.conj().T
    return np.array([d, (rotated + rotated.conj().T)/2])


def test_value_matches_the_eigh_composition():
    rng = np.random.default_rng(0)
    hks = _random_batch(rng, 5, 7)
    P, es = fermi_projector(jnp.asarray(hks), 0.1, 0.05)
    P0, es0 = _plain_projector(jnp.asarray(hks), 0.1, 0.05)
    assert np.allclose(P, P0, atol=1e-12)
    assert np.allclose(es, es0, atol=1e-12)


@pytest.mark.parametrize("T", [0.3, 1e-2])
def test_derivative_matches_jax_autodiff_away_from_degeneracies(T):
    """Random matrices have no degeneracies, so autodiff through eigh is
    exact there and must agree with the Daleckii-Krein rule, for tangents
    in H (non-Hermitian on purpose, eigh symmetrizes), mu and T"""
    rng = np.random.default_rng(1)
    hks = jnp.asarray(_random_batch(rng, 4, 6))
    dh = jnp.asarray(_random_batch(rng, 4, 6))
    primals, tangents = (hks, 0.2, T), (dh, 0.7, 0.3*T)
    _, (dP, des) = jax.jvp(fermi_projector, primals, tangents)
    _, (dP0, des0) = jax.jvp(_plain_projector, primals, tangents)
    scale = np.max(np.abs(dP0))
    assert np.max(np.abs(dP - dP0))/scale < 1e-9
    assert np.max(np.abs(des - des0)) < 1e-9


def test_derivative_is_finite_and_exact_at_degeneracies():
    """Where eigh's eigenvector tangent divides by zero, the projector
    derivative must still be finite and match a central finite difference
    (P is smooth in H even though V is not)"""
    rng = np.random.default_rng(2)
    hks = _degenerate_batch(rng)
    dh = _random_batch(rng, 2, 6)
    mu, T = 0.1, 0.2
    _, (dP, _) = jax.jvp(fermi_projector, (jnp.asarray(hks), mu, T),
                         (jnp.asarray(dh), 0., 0.))
    assert np.all(np.isfinite(dP))
    eps = 1e-5
    Pp, _ = fermi_projector(jnp.asarray(hks + eps*dh), mu, T)
    Pm, _ = fermi_projector(jnp.asarray(hks - eps*dh), mu, T)
    fd = (np.asarray(Pp) - np.asarray(Pm))/(2*eps)
    assert np.max(np.abs(dP - fd)) < 1e-7


def test_reverse_mode_matches_jax_autodiff_and_stays_finite():
    """jax.vjp/jax.grad (the lbfgs and levenberg_marquardt solvers) get the
    rule transposed automatically. Away from degeneracies that must match
    jax's own reverse mode through eigh; at a degeneracy it must be finite"""
    rng = np.random.default_rng(3)
    hks = jnp.asarray(_random_batch(rng, 3, 5))
    ct = (jnp.asarray(_random_batch(rng, 3, 5)), jnp.asarray(rng.standard_normal((3, 5))))
    _, vjp = jax.vjp(lambda h, mu: fermi_projector(h, mu, 0.1), hks, 0.2)
    _, vjp0 = jax.vjp(lambda h, mu: _plain_projector(h, mu, 0.1), hks, 0.2)
    (gh, gmu), (gh0, gmu0) = vjp(ct), vjp0(ct)
    assert np.max(np.abs(gh - gh0))/np.max(np.abs(gh0)) < 1e-9
    assert abs(gmu - gmu0) < 1e-9*max(1., abs(gmu0))
    hks = jnp.asarray(_degenerate_batch(rng))
    _, vjp = jax.vjp(lambda h: fermi_projector(h, 0.1, 0.2)[0], hks)
    (gh,) = vjp(jnp.asarray(_random_batch(rng, 2, 6)))
    assert np.all(np.isfinite(gh))
