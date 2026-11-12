import numpy as np
import pytest

from pyqula import geometry
from pyqula.kondotk import spinonspectrum
from pyqula.spinon import SpinonHamiltonian

# SpinonHamiltonian.get_kondo_spectrum evaluates the second- and third-order
# tunneling spectrum of Ternes, arXiv:1505.04430, on the state of a quadratic
# Hamiltonian, with Wick's theorem and the local spectral matrices of the site
# under the tip (kondotk/spinonspectrum.py). The reference of these tests is
# the same perturbation theory written as sums over the many-body eigenstates
# of the Fock space of a small cluster, which shares no code and no
# intermediate quantity with the module.

SIGMA = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
                 dtype=np.complex128)
EPS = np.zeros((3, 3, 3))
for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    EPS[i, j, k] = 1.
    EPS[i, k, j] = -1.


def thermal_step(x):
    """Derivative with the bias of the current between two Fermi seas, at
    x = energy/T"""
    x = np.asarray(x, dtype=float)
    out = np.where(x > 0, 1., 0.)
    mid = (np.abs(x) > 1e-4) & (np.abs(x) < 40)
    out[mid] = 0.5 + (np.sinh(x[mid]) - x[mid])/(2*(np.cosh(x[mid]) - 1))
    out[np.abs(x) <= 1e-4] = 0.5
    return out


def thermal_step_derivative(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    small = np.abs(x) < 0.05
    mid = ~small & (np.abs(x) < 40)
    out[small] = 1./6 - x[small]**2/60.
    half = 0.5*x[mid]
    out[mid] = (x[mid]*np.cosh(half) - 2*np.sinh(half))/(4*np.sinh(half)**3)
    return out


class ThermalLog():
    """The Kondo logarithm ln(w0+|e|) - ln(e^2+G0^2)/2 convolved with the
    derivative of the thermal step, by plain quadrature on a table"""
    def __init__(self, T, omega0, Gamma0, xmax=12.):
        # F varies over the scale T once it is broadened
        self.x = np.arange(-xmax, xmax, T/25.)
        u = np.linspace(-45*T, 45*T, 6001)
        kernel = thermal_step_derivative(u/T)/T
        self.f = np.empty(len(self.x))
        for i0 in range(0, len(self.x), 500):
            y = self.x[i0:i0+500, None] + u[None, :]
            log = np.log(omega0 + np.abs(y)) - 0.5*np.log(y**2 + Gamma0**2)
            self.f[i0:i0+500] = np.trapezoid(kernel[None, :]*log, u, axis=1)
    def __call__(self, x):
        assert np.max(np.abs(x)) < -self.x[0]
        return np.interp(x, self.x, self.f)


def fock_annihilation_operators(M):
    a = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    z, iden = np.diag([1., -1.]), np.eye(2)
    ops = []
    for i in range(M):
        m = np.array([[1.]], dtype=np.complex128)
        for j in range(M):
            m = np.kron(m, z if j < i else (a if j == i else iden))
        ops.append(m)
    return ops


def many_body_spectrum(h1, site, eVs, Jrho_s, T, omega0, Gamma0, Ts=None):
    """Second-order and third-order terms as sums over the eigenstates of
    the Fock space of the quadratic Hamiltonian h1, in the grand canonical
    ensemble at zero chemical potential and temperature Ts (T when not
    given); T is the temperature of the electrodes"""
    if Ts is None: Ts = T
    M = h1.shape[0]
    c = fock_annihilation_operators(M)
    H = sum(h1[i, j]*c[i].conj().T@c[j] for i in range(M) for j in range(M))
    E, V = np.linalg.eigh(H)
    p = np.exp(-(E - E.min())/Ts)
    p = p/p.sum()
    S = np.array([V.conj().T@sum(0.5*SIGMA[a][s, t]
                  *c[2*site+s].conj().T@c[2*site+t]
                  for s in range(2) for t in range(2))@V for a in range(3)])
    F = ThermalLog(T, omega0, Gamma0)
    eps = E[None, :] - E[:, None] # [i,f] = E_f - E_i
    M2 = np.sum(np.abs(S)**2, axis=0).T # [i,f]
    coeff = np.einsum('jkl,lif,kfm,jmi->ifm', EPS, S, S, S, optimize=True)
    pc = p[:, None, None]*coeff.imag/2.
    second, third = np.zeros(len(eVs)), np.zeros(len(eVs))
    for n, eV in enumerate(eVs):
        for v in (eV, -eV): # the two tunneling directions
            step = thermal_step((v - eps)/T)
            second[n] += np.sum(p[:, None]*M2*step)
            third[n] += np.einsum('ifm,if,im->', pc, step, F(v - eps))
            third[n] += np.einsum('ifm,if,fm->', pc, step, F(v + eps))
    return 2*np.pi*second, 4*np.pi*Jrho_s*third


def cluster(n, zeeman=None):
    """A chain of n sites with open ends and different onsite energies"""
    g = geometry.chain().get_supercell(n)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(lambda r: 0.37*np.sin(1.3*r[0] + 0.4))
    if zeeman is not None: h.add_zeeman(zeeman)
    return h


PARAMETERS = dict(Jrho_s=-0.05, T=0.05, omega0=2.0, Gamma0=0.01)


@pytest.mark.parametrize("zeeman", [None, [0.11, -0.23, 0.31]])
def test_agrees_with_the_sums_over_many_body_states(zeeman):
    """Both orders against the sums over the 16 states of the Fock space of
    two sites, without a field and with one in a generic direction, where
    the sites have a moment and the terms with an averaged spin operator
    enter. The exchange diagram is written here with E_f - E_m = -eps[f,m]
    transposed, so a wrong argument of its logarithm fails too."""
    h = cluster(2, zeeman)
    h1 = np.asarray(h.get_hk_gen()([0., 0., 0.]))
    eVs = np.linspace(-3., 3., 25)
    for site in range(2):
        ref2, ref3 = many_body_spectrum(h1, site, eVs, **PARAMETERS)
        y2 = spinonspectrum.kondo_spectrum(h, eVs, site=site, order=2, nk=1,
                de=PARAMETERS["T"]/8, **PARAMETERS)[1]
        y3 = spinonspectrum.kondo_spectrum(h, eVs, site=site, order=3, nk=1,
                de=PARAMETERS["T"]/8, **PARAMETERS)[1]
        assert np.max(np.abs(ref3)) > 0.1 # the third order is not small
        assert np.max(np.abs(y2 - ref2)) < 1e-3*np.max(np.abs(ref2))
        assert np.max(np.abs(y3 - y2 - ref3)) < 1e-3*np.max(np.abs(ref3))


def test_spinons_and_electrodes_at_different_temperatures():
    """Ts is the temperature of the spinon occupations and T the one of
    the step and of the logarithm: with a hot cluster under a cold tip the
    excited many-body states are occupied, and both orders still agree
    with the sums over the Fock space with those weights."""
    h = cluster(2, [0.11, -0.23, 0.31])
    h1 = np.asarray(h.get_hk_gen()([0., 0., 0.]))
    eVs = np.linspace(-3., 3., 25)
    ref2, ref3 = many_body_spectrum(h1, 0, eVs, Ts=0.4, **PARAMETERS)
    cold2 = many_body_spectrum(h1, 0, eVs, **PARAMETERS)[0]
    assert np.max(np.abs(ref2 - cold2)) > 0.05 # the two ensembles differ
    kw = dict(site=0, nk=1, de=PARAMETERS["T"]/8, Ts=0.4, **PARAMETERS)
    y2 = spinonspectrum.kondo_spectrum(h, eVs, order=2, **kw)[1]
    y3 = spinonspectrum.kondo_spectrum(h, eVs, order=3, **kw)[1]
    assert np.max(np.abs(y2 - ref2)) < 1e-3*np.max(np.abs(ref2))
    assert np.max(np.abs(y3 - y2 - ref3)) < 1e-3*np.max(np.abs(ref3))


def test_decoupled_site_is_half_of_a_spin_one_half():
    """A site without any exchange is a level at the Fermi energy with one
    fermion. The spectrum of a spin 1/2 is 2 pi [3/4 - (3/2) Jrho_s F(eV)],
    and the free fermion gives half of it at both orders, its local moment
    being 3/8. With the parameters of arXiv:1505.04430 (1 K, 20 meV, in
    meV) twice the spectrum is the one of its figure for B = 0, 1.13 at
    zero bias and 0.886 at 4 mV."""
    g = geometry.chain()
    g.dimensionality = 0
    h = SpinonHamiltonian(g)
    kT = 8.617333262e-2 # 1 K in meV
    eVs = np.array([0., 0.5, 4.])
    (es, y) = h.get_kondo_spectrum(eVs, site=0, Jrho_s=-0.05, T=kT,
            omega0=20., Gamma0=5e-3)
    F = ThermalLog(kT, 20., 5e-3, xmax=5.)
    spin_half = 2*np.pi*(0.75 + 1.5*0.05*F(eVs))
    assert np.max(np.abs(2*y - spin_half)) < 2e-3
    assert abs(2*y[0]/(2*np.pi) - 1.13) < 0.01
    assert abs(2*y[2]/(2*np.pi) - 0.886) < 0.005
    y2 = h.get_kondo_spectrum(eVs, site=0, order=2, T=kT)[1]
    assert np.max(np.abs(2*y2/(2*np.pi) - 0.75)) < 1e-10


def test_supercell_gives_the_spectrum_of_the_primitive_cell():
    """The same chain in a cell of one site and of three, on k-meshes that
    hold the same momenta: every site of the large cell has the spectrum
    of the site of the small one. The Zeeman field makes the path with the
    four components of the spectral matrices the one that runs."""
    eVs = np.linspace(-2., 2., 21)
    kw = dict(T=0.05, omega0=2.0, de=0.01)
    spectra = []
    for n in (1, 3):
        h = geometry.chain().get_supercell(n).get_hamiltonian(has_spin=True)
        h.add_zeeman([0.1, 0.05, -0.2])
        spectra.append(spinonspectrum.kondo_spectrum(h, eVs, nk=12//n,
                                                     **kw)[1])
    assert spectra[1].shape == (3, len(eVs))
    assert np.max(np.abs(spectra[0])) > 0.5
    assert np.max(np.abs(spectra[1] - spectra[0])) < 1e-8


def test_large_bias_counts_the_local_moment():
    """Above the bandwidth every spin flip is open, and the second order is
    2 pi <S^2> of the site: 3/8 for a half-filled band without a moment."""
    h = geometry.chain().get_hamiltonian(has_spin=True)
    y = spinonspectrum.kondo_spectrum(h, [6.0], site=0, order=2, T=0.02,
                                      nk=200)[1]
    assert abs(y[0]/(2*np.pi) - 0.375) < 1e-6


def trimerized_chain():
    """Spinon mean field of a chain of trimers: exchange 1 inside a trimer
    and 0.1 between two of them"""
    np.random.seed(0)
    g = geometry.chain().get_supercell(3)
    xmin, xmax = np.min(g.r[:, 0]), np.max(g.r[:, 0])
    def J(r1, r2):
        if abs(np.linalg.norm(r1 - r2) - 1.) > 1e-3: return 0.
        inside = xmin < (r1[0] + r2[0])/2. < xmax
        return 1.0 if inside else 0.1
    h = SpinonHamiltonian(g)
    return h.get_mean_field_hamiltonian(Jr=J, rcut=1.5, decoupling="bond",
            ansatz=g.get_hamiltonian(has_spin=True), hold="phases", nk=40,
            T=0.01, mix=0.5)


def test_zero_bias_peak_needs_a_spinon_level_at_the_fermi_energy():
    """The uniform chain has a continuum of spinons: the spin spectrum
    starts linearly, the conductance starts as the square of the bias, and
    the third order adds no peak. A trimer has one spinon in a level at the
    Fermi energy that lives on its two end sites: there the Kondo term
    gives a maximum at zero bias, and the middle site, where that level
    has no weight, stays dark."""
    eVs = np.array([0., 0.1])
    kw = dict(T=0.005, omega0=2.0, nk=200)
    h = SpinonHamiltonian(geometry.chain())
    h = h.get_mean_field_hamiltonian(J1=1.0, decoupling="bond", nk=40,
                                     T=0.01)
    y = h.get_kondo_spectrum(eVs, site=0, **kw)[1]
    assert y[0] < 0.1*y[1] # no weight at zero bias
    h = trimerized_chain()
    y2 = h.get_kondo_spectrum(eVs, order=2, **kw)[1]
    y3 = h.get_kondo_spectrum(eVs, order=3, **kw)[1]
    for end in (0, 2):
        assert y2[end, 0] > 0.5 # elastic spin flips of the level
        assert abs(y2[end, 0] - y2[end, 1]) < 1e-3 # a flat second order
        assert y3[end, 0] > 1.05*y3[end, 1] # the peak of the third order
    assert np.max(np.abs(y3[1])) < 1e-3 # the middle site


def test_arguments_are_checked():
    h = geometry.chain().get_hamiltonian(has_spin=True)
    with pytest.raises(ValueError, match="accepted values are 2, 3"):
        spinonspectrum.kondo_spectrum(h, [0.], order=4)
    with pytest.raises(ValueError, match="site=5"):
        spinonspectrum.kondo_spectrum(h, [0.], site=5)
    with pytest.raises(ValueError):
        spinonspectrum.kondo_spectrum(
                geometry.chain().get_hamiltonian(has_spin=False), [0.])
    h.setup_nambu_spinor()
    with pytest.raises(NotImplementedError, match="Nambu"):
        spinonspectrum.kondo_spectrum(h, [0.])
