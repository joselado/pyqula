import numpy as np
from scipy.special import wofz

from pyqula import geometry
from pyqula import kpm


def _rashba_chain():
    """A spinful chain with Rashba coupling, an in-plane Zeeman field and
    onsite disorder, so that the correlators between different orbitals
    are complex and not symmetric in i and j"""
    np.random.seed(0)
    g = geometry.chain().get_supercell(30)
    g.dimensionality = 0
    h = g.get_hamiltonian()
    h.add_rashba(0.5)
    h.add_zeeman([0., 1., 0.])
    h.intra += np.diag(np.random.random(h.intra.shape[0]))
    m = h.intra.toarray() if hasattr(h.intra, "toarray") else np.array(h.intra)
    return m


def _gaussian_green(m, i, j, es, scale, nmoments):
    """Exact G^R_ij(E) = <i|(E + i0 - H)^-1|j>, with each level broadened
    by the Gaussian that the Jackson kernel approximates, of standard
    deviation pi*scale*sqrt(1-(E_n/scale)^2)/N (Weisse et al., Rev. Mod.
    Phys. 78, 275 (2006), Eqs. (75) and (76)). A Gaussian-broadened pole is
    the Faddeeva function w(z)"""
    E, V = np.linalg.eigh(m)
    w = V[i, :]*np.conj(V[j, :]) # <i|n><n|j>
    sig = np.pi*scale/nmoments*np.sqrt(1.-(E/scale)**2)
    z = (es[:, None]-E[None, :])/(np.sqrt(2.)*sig[None, :])
    return np.sum(w[None, :]*(-1j*np.sqrt(np.pi/2.)/sig[None, :])*wofz(z),
                  axis=1)


def test_correlator0d_returns_real_part_and_minus_imaginary_part(
        tmp_path, monkeypatch):
    """kpm.correlator0d returns (E, Re G^R_ij, -Im G^R_ij), the same pair
    that correlator.correlator0d returns from the matrix inversion, so that
    for i=j the second array is pi times the local DOS. Joined as gr+1j*gi
    it is therefore the complex conjugate of G^R_ij, which a physics review
    of the notebooks read as a sign error; it is the convention, and the
    test holds it in place against an exact eigendecomposition"""
    monkeypatch.chdir(tmp_path)
    m = _rashba_chain()
    npol, scale = 300, 10.
    for (i, j) in [(0, 0), (0, 9), (3, 4)]:
        (x, gr, gi) = kpm.correlator0d(m, i=i, j=j, npol=npol, ne=801,
                                       scale=scale, write=False)
        G = _gaussian_green(m, i, j, x, scale, 2*npol)
        err = np.max(np.abs((gr-1j*gi)-G))/np.max(np.abs(G))
        err_conj = np.max(np.abs((gr+1j*gi)-G))/np.max(np.abs(G))
        assert err < 0.05 # the Jackson peak is only nearly a Gaussian
        assert err_conj > 0.5
    # for i=j the imaginary part is pi times the local DOS of the same
    # expansion, exactly
    (x, gr, gi) = kpm.correlator0d(m, i=5, j=5, npol=npol, ne=801,
                                   write=False)
    (x2, d) = kpm.ldos(m, i=5, npol=npol, ne=801)
    assert np.allclose(x, x2)
    assert np.allclose(gi, np.pi*d, atol=1e-10)


def test_correlator0d_file_has_the_columns_it_returns(tmp_path, monkeypatch):
    """CORRELATOR_KPM.OUT used to hold (E, Im G^R, Re G^R), with the second
    column the negative of the local DOS, while the arrays returned and the
    CORRELATOR.OUT that correlator.correlator0d writes both hold
    (E, Re G^R, -Im G^R)"""
    monkeypatch.chdir(tmp_path)
    m = _rashba_chain()
    (x, gr, gi) = kpm.correlator0d(m, i=2, j=2, npol=100, ne=200, write=True)
    d = np.genfromtxt("CORRELATOR_KPM.OUT")
    assert np.allclose(d[:, 0], x)
    assert np.allclose(d[:, 1], gr)
    assert np.allclose(d[:, 2], gi)
    assert np.min(d[:, 2]) > -1e-6 # a local DOS, up to roundoff


def test_dm_ij_energy_is_the_spectral_part_of_the_green_function():
    """kpm.dm_ij_energy returns pi<i|delta(E-H)|j> = (i/2)[G^R_ij - G^A_ij]
    with G^A_ij = conj(G^R_ji), which for i=j is pi times the local DOS and
    for i!=j integrates to zero over the energy"""
    m = _rashba_chain()
    npol, scale = 300, 10.
    for (i, j) in [(0, 0), (0, 9), (3, 4)]:
        (x, y) = kpm.dm_ij_energy(m, i=i, j=j, npol=npol, ne=801, scale=scale)
        GR_ij = _gaussian_green(m, i, j, x, scale, 2*npol)
        GR_ji = _gaussian_green(m, j, i, x, scale, 2*npol)
        A = 0.5j*(GR_ij-np.conj(GR_ji))
        assert np.max(np.abs(y-A))/np.max(np.abs(A)) < 0.05
        assert np.isclose(np.trapezoid(y, x), np.pi*float(i == j), atol=2e-2)


def test_dm_vivj_energy_matches_dm_ij_energy():
    """dm_vivj_energy with the two site vectors is dm_ij_energy. It used to
    damp the moments of the real part with the Lorentz kernel and those of
    the imaginary part with the Jackson kernel, two different broadenings
    of the same function"""
    m = _rashba_chain()
    n = m.shape[0]
    for (i, j) in [(0, 0), (0, 9), (3, 4)]:
        vi = np.zeros(n, dtype=complex)
        vj = np.zeros(n, dtype=complex)
        vi[i] = 1.
        vj[j] = 1.
        (x, y) = kpm.dm_ij_energy(m, i=i, j=j, npol=200, ne=400)
        (x2, y2) = kpm.dm_vivj_energy(m, vi, vj, npol=200, ne=400)
        assert np.allclose(x, x2)
        assert np.allclose(y, y2, atol=1e-10)


def test_correlator0d_lorentz_kernel_is_the_green_function_at_i_delta(
        tmp_path, monkeypatch):
    """With the Lorentz kernel (lambda=3) the expansion broadens each pole
    into a Lorentzian of half width 3*scale/N for N=2*npol moments
    (Weisse et al., Rev. Mod. Phys. 78, 275 (2006), Sec. II.C.4), so it is
    the G^R(E + i delta) of a matrix inversion with that delta. Not
    exactly: the kernel stops at N moments, where exp(-lambda n/N) is
    still exp(-3) = 0.05, which on a single pole leaves an error of 10% of
    its height, and the width narrows as sqrt(1-(E/scale)^2); the complex
    conjugate, which the same arrays give when joined the other way, is off
    by more than the function itself"""
    monkeypatch.chdir(tmp_path)
    from pyqula import correlator
    m = _rashba_chain()
    delta, scale = 0.1, 10.
    npol = int(round(3.*scale/(2.*delta)))
    es = np.linspace(-4., 5., 451)
    for (i, j) in [(0, 0), (0, 9)]:
        (x, gr, gi) = kpm.correlator0d(m, i=i, j=j, npol=npol, x=es,
                                       scale=scale, kernel="lorentz",
                                       write=False)
        (x2, zr, zi) = correlator.correlator0d(m, energies=es, i=i, j=j,
                                               delta=delta, write=False)
        G = zr-1j*zi
        assert np.max(np.abs((gr-1j*gi)-G))/np.max(np.abs(G)) < 0.12
        assert np.max(np.abs((gr+1j*gi)-G))/np.max(np.abs(G)) > 0.5
    assert not (tmp_path/"CORRELATOR.OUT").exists() # write=False
