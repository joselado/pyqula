"""Correctness of the X-wave collinear magnet models
(src/pyqula/specialhamiltoniantk/xwave.py).

These are the lattice models of Ezawa, arXiv:2411.16036 (Phys. Rev. B 111,
125420 (2025)). Everything the nonlinear-response selection rule keys off is
a property of the spin-splitting form factor, so the tests here pin the form
factor rather than any derived observable: that it reduces to the right
k-space harmonic near Gamma with the right prefactor (which is how those
prefactors were fixed in the first place -- the paper's own text garbles some
of them, and its g-wave tight-binding model disagrees with its own continuum
equation by a factor of -2), that it has the right number of nodes, and that
it carries no net magnetization.
"""
import numpy as np
import pytest

from pyqula import algebra
from pyqula import specialhamiltonian
from pyqula.specialhamiltoniantk import xwave
from pyqula.klist import kmesh


# The continuum harmonics of Ezawa's Eqs. (2)-(6), and the number of nodal
# lines of each. A degree-n harmonic has n nodal lines through the origin,
# hence 2n sign changes around a loop enclosing it -- Ezawa's "l+1 nodes"
# counts the lines.
WAVES = {
    "p":(lambda kx,ky: kx,                                             1),
    "d":(lambda kx,ky: kx*ky,                                          2),
    "f":(lambda kx,ky: kx*(kx**2-3*ky**2),                             3),
    "g":(lambda kx,ky: kx*ky*(kx**2-ky**2),                            4),
    "i":(lambda kx,ky: kx*ky*(3*kx**2-ky**2)*(kx**2-3*ky**2),          6),
}


def _form_factor(h,kcart):
    """The spin splitting (E_up - E_dn)/2 = J F_X(k) at a Cartesian k.

    Every one of these models is one orbital per spin per cell, so the Bloch
    matrix is 2x2 and diagonal and the splitting can be read straight off it
    without any band-pairing ambiguity."""
    g = h.geometry
    kcart = np.array(kcart,dtype=np.float64)
    # reduced coordinates: k_red_i = k_cart . a_i/(2 pi)
    kred = np.array([kcart.dot(g.a1[0:2]),kcart.dot(g.a2[0:2])])/(2.*np.pi)
    m = np.array(h.get_hk_gen()([kred[0],kred[1],0.]))
    return ((m[0,0]-m[1,1])/2.).real


@pytest.mark.parametrize("wave",list(WAVES))
def test_form_factor_reduces_to_the_continuum_harmonic(wave):
    """Near Gamma the lattice form factor must equal Ezawa's continuum
    harmonic Eqs. (2)-(6) exactly, prefactor included.

    This is the test that fixes the normalization constants in xwave.py (1,
    1, -4, -2 and 16/(3 sqrt 3) for p, d, f, g, i). It is not a restatement
    of the paper's own tight-binding prefactors: two of those are
    inconsistent with the paper's continuum equations, and this check picks
    the continuum equations as the definition. Anything that rescaled a form
    factor -- a wrong prefactor, a bond vector of the wrong length, a
    dropped sine -- moves this ratio away from 1."""
    harmonic,_ = WAVES[wave]
    h = xwave.xwave_magnet(wave=wave,J=1.,t=-1.)
    rng = np.random.RandomState(0)
    for k in rng.uniform(-0.02,0.02,size=(8,2)):
        got = _form_factor(h,k)
        want = harmonic(k[0],k[1])
        assert np.isclose(got,want,rtol=2e-3)


@pytest.mark.parametrize("wave",list(WAVES))
def test_node_count_around_a_loop(wave):
    """The form factor of an X-wave magnet has as many nodal lines as the
    order of its harmonic, so it changes sign 2n times around a loop
    enclosing Gamma: 2, 4, 6, 8 and 12 for p, d, f, g and i.

    This is what distinguishes the waves from one another, and it is the
    property the selection rule ultimately measures. A form factor that
    degenerated into a uniform spin-dependent bandwidth -- the failure mode
    of putting a cos(n theta) factor on nearest-neighbour bonds alone, where
    all six triangular bond directions sit at cos(6 theta) = 1 -- would show
    up here as zero sign changes."""
    _,nlines = WAVES[wave]
    h = xwave.xwave_magnet(wave=wave,J=1.,t=-1.)
    # offset grid: a grid containing phi = 0, pi/2, ... lands exactly on the
    # nodes, and each sampled zero would be counted as two sign changes
    n = 2000
    phis = (np.arange(n)+0.5)*2.*np.pi/n
    r = 0.3 # small enough that the leading harmonic dominates
    vals = np.array([_form_factor(h,[r*np.cos(p),r*np.sin(p)]) for p in phis])
    vals = vals[np.abs(vals)>1e-13*np.max(np.abs(vals))]
    signs = np.sign(vals)
    changes = int(np.sum(signs[1:]!=signs[:-1]))
    changes += int(signs[0]!=signs[-1]) # close the loop
    assert changes == 2*nlines


@pytest.mark.parametrize("wave",list(WAVES))
def test_no_net_magnetization_and_a_real_splitting(wave):
    """An X-wave magnet is compensated: the spin splitting averages to zero
    over the Brillouin zone, so there is no net moment, while the splitting
    itself is not identically zero. Together these say the model is a
    magnet with zero magnetization rather than either a ferromagnet or a
    paramagnet."""
    h = xwave.xwave_magnet(wave=wave,J=0.3,t=-1.)
    hk = h.get_hk_gen()
    ks = kmesh(2,nk=24)
    split = np.array([np.real(np.array(hk(k))[0,0]-np.array(hk(k))[1,1])
                      for k in ks])
    assert abs(np.mean(split)) < 1e-12 # compensated
    assert np.max(np.abs(split)) > 0.1*0.3 # but split


@pytest.mark.parametrize("wave",list(WAVES))
def test_hermitian_and_spin_diagonal(wave):
    """These models have no spin-orbit coupling by construction, which is
    the whole point (the nonlinear spin currents exist without it). The
    Bloch Hamiltonian must therefore stay Hermitian and exactly spin
    diagonal at every k -- including for p and f wave, whose odd form
    factors give *imaginary* spin-dependent hoppings, which is correct
    rather than a symptom of a broken construction."""
    h = xwave.xwave_magnet(wave=wave,J=0.3,t=-1.)
    hk = h.get_hk_gen()
    for k in np.random.RandomState(1).uniform(-0.5,0.5,size=(6,3)):
        m = np.array(hk([k[0],k[1],0.]))
        assert np.max(np.abs(m-m.conj().T)) < 1e-12
        assert abs(m[0,1]) < 1e-12 and abs(m[1,0]) < 1e-12


def test_parity_of_the_form_factors():
    """p and f wave are odd under k -> -k (odd-parity magnets), while d, g
    and i wave are even (altermagnets proper). Getting this backwards would
    silently change which nonlinear orders are allowed."""
    for (wave,parity) in [("p",-1),("d",1),("f",-1),("g",1),("i",1)]:
        h = xwave.xwave_magnet(wave=wave,J=1.,t=-1.)
        for k in np.random.RandomState(2).uniform(-0.6,0.6,size=(4,2)):
            assert np.isclose(_form_factor(h,-k),parity*_form_factor(h,k),
                    atol=1e-12)


def test_public_builders_match_the_generic_one():
    """The named aliases exported from specialhamiltonian are the same
    models as xwave_magnet(wave=...)"""
    pairs = [(specialhamiltonian.dwave_altermagnet,"d"),
             (specialhamiltonian.fwave_magnet,"f"),
             (specialhamiltonian.gwave_altermagnet,"g"),
             (specialhamiltonian.iwave_altermagnet,"i")]
    for (builder,wave) in pairs:
        h1 = builder(J=0.2)
        h2 = xwave.xwave_magnet(wave=wave,J=0.2)
        k = [0.13,0.29,0.]
        assert np.allclose(np.array(h1.get_hk_gen()(k)),
                           np.array(h2.get_hk_gen()(k)))


@pytest.mark.parametrize("wave",list(WAVES))
def test_harmonic_content_of_the_splitting(wave):
    """The spin splitting around a loop must be a pure X-wave harmonic: the
    intended l must dominate and every lower one be absent.

    This is a sharper statement than the node count above, and a more robust
    way to measure it -- a Fourier transform of Delta(phi) needs no
    sign-change bookkeeping and no crossing tracking. (For a multiorbital
    mean field, node counting at fixed sorted band index is actively
    unreliable: Delta_n(phi) jumps at every level crossing and manufactures
    spurious sign changes. These models are one orbital per spin, so both
    work here, but the harmonic test is the one that ports.)

    The circle is taken in Cartesian coordinates deliberately: a circle in
    reduced coordinates is an ellipse on a hexagonal lattice, and the path
    alone would then mix harmonics."""
    harmonic,nlines = WAVES[wave]
    h = xwave.xwave_magnet(wave=wave,J=1.,t=-1.)
    n = 720
    phis = (np.arange(n)+0.5)*2.*np.pi/n
    r = 0.3
    d = np.array([_form_factor(h,[r*np.cos(p),r*np.sin(p)]) for p in phis])
    amp = np.abs(np.fft.rfft(d))/n
    assert np.argmax(amp) == nlines # the intended harmonic dominates
    lower = np.max(amp[1:nlines]) if nlines>1 else 0.
    assert lower < 1e-6*amp[nlines] # and nothing below it survives


def test_a_real_hamiltonian_carries_only_even_harmonics():
    """Reality of the Hamiltonian forces the spin splitting to be even in k,
    hence to carry only even harmonics.

    H_s(k)* = H_s(-k) for real hoppings with no spin-orbit coupling, so
    eps_s(k) = eps_s(-k) and Delta(phi+180 deg) = Delta(phi), which kills
    every odd l. The models bear this out and it explains their
    construction: p-wave (l=1) and f-wave (l=3) come out with *imaginary*
    hoppings, while d, g and i wave are real.

    The consequence is worth knowing before building a fixture: on a
    C3-symmetric cell, C3 kills every l not divisible by 3 and reality kills
    every odd l, so a real C3 model can only carry l = 6, 12, ... -- it is
    necessarily i-wave, and no vacancy pattern can make it f-wave."""
    for wave in WAVES:
        h = xwave.xwave_magnet(wave=wave,J=1.,t=-1.)
        hm = h.get_multicell()
        ms = [np.asarray(algebra.todense(hm.intra))]
        ms += [np.asarray(algebra.todense(t.m)) for t in hm.hopping]
        imag = max(np.max(np.abs(np.imag(m))) for m in ms)
        odd = WAVES[wave][1]%2==1
        if odd: assert imag > 1e-6, wave # p, f: must be complex
        else: assert imag < 1e-12, wave # d, g, i: real
