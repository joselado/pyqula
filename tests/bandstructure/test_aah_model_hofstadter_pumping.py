import numpy as np

from pyqula import geometry


def _aah_spectrum(omega=0.0, phi=0.0, amplitude=1.0):
    """Sorted spectrum of a finite (0d) spinful Aubry-Andre-Harper bichain:
    a staggered exchange whose envelope is a cosine of frequency omega and
    phason phase phi."""
    g = geometry.bichain(30)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)

    def fm(r):
        return amplitude * (.5 + .5 * np.cos(2 * np.pi * (omega * r[0] + phi)))

    h.add_antiferromagnetism(fm)
    inds, es = h.get_bands()
    return np.sort(np.array(es))


def _gaps(e, minsize):
    """(integrated density of states, size) of every spectral gap wider than
    minsize."""
    d = np.diff(e)
    return [((i + 1) / len(e), d[i]) for i in np.where(d > minsize)[0]]


def test_aah_spectrum_obeys_the_hofstadter_duality_and_is_phason_periodic():
    """The AAH modulation cos(2*pi*(omega*x+phi)) is evaluated on sites at
    half-integer x, so replacing omega by 1-omega and phi by 1/2-phi leaves
    the on-site potential of every site unchanged: the Hofstadter butterfly
    must be exactly symmetric about omega=1/2. The spectrum is likewise
    exactly periodic in the phason phi. Both are properties of the cosine
    modulation itself, which the old sum-of-all-spectra reference could not
    see -- that sum is sum over the sweep of Tr H, and the staggered
    exchange cancels out of Tr H for every omega, phi and amplitude."""
    for (omega, phi) in [(0.3, 0.2), (0.1, 0.0)]:
        a = _aah_spectrum(omega=omega, phi=phi)
        b = _aah_spectrum(omega=1. - omega, phi=0.5 - phi)
        assert np.allclose(a, b, atol=1e-9)
    assert np.allclose(_aah_spectrum(omega=0.1, phi=0.),
                       _aah_spectrum(omega=0.1, phi=1.), atol=1e-9)


def test_aah_gaps_obey_gap_labelling_and_pump_a_state_across():
    """Two statements about the omega=0.1 AAH chain that the recorded
    spectrum sum could not make.

    Gap labelling: every wide gap of a quasiperiodic chain sits at an
    integrated density of states that is an integer multiple of the
    modulation frequency. At omega=0.1 the three gaps wider than 0.2 sit at
    IDS = 0.4, 0.5 and 0.6; at omega=0.37 they sit at 0.1333, 0.5 and
    0.8667, which are not multiples of 0.1.

    Thouless pumping: advancing the phason through one full period returns
    the spectrum to itself, but transports exactly one state across the gap
    on the way, so the occupation of the gap-labelled level dips by one and
    comes back."""
    omega, amplitude = 0.1, 1.0
    e0 = _aah_spectrum(omega=omega, amplitude=amplitude)

    # Gershgorin brackets the spectral radius between the largest on-site
    # exchange field and that field plus the two unit hoppings every site has
    x = geometry.bichain(30).r[:, 0]
    vmax = amplitude * np.max(.5 + .5 * np.cos(2 * np.pi * omega * x))
    assert vmax <= np.max(np.abs(e0)) <= vmax + 2.

    wide = _gaps(e0, 0.2)
    assert len(wide) >= 3
    for (ids, size) in wide:
        assert np.isclose(ids / omega, round(ids / omega), atol=1e-6)

    # pump: count the states below the IDS=0.4 gap through one phason cycle
    i = int(round(0.4 * len(e0))) - 1
    egap = 0.5 * (e0[i] + e0[i + 1])
    counts = [int(np.sum(_aah_spectrum(omega=omega, phi=p) < egap))
              for p in np.linspace(0., 1., 11)]
    assert counts[0] == counts[-1]  # the cycle closes
    assert min(counts) == counts[0] - 1  # and carries one state across
