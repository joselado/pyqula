"""Inputs that used to be accepted in silence and return a plausible but
wrong answer, instead of being refused."""
import numpy as np
import pytest

from pyqula import geometry


def test_superconducting_guess_needs_the_nambu_degree_of_freedom():
    """mf="swave" on a Hamiltonian without the electron-hole degree of
    freedom produced an identically zero guess, so the SCF returned a
    normal-state Hamiltonian with zero gap and no warning."""
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError):
        h.get_mean_field_hamiltonian(U=-2.0, filling=0.5, mf="swave", nk=8)
    hb = geometry.chain().get_hamiltonian()
    hb.setup_nambu_spinor()
    h2 = hb.get_mean_field_hamiltonian(U=-2.0, filling=0.5, mf="swave", nk=8)
    assert h2 is not None and h2.get_gap() > 1e-3


def test_local_hubbard_u_needs_spin():
    """U is the up-down density-density interaction, so it has no meaning
    without spin. It used to be built and then quietly dropped, so U=50
    returned the bare Hamiltonian bit for bit."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        h.get_mean_field_hamiltonian(U=50., filling=0.5, nk=6, maxite=2)


def test_unknown_mean_field_guess_and_constrain_are_refused():
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError):
        h.get_mean_field_hamiltonian(U=2.0, filling=0.5, mf="swaev", nk=6)
    with pytest.raises(ValueError):
        h.get_mean_field_hamiltonian(U=2.0, filling=0.5, mf="ferro", nk=6,
                                     maxite=2, constrains=["no_magnetisms"])


def test_misspelled_keyword_does_not_run_with_the_default():
    """A keyword nobody consumes reached the end of the mean-field call
    chain and was dropped, so fillinng=0.9 silently ran at filling=0.5."""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    with pytest.raises(TypeError):
        h.get_mean_field_hamiltonian(V1=1.0, fillinng=0.9, nk=6, maxite=2)


@pytest.mark.parametrize("filling", [-0.3, 1.5, 2.0])
def test_filling_outside_the_unit_interval_is_refused(filling):
    """filling is the fraction of occupied states, so it lives in [0,1]. A
    negative one used to wrap around through negative indexing and give the
    Fermi energy of filling 1+f."""
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError):
        h.get_fermi4filling(filling, nk=8)


def test_negative_broadening_is_refused():
    """A negative delta enters the Lorentzian as d/(d^2+de^2) and returns a
    negative density of states."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError):
        h.get_dos(delta=-0.1, nk=4)
    with pytest.raises(ValueError):
        h.get_ldos(delta=-0.1, nk=4)


def test_geometry_get_hamiltonian_refuses_unknown_keywords():
    """A misspelled has_spin used to be dropped, handing back the default
    Hamiltonian as if the option had been applied."""
    g = geometry.honeycomb_lattice()
    for bad in ["has_spinn", "sparse", "ts"]:
        with pytest.raises(TypeError):
            g.get_hamiltonian(**{bad: True})


def test_ldos_accepts_the_energy_spelling():
    """`energy` is how the Green's function/transport routines spell it; in
    get_ldos it was swallowed, silently returning the LDOS at zero."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    a = h.get_ldos(e=3.0, nk=4)
    b = h.get_ldos(energy=3.0, nk=4)
    c = h.get_ldos(e=0.0, nk=4)
    assert np.allclose(a, b)
    assert not np.allclose(b, c)


def test_didv_accepts_the_temperature_spellings():
    """didv's temperature keyword is `temp`; `T`/`temperature` used to be
    swallowed and the zero-temperature result returned."""
    from pyqula import heterostructures
    h = geometry.chain().get_hamiltonian()
    ht = heterostructures.build(h.copy(), h.copy())
    ht.set_coupling(1.0)
    e = 1.95  # just inside the band edge, where temperature matters
    cold = ht.didv(energy=e)
    warm = ht.didv(energy=e, temp=0.2)
    assert abs(warm - cold) > 1e-2
    assert abs(ht.didv(energy=e, T=0.2) - warm) < 1e-8
    assert abs(ht.didv(energy=e, temperature=0.2) - warm) < 1e-8


def test_localprobe_didv_reads_T_as_the_transparency_and_temp_as_the_temperature():
    """The same three spellings on a LocalProbe, whose conventions differ
    from the Heterostructure's above: `temp`/`temperature` are the
    temperature, but `T` is the probe TRANSPARENCY -- the same knob as
    LocalProbe(...,T=...), set_coupling and get_kappa(T=...), and the one
    examples/transport/didv_kitaev/main.py sweeps through Hamiltonian.didv.
    Both readings used to be impossible: didv declared T and never used it,
    so the probe's own transparency was silently used instead."""
    from pyqula.transporttk.localprobe import LocalProbe
    h = geometry.chain().get_hamiltonian(has_spin=False)
    e = 1.9  # just inside the band edge, where temperature matters
    lp = LocalProbe(h, delta=1e-3)
    lp.T = 0.2
    cold = lp.didv(energy=e)
    warm = lp.didv(energy=e, temp=0.2)
    assert abs(warm - cold) > 1e-2
    assert abs(lp.didv(energy=e, temperature=0.2) - warm) < 1e-8
    # T is the transparency, so the keyword must return what the attribute
    # returns -- and must not be re-read as the temperature
    lp2 = LocalProbe(h, delta=1e-3)
    lp2.T = 0.9
    assert abs(lp.didv(energy=e, T=0.9) - lp2.didv(energy=e)) < 1e-12
    assert abs(lp.didv(energy=e, T=0.9) - cold) > 1e-2
    assert abs(lp.didv(energy=e, T=0.2) - cold) < 1e-12
    # didv_curve is the array-of-energies twin of didv, so T means the
    # same thing there
    assert abs(lp.didv_curve([e], T=0.9)[0] - lp2.didv(energy=e)) < 1e-12


def test_the_dead_legacy_selfconsistency_interface_is_refused():
    """`scftypes.selfconsistency` is now an alias of
    `densitydensity.Vinteraction`, whose signature shares none of the old
    interface's names. `selfconsistency(h, g=1.0, mode="U")` therefore ran
    with no interaction at all -- provably: it gave exactly the same answer
    as passing nothing. It must refuse and name the modern spelling."""
    from pyqula import scftypes
    g = geometry.honeycomb_lattice().supercell(2)
    g.dimensionality = 0
    h = g.get_hamiltonian()
    mf = scftypes.guess(h, mode="antiferro")
    for bad in [dict(g=1.0, mode="U"), dict(g=3.0), dict(vfun=lambda r: 0.)]:
        with pytest.raises(TypeError):
            scftypes.selfconsistency(h, filling=0.5, mix=0.9, mf=mf, **bad)
    # nkp is an exact rename of nk, so it is accepted rather than refused
    scf = scftypes.selfconsistency(h, filling=0.5, U=1.0, mix=0.9, mf=mf,
                                   nkp=6)
    assert np.mean(np.abs(scf.hamiltonian.get_magnetization()[:, 2])) > 0.


def test_fun_is_the_old_name_of_tij_and_is_honoured():
    """`fun` is what `tij` used to be called, and it was being dropped --
    so every caller spelling it that way (spinwaves, surface_TI,
    operators, several examples) silently got a plain first-neighbor
    Hamiltonian instead of the one its hopping function describes."""
    g = geometry.honeycomb_lattice()

    def f(r1, r2):
        return 0.37 if 0.9 < (r1-r2).dot(r1-r2) < 1.1 else 0.0

    with_tij = np.array(g.get_hamiltonian(has_spin=False, tij=f).intra)
    with_fun = np.array(g.get_hamiltonian(has_spin=False, fun=f).intra)
    default = np.array(g.get_hamiltonian(has_spin=False).intra)
    assert np.allclose(with_tij, with_fun)
    assert not np.allclose(with_fun, default)  # it is not the default build
    with pytest.raises(TypeError):
        g.get_hamiltonian(has_spin=False, tij=f, fun=f)
