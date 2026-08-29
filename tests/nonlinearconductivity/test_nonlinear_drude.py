"""Physical invariants of the l-th order nonlinear Drude conductivity
(src/pyqula/conductivity.py, implemented in conductivitytk/nonlineardrude.py).

The headline test is the X-wave selection rule of Ezawa, arXiv:2411.16036
(Phys. Rev. B 111, 125420 (2025)): a collinear magnet whose spin-splitting
form factor is a harmonic of order l+1 generates a nonlinear spin current
only from order l upwards, so the lowest nonvanishing order reads off the
wave index -- p:0, d:1, f:2, g:3, i:5. That is the measurement of
altermagnetic order this module exists for, and it needs no spin-orbit
coupling.

The absolute normalization is pinned separately against the paper's closed
form sigma_spin^{yyyyy;x} = 360 (e/hbar)^6 V^F J/(i w + 1/tau)^5 for the
i-wave altermagnet, and the k-derivative machinery against pyqula's own
current.hk_derivative.
"""
import itertools
import warnings

import numpy as np
import pytest

from pyqula import conductivity
from pyqula import current
from pyqula import geometry
from pyqula import specialhamiltonian
from pyqula.conductivitytk import nonlineardrude
from pyqula.multihopping import MultiHopping
from pyqula import multicell
from pyqula.specialhamiltoniantk import xwave


# lowest order at which each X-wave generates a nonlinear spin current
THRESHOLD = {"p":0,"d":1,"f":2,"g":3,"i":5}

# The triangular models use t = -1, whose band runs over [-6,3]; the square
# ones over [-4,4]. A chemical potential a little above the band bottom puts
# the Fermi surface in the small, closed pocket around Gamma the analytic
# results describe.
BOTTOM = {"p":-4.,"d":-4.,"f":-6.,"g":-4.,"i":-6.}


def _model(wave,J=0.3):
    return xwave.xwave_magnet(wave=wave,J=J,t=-1.)


@pytest.mark.parametrize("wave",list(THRESHOLD))
def test_selection_rule(wave):
    """Every order below the threshold must vanish identically, and the
    threshold order must not.

    This is the whole deliverable in one assertion. The "vanish" side is a
    genuine zero, not a small number: the lower-order integrands are odd
    over the Brillouin zone and cancel to machine precision on any
    symmetric mesh, so the tolerance can be set far below the size of the
    surviving response rather than at some fraction of it."""
    l0 = THRESHOLD[wave]
    h = _model(wave)
    mu = BOTTOM[wave]+0.5
    orders = conductivity.nonlinear_drude_orders(h,lmax=l0+1,nk=48,T=0.02,
            mu=mu)
    for l in range(l0):
        assert orders[l] < 1e-12, ("order %d should vanish for %s-wave"
                % (l,wave))
    # the surviving response varies a lot in size between the waves (the
    # p-wave l=0 persistent spin current is the smallest, ~3e-4 here), so
    # the "nonzero" side is asserted well above the 1e-12 zeros rather than
    # at any particular scale
    assert orders[l0] > 1e-5, ("order %d should be the threshold for "
            "%s-wave" % (l0,wave))


def test_iwave_is_the_one_that_needs_fifth_order():
    """The i-wave result stated on its own, because it is the case that
    motivates the whole module: orders 0 through 4 are all zero, and the
    response appears at fifth order. Ezawa's second-order charge response
    (arXiv:2409.09241) cannot see this magnet at all."""
    h = _model("i")
    orders = conductivity.nonlinear_drude_orders(h,lmax=5,nk=48,T=0.02,
            mu=-5.5)
    assert np.max(orders[0:5]) < 1e-12
    assert orders[5] > 1e-2


@pytest.mark.parametrize("wave,l",[("d",1),("f",2),("g",3),("i",5)])
def test_charge_channel_vanishes_while_the_spin_channel_does_not(wave,l):
    """At the threshold order the *spin* response is nonzero but the
    *charge* response vanishes: the two spin channels contribute with
    opposite signs, which is what makes this a pure spin current.

    This is the cheap null test that catches a sign error in the
    spin/charge combination -- swapping the two would leave the
    selection-rule test above passing while inverting the physics."""
    h = _model(wave)
    mu = BOTTOM[wave]+0.5
    kw = dict(nk=48,T=0.02,mu=mu)
    spin = conductivity.nonlinear_drude_components(h,l,channel="spin",**kw)
    charge = conductivity.nonlinear_drude_components(h,l,channel="charge",**kw)
    key = max(spin,key=lambda k: abs(spin[k]))
    assert abs(spin[key]) > 1e-3
    assert abs(charge[key]) < 1e-12*max(1.,abs(spin[key]))


def test_iwave_component_ratios_match_the_paper():
    """Ezawa's Eq. (i2Ds) gives sigma^{yyyyy;x} = sigma^{xxxxx;y} =
    -sigma^{xxxyy;y} for the 2D i-wave altermagnet. Those relative signs
    are a property of the l=6 harmonic and hold on the lattice too, well
    away from the continuum limit, so they pin the index bookkeeping (which
    index is the current and which are the field powers) independently of
    any overall scale."""
    h = _model("i",J=0.05)
    c = conductivity.nonlinear_drude_components(h,5,nk=200,T=0.01,mu=-5.75)
    ref = c["yyyyy;x"]
    assert abs(ref) > 1e-6
    assert np.isclose(np.real(c["xxxxx;y"]/ref),1.,atol=1e-6)
    assert np.isclose(np.real(c["xxxyy;y"]/ref),-1.,atol=1e-6)


def test_iwave_absolute_scale_extrapolates_to_the_analytic_360():
    """Ezawa's closed form for the i-wave altermagnet is

      sigma_spin^{yyyyy;x} = 360 (e/hbar)^6 V^F J/(i omega + 1/tau)^5,

    derived from the continuum model, where d^6 eps_s/dk_y^5 dk_x = 360 s J
    is a constant. On the lattice that constant picks up corrections of
    order mu, so the ratio approaches 360 only as the Fermi pocket shrinks
    onto Gamma. Two chemical potentials and a linear (Richardson)
    extrapolation in mu recover it to better than a percent, which pins the
    absolute normalization -- every prefactor, the (2 pi)^D convention and
    the cell volume together.

    A plain assertion at one mu would either be loose enough to pass with a
    wrong prefactor or would just be fitting the lattice correction."""
    J = 0.02
    h = _model("i",J=J)
    ratios = []
    ds = [0.05,0.03]
    for (d,nk,T) in [(ds[0],800,0.0025),(ds[1],900,0.0015)]:
        mu = -6.+d
        s = conductivity.nonlinear_drude_conductivity(h,field="yyyyy",
                current="x",channel="spin",nk=nk,T=T,mu=mu,tau=1.)
        vf = conductivity.fermi_volume(h,nk=nk,T=T,mu=mu,channel="charge")/2.
        ratios.append(np.real(s)/(J*vf))
    # linear extrapolation of the two to mu -> band bottom
    r0 = ratios[1]+(ratios[1]-ratios[0])*(0.-ds[1])/(ds[1]-ds[0])
    assert np.isclose(r0,360.,rtol=0.02)


@pytest.mark.parametrize("wave,l",[("d",1),("f",2),("g",3)])
def test_tau_and_omega_scaling(wave,l):
    """The l-th order Drude response carries the factor
    1/(i omega + 1/tau)^l and nothing else that depends on tau or omega, so
    it scales as tau^l in the static limit and follows the complex
    Lorentzian at finite frequency. A response assigned the wrong order --
    off by one in the count of field powers -- would fail this even though
    it might still pass the selection-rule test."""
    h = _model(wave)
    kw = dict(field="x"*l,current="y",nk=40,T=0.05,mu=BOTTOM[wave]+1.0)
    s1 = conductivity.nonlinear_drude_conductivity(h,tau=1.,**kw)
    s2 = conductivity.nonlinear_drude_conductivity(h,tau=2.,**kw)
    assert np.isclose(abs(s2/s1),2.**l,rtol=1e-10)
    w,tau = 0.7,1.3
    sw = conductivity.nonlinear_drude_conductivity(h,tau=tau,omega=w,**kw)
    s0 = conductivity.nonlinear_drude_conductivity(h,tau=tau,omega=0.,**kw)
    assert np.isclose(sw/s0,(1./tau)**l/(1j*w+1./tau)**l,rtol=1e-10)


def test_cartesian_derivative_matches_hk_derivative():
    """The module differentiates the Bloch series analytically, multiplying
    each term by i R_a per Cartesian derivative. pyqula's own shared
    k-derivative, current.hk_derivative, instead differentiates with respect
    to the *reduced* momentum, so the two agree only through the
    reduced-to-Cartesian chain rule with the jacobian a_i[a]/(2 pi).

    Checking them against each other at sixth order, on the triangular
    lattice whose lattice vectors are not orthogonal, is the one place a
    factor of 2 pi or a transposed jacobian would hide -- and it is exactly
    the derivative the i-wave selection rule depends on."""
    h = _model("i")
    hs = h.copy(); hs.remove_spin(channel="up")
    hm = hs.get_multicell().copy()
    chan = nonlineardrude._Channel(hm,[[0.13,0.29,0.]])
    g = hm.geometry
    jac = np.array([g.a1,g.a2])/(2.*np.pi) # dk_i/dK_a
    for axes in [[1,1,1,1,1,0],[0,0,0,1,1,1],[0,1],[1,1,1,0]]:
        # the same derivative via current.hk_derivative and the chain rule
        want = 0.
        for idxs in itertools.product(range(2),repeat=len(axes)):
            c = np.prod([jac[i,a] for (i,a) in zip(idxs,axes)])
            if c==0.: continue
            order = [list(idxs).count(0),list(idxs).count(1)]
            m = current.hk_derivative(hm,[0.13,0.29,0.],order=order)
            want = want+c*np.array(m)[0,0]
        got = chan.derivative(axes)[0,0]
        assert np.isclose(got,np.real(want),rtol=1e-9,atol=1e-12)


def test_components_agree_with_individual_calls():
    """nonlinear_drude_components shares one k-mesh and one set of Bloch
    phases across every component; that shortcut must not change any
    number."""
    h = _model("g")
    kw = dict(nk=36,T=0.05,mu=-3.0)
    c = conductivity.nonlinear_drude_components(h,3,**kw)
    for (key,val) in c.items():
        field,b = key.split(";")
        one = conductivity.nonlinear_drude_conductivity(h,field=field,
                current=b,**kw)
        assert np.isclose(val,one,rtol=1e-12,atol=1e-14)


def test_supercell_reproduces_the_primitive_cell():
    """A supercell describes the same physics, so it must give the same
    conductivity. This is what pins the k-mesh and cell-volume
    normalization: a one-site cell cannot catch an error there, since every
    wrong convention is a constant that a single model cannot separate from
    the answer.

    It also exercises the degenerate branch hard: a folded band structure is
    nothing but degeneracies -- every pair of bands that came from the same
    primitive band touches somewhere -- so a per-band expansion would divide
    by zero all over the mesh. Getting the primitive answer back is the
    evidence that the block (Kato) treatment of degenerate multiplets is
    right.

    The comparison has to be made on equivalent k-meshes. A supercell(2)
    evaluated at nk samples the *primitive* Brillouin zone at 2*nk per
    direction, so the primitive calculation is run at twice the nk; against
    a same-nk primitive run the two would differ by ordinary mesh
    discretization error and the test would be measuring that instead."""
    h = specialhamiltonian.dwave_altermagnet(J=0.3)
    kw = dict(field="x",current="y",T=0.05,mu=-1.0,channel="spin")
    prim = conductivity.nonlinear_drude_conductivity(h,nk=40,**kw)
    sup = conductivity.nonlinear_drude_conductivity(h.supercell(2),nk=20,**kw)
    assert abs(prim) > 1e-3
    assert np.isclose(np.real(sup),np.real(prim),rtol=1e-12)


def test_a_nonmagnetic_lattice_gives_no_spin_current():
    """A plain square lattice with no spin splitting has identical spin
    channels, so every order of the spin response vanishes while the charge
    response does not.

    It must also trip the spin-degeneracy guard: asking for a spin response
    from a system with no spin splitting at all is exactly what the guard
    exists to flag."""
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    h.shift_fermi(-1.0)
    with pytest.warns(RuntimeWarning,match="degenerate to within rounding"):
        orders = conductivity.nonlinear_drude_orders(h,lmax=3,nk=36,T=0.05)
    assert np.max(orders) < 1e-12
    charge = conductivity.nonlinear_drude_conductivity(h,field="x",
            current="x",channel="charge",nk=36,T=0.05)
    assert abs(charge) > 1e-3


def test_spin_orbit_coupling_is_rejected():
    """With spin-orbit coupling the spin channels mix, the single-band
    Boltzmann derivation behind this formula no longer applies, and the
    quantum-metric and Berry-curvature-dipole channels of arXiv:2409.09241
    contribute as well. Silently dropping the spin off-diagonal blocks --
    which is what remove_spin would do -- would give a plausible-looking
    wrong number, so this must raise instead."""
    h = specialhamiltonian.dwave_altermagnet(J=0.3)
    h.add_rashba(0.2)
    with pytest.raises(ValueError):
        conductivity.nonlinear_drude_conductivity(h,field="x",current="y",
                nk=12,T=0.05)


def _direct_sum(hA,hB):
    """Two one-orbital Hamiltonians on the same lattice, combined into a
    single two-orbital Hamiltonian with block-diagonal hoppings.

    Band energies depend only on the hopping dictionary and the lattice
    vectors, not on where the sites sit, so giving the second orbital an
    arbitrary position is harmless. The point is to build a multiorbital
    Hamiltonian whose exact answer is known independently -- it is the sum
    of the two one-orbital answers -- which is what makes it a test of the
    multiband machinery rather than of itself."""
    g = hA.geometry.copy()
    g.r = np.array([[0.,0.,0.],[0.3,0.2,0.]])
    g.x,g.y,g.z = g.r[:,0],g.r[:,1],g.r[:,2]
    g.get_fractional()
    h = g.get_hamiltonian(has_spin=True,is_multicell=True,
            tij=lambda r1,r2: 0.0).get_multicell()
    dA = multicell.get_hopping_dict(hA.get_multicell())
    dB = multicell.get_hopping_dict(hB.get_multicell())
    out = dict()
    for key in set(dA)|set(dB):
        m = np.zeros((4,4),dtype=np.complex128) # (site,spin), site slowest
        m[0:2,0:2] = np.array(dA.get(key,np.zeros((2,2))),dtype=np.complex128)
        m[2:4,2:4] = np.array(dB.get(key,np.zeros((2,2))),dtype=np.complex128)
        out[key] = m
    h.set_multihopping(MultiHopping(out))
    return h


@pytest.mark.parametrize("l",[1,2,3,4,5])
def test_multiband_reproduces_the_exact_one_orbital_result(l):
    """A two-orbital Hamiltonian built as the direct sum of two decoupled
    one-orbital models, with the second pushed far above the Fermi level, is
    physically the first model alone -- but it is computed through the
    multiband path, where the band energies are eigenvalues and their
    derivatives come from perturbation theory rather than from the Bloch
    series directly.

    Both the value at the threshold order and, just as importantly, the
    zeros below it must survive. That second half is what a
    finite-difference scheme cannot do: it manufactures a spurious nonzero
    value where the selection rule demands an exact zero, which would look
    exactly like the i-wave response appearing at the wrong order."""
    hA = xwave.xwave_magnet(wave="i",J=0.3,t=-1.)
    hB = xwave.xwave_magnet(wave="i",J=0.15,t=-0.6)
    hB.intra = np.array(hB.intra)+np.eye(2)*12. # far above, never occupied
    h2 = _direct_sum(hA,hB)
    kw = dict(field="y"*l,current="x",nk=30,T=0.05,mu=-5.0)
    one = conductivity.nonlinear_drude_conductivity(hA,**kw)
    two = conductivity.nonlinear_drude_conductivity(h2,**kw)
    if l==5: # the threshold order: same number
        assert abs(one) > 1e-2
        assert np.isclose(np.real(two),np.real(one),rtol=1e-8)
    else: # below threshold: both exactly zero
        assert abs(one) < 1e-12 and abs(two) < 1e-12


def test_degenerate_bands_are_handled():
    """A C3-symmetric cell has two-dimensional irreducible representations,
    and hence exactly degenerate bands, at its high-symmetry points -- which
    is the generic situation for the superlattices these responses are
    interesting for, not a corner case. The block treatment must get through
    it and return a finite answer rather than dividing by a zero band
    spacing.

    Asked for the CHARGE channel deliberately. This fixture is a compensated
    Neel state, which is spin degenerate, so its spin response is rounding
    and is (correctly) flagged as such by the guard -- see
    test_spin_degenerate_state_is_flagged. The charge response is a genuine
    quantity here, and it exercises the same degenerate machinery."""
    g = geometry.honeycomb_lattice().get_supercell(3)
    g.supercell_replica = None
    g.supercell_primal_index = None
    g.supercell_matrix = None
    c = np.mean(g.r,axis=0)
    g = g.remove(lambda r: np.sqrt(((r-c)**2).sum())<0.9) # carve an antidot
    h = g.get_hamiltonian(has_spin=True)
    h.add_antiferromagnetism(0.35)
    with warnings.catch_warnings():
        warnings.simplefilter("error",RuntimeWarning) # must not warn
        v = conductivity.nonlinear_drude_conductivity(h,field="yyy",
                current="x",channel="charge",nk=12,T=0.05,mu=0.45)
    assert np.isfinite(np.real(v))


def test_a_gapped_insulator_gives_exactly_zero_at_every_order():
    """With the chemical potential in a gap the response vanishes
    identically, at every order and for every wave symmetry.

    f is 1 on every valence band and 0 on every conduction band, so the
    integrand is a pure k-derivative of Tr(P H) with P the valence
    projector; that is smooth and periodic for a gapped system, and the
    integral of a derivative of a smooth periodic function over a full
    period is zero. Worth pinning as a test because it is a trap: an
    insulator reproduces the "no response below fifth order" pattern
    trivially, so an absence of low orders only identifies i-wave order when
    the fifth order is simultaneously present, which needs a metal."""
    # a gapped model: two orbitals pushed far apart, the lower one filled
    h = xwave.xwave_magnet(wave="i",J=0.3,t=-1.)   # band over [-6,3]
    hB = xwave.xwave_magnet(wave="i",J=0.15,t=-0.6)
    hB.intra = np.array(hB.intra)+np.eye(2)*30. # upper band over [26.4,31.8]
    h2 = _direct_sum(h,hB)
    orders = conductivity.nonlinear_drude_orders(h2,lmax=5,nk=24,T=0.02,mu=8.)
    assert np.max(orders) < 1e-10


@pytest.mark.parametrize("l",[1,2,3,4,5])
def test_degenerate_bands_at_high_order(l):
    """Degenerate bands AND a sixth-order derivative at the same time --
    which is the case this module actually gets used for, and the one where
    each ingredient alone proves nothing.

    A supercell(2) of the i-wave model has folded bands that touch along
    whole curves, so the block (Kato) treatment is exercised everywhere, and
    the i-wave selection rule needs the expansion carried to sixth order. If
    the trace of the block effective Hamiltonian were mishandled at any
    order of the recursion, either the four zeros below the threshold would
    stop being zero or the fifth-order value would drift.

    Meshes are matched: a supercell(2) at nk samples the primitive zone at
    2*nk per direction."""
    h = xwave.xwave_magnet(wave="i",J=0.3,t=-1.)
    kw = dict(field="y"*l,current="x",T=0.05,mu=-5.0,channel="spin")
    prim = conductivity.nonlinear_drude_conductivity(h,nk=24,**kw)
    sup = conductivity.nonlinear_drude_conductivity(h.supercell(2),nk=12,**kw)
    if l==5: # the threshold order survives folding, to machine precision
        assert abs(prim) > 1e-2
        assert np.isclose(np.real(sup),np.real(prim),rtol=1e-10)
    else: # and the zeros below it stay zero in both
        assert abs(prim) < 1e-12 and abs(sup) < 1e-12


def test_band_derivatives_match_jax_autodiff():
    """The multiorbital band derivatives, checked against an independent
    tool rather than against another part of this module.

    conductivitytk/nonlineardrude.py derives the high-order derivatives of
    the band energies by hand, through a Rayleigh-Schroedinger recursion in
    a truncated Taylor ring. jax (already a hard dependency) can get the
    same numbers by nesting jacfwd six times over jnp.linalg.eigvalsh, along
    a completely different route: automatic differentiation of the
    eigendecomposition itself. Agreement at sixth order on a genuine
    two-orbital Hamiltonian is strong evidence the recursion is right, in a
    way that no internal consistency check could be.

    Deliberately run on a gapped honeycomb. jax's eigendecomposition
    derivative divides by eigenvalue differences and returns NaN at an exact
    degeneracy -- which is precisely why autodiff is not used in the module
    itself: the C3 superlattices this is for have exactly degenerate bands
    at their high-symmetry points, and only the block (Kato) treatment
    survives there."""
    import jax
    jax.config.update("jax_enable_x64",True)
    import jax.numpy as jnp

    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.8) # lift the Dirac point: jax needs a gap
    hs = h.copy(); hs.remove_spin(channel="up")
    hm = hs.get_multicell().copy()

    k = [0.13,0.29,0.]
    chan = nonlineardrude._Channel(hm,[k])
    dirs,rcart,ms = nonlineardrude._bloch_series(hm)

    # the same Bloch series, but differentiated by jax with respect to the
    # Cartesian momentum
    Rc = jnp.array(rcart[:,0:2]); M = jnp.array(ms)
    def band(kc,n):
        ph = jnp.exp(1j*(Rc@kc))
        A = jnp.einsum("h,hij->ij",ph,M)
        return jnp.linalg.eigvalsh((A+jnp.conjugate(A.T))/2.)[n]
    bvec = 2.*np.pi*np.linalg.inv(np.array([g.a1[0:2],g.a2[0:2]]).T)
    kc = jnp.array(np.array([k[0],k[1]])@bvec)

    for axes in [[0,1],[1,1,1,0],[1,1,1,1,1,0],[0,0,0,1,1,1]]:
        mine = chan.derivative(axes)[0]
        theirs = []
        for n in range(chan.nbands):
            f = lambda kk: band(kk,n)
            for i in range(len(axes)): f = jax.jacfwd(f)
            theirs.append(float(np.array(f(kc))[tuple(axes)]))
        assert np.allclose(mine,np.array(theirs),rtol=1e-9,atol=1e-9), axes


def test_spin_degenerate_state_is_flagged():
    """A state whose two spin channels are the same is not an altermagnet,
    and any "spin response" it returns is rounding.

    A compensated Neel state on a bipartite lattice is PT symmetric and
    exactly spin degenerate -- an antiferromagnet. Its spin response is not
    caught by inspecting the answer: the two channels are diagonalized
    independently, so near a degeneracy their eigenvector gauges differ and
    the high-order derivatives drift apart far more than the band energies
    do (measured: bands agreeing to 1.2e-14 gave sigma_up and sigma_dn a
    factor of 8 apart). The diagnostic has to be the band splitting itself.

    Silently returning such a number would be the worst failure this module
    could have -- it looks exactly like the i-wave signal it is supposed to
    measure."""
    g = geometry.honeycomb_lattice().get_supercell(3)
    g.supercell_replica = None
    g.supercell_primal_index = None
    g.supercell_matrix = None
    c = np.mean(g.r,axis=0)
    g = g.remove(lambda r: np.sqrt(((r-c)**2).sum())<0.9)
    h = g.get_hamiltonian(has_spin=True)
    h.add_antiferromagnetism(0.35) # Neel: compensated but spin degenerate
    with pytest.warns(RuntimeWarning,match="degenerate to within rounding"):
        conductivity.nonlinear_drude_conductivity(h,field="yyy",current="x",
                nk=8,T=0.05,mu=0.45)


def test_a_real_altermagnet_is_not_flagged():
    """The converse: a genuine altermagnet must not trip the guard, either
    in its primitive cell or in a supercell whose folded bands are
    degenerate. Band degeneracy within a channel is not spin degeneracy
    between channels, and the guard must not confuse the two."""
    h = xwave.xwave_magnet(wave="i",J=0.3,t=-1.)
    for hh,nk in [(h,24),(h.supercell(2),12)]:
        with warnings.catch_warnings():
            warnings.simplefilter("error",RuntimeWarning)
            v = conductivity.nonlinear_drude_conductivity(hh,field="yyyyy",
                    current="x",nk=nk,T=0.05,mu=-5.)
        assert abs(v) > 1e-2
