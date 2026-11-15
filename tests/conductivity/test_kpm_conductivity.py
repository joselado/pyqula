"""The Chebyshev (KPM) Kubo conductivity of conductivitytk/kpmkubo.py
against the band calculation of conductivity.py.

The KPM path builds the same velocity operator sparsely and expands the
spectral density of two currents in Chebyshev polynomials, so on a small
cell it must reproduce the Lehmann sum: exactly for the moments, to the
kernel line shape for the spectrum, and to the stochastic error bar for
random vectors. The Hall part is pinned against the Chern number.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import conductivity
from pyqula import topology
from pyqula.conductivitytk import kpmkubo
from pyqula.conductivitytk import kubo


def _disordered_cell(seed=1):
    """Gapped honeycomb (sublattice imbalance 0.5, gap 1) in a 3x3
    supercell, 18 orbitals, with random onsite disorder of width 0.3, below
    the gap"""
    rng = np.random.default_rng(seed)
    g = geometry.honeycomb_lattice().get_supercell(3)
    h = g.get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.5)
    h.intra = h.intra + np.diag(0.3*(rng.random(len(g.r))-0.5))
    return h


def _haldane(t2):
    """Spinless Haldane model with the Fermi level in the gap, as in
    test_optical_conductivity.py"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_haldane(t2)
    h.shift_fermi(0.3)
    return h


def test_chebyshev_moments_match_the_diagonalization():
    """The double moments from the Chebyshev recursion with the exact
    trace equal Tr[v_a T_n(H) v_b T_m(H)] built from the eigenstates"""
    h = _disordered_cell()
    st = kpmkubo._Setup(h,nk=2,npol=60)
    mu = next(kpmkubo.moments(st,trace="exact"))
    ref = kpmkubo.exact_moments(st)
    assert np.max(np.abs(mu-ref)) < 1e-10*np.max(np.abs(ref))
    # a tiny memory budget stores a few polynomials at a time and runs the
    # left set again for every block, with the same result
    assert kpmkubo._block_size(18,18,2,st.M,1e-4) < st.M
    mu = next(kpmkubo.moments(st,trace="exact",memory=1e-4))
    assert np.max(np.abs(mu-ref)) < 1e-10*np.max(np.abs(ref))


def test_kpm_absorption_matches_the_band_calculation():
    """With the Lorentz kernel a transition gives a Lorentzian line of half
    width delta, the line shape of integration="grid". The two then agree
    to the kernel's deviation from a pure Lorentzian, a few percent of the
    peak at delta=0.05 on the same k-mesh"""
    h = _disordered_cell()
    ws = np.linspace(0.,4.,21)
    ws,s0 = h.get_optical_conductivity(energies=ws,nk=2,T=0.05,delta=0.05)
    ws,s1 = h.get_optical_conductivity(energies=ws,nk=2,T=0.05,delta=0.05,
            integration="kpm",kernel="lorentz",trace="exact",reactive=False)
    for a in range(2):
        ref = s0[:,a,a].real
        assert np.max(np.abs(s1[:,a,a].real-ref)) < 0.05*np.max(ref)
        assert np.max(np.abs(s1[:,a,a].imag)) < 1e-10*np.max(ref)


def test_kpm_circular_dichroism_matches_the_band_calculation():
    """The off-diagonal absorptive tensor, the Hermitian part
    (sigma_xy + conj(sigma_yx))/2, is i Im sigma^A_xy, the circular
    dichroism, which a time-reversal-symmetric cell does not have. On the
    Haldane model it is as large as sigma_xx, and the KPM value follows
    the band one to the kernel line shape. Its transpose has the opposite
    sign, so an exchange of the two Cartesian indices fails here"""
    h = _haldane(0.2)
    ws = np.linspace(0.,4.,21)
    kw = dict(energies=ws,nk=12,T=0.05,delta=0.1)
    ws,s0 = h.get_optical_conductivity(**kw)
    ws,s1 = h.get_optical_conductivity(integration="kpm",kernel="lorentz",
            trace="exact",reactive=False,**kw)
    ref = (s0[:,0,1]+np.conj(s0[:,1,0]))/2.
    scale = np.max(np.abs(ref))
    assert scale > 0.5*np.max(np.abs(s0[:,0,0].real)) # a real signal
    assert np.max(np.abs(ref.real)) < 1e-10*scale # i Im, by C3
    assert np.max(np.abs(s1[:,0,1]-ref)) < 0.06*scale
    assert np.max(np.abs(s1[:,0,1]-np.conj(ref))) > scale # not transposed
    assert np.max(np.abs(s1[:,1,0]-np.conj(s1[:,0,1]))) < 1e-10*scale


def test_kramers_kronig_weights_transform_a_lorentzian():
    """The principal value weights of the reactive part are exact for a
    function linear between the nodes; on a Lorentzian line of half width
    d sampled every d/8 the transform is the dispersive Lorentzian to a few
    parts in a thousand of its peak, frequencies on a node included"""
    d,w0,h = 0.05,0.7,0.05/8.
    xs = h*np.arange(-int(8./h),int(8./h)+1)
    fs = d/((xs-w0)**2+d**2)/np.pi
    ws = np.concatenate([np.linspace(-1.,3.,401),xs[1500:1510]])
    out = kpmkubo._kk_weights(xs,ws,h)@fs
    ref = (ws-w0)/((ws-w0)**2+d**2)
    assert np.max(np.abs(out-ref)) < 5e-3*np.max(np.abs(ref))


@pytest.mark.parametrize("shift",[0.,1.2])
def test_kpm_full_conductivity_matches_the_band_calculation(shift):
    """The reactive part is the Kramers-Kronig transform of the absorptive
    one, so the full complex sigma_xx(w), real and imaginary parts, follows
    the band calculation to the same few percent as the absorption (3% of
    the peak, against 2.6% for the absorptive part alone, in the gap), and
    its Hermitian part is the absorptive part exactly. With the Fermi level
    in a band the Drude peak dominates, and its reactive tail is held to 4%"""
    h = _disordered_cell()
    h.shift_fermi(shift)
    ws = np.linspace(0.,4.,21)
    kw = dict(energies=ws,nk=2,T=0.05,delta=0.05)
    ws,s0 = h.get_optical_conductivity(**kw)
    ws,s1 = h.get_optical_conductivity(integration="kpm",kernel="lorentz",
            trace="exact",**kw)
    ws,sa = h.get_optical_conductivity(integration="kpm",kernel="lorentz",
            trace="exact",reactive=False,**kw)
    for a in range(2):
        ref = s0[:,a,a]
        scale = np.max(np.abs(ref))
        assert np.max(np.abs(ref.imag)) > 0.3*scale # a real signal
        assert np.max(np.abs(s1[:,a,a]-ref)) < 0.05*scale
    herm = (s1+np.conj(np.transpose(s1,(0,2,1))))/2.
    assert np.max(np.abs(herm-sa)) < 1e-10*np.max(np.abs(sa))


@pytest.mark.parametrize("t2",[0.2,-0.2])
def test_kpm_hall_conductivity_at_zero_frequency_is_the_chern_number(t2,
        tmp_path,monkeypatch):
    """At w -> 0 the reactive part of sigma_xy is the DC Hall conductivity,
    the Kramers-Kronig integral of the circular dichroism over all
    frequencies, so 2 pi Re sigma_xy(0) = -C in the gap. This route never
    goes through the Kubo-Bastin formula, so it is an independent check of
    both"""
    monkeypatch.chdir(tmp_path) # topology.chern writes *.OUT files to cwd
    h = _haldane(t2)
    c = round(topology.chern(h,nk=20))
    assert c != 0
    ws,s = h.get_optical_conductivity(energies=[0.,0.02],nk=12,T=0.02,
            delta=0.1,integration="kpm",trace="exact")
    assert np.allclose(2.*np.pi*s[:,0,1].real,-c,atol=0.02)
    assert np.allclose(s[:,1,0].real,-s[:,0,1].real,atol=1e-3)


def test_singleshot_matches_the_lorentzian_kubo_greenwood_formula():
    """mode="singleshot" is pi sum_nm |v_nm|^2 L(E-E_n) L(E-E_m)/V with L a
    Lorentzian of half width delta/2: with the exact trace the Chebyshev
    series of Im G gives it to its truncation, at every Fermi level, for
    both directions of a cell without any symmetry between them"""
    rng = np.random.default_rng(1)
    g = geometry.honeycomb_lattice().get_supercell(3)
    h = g.get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.2)
    h.intra = h.intra + np.diag(0.6*(rng.random(len(g.r))-0.5))
    fermis = np.array([-2.,-1.2,0.9,1.7])
    delta = 0.1
    s = h.get_dc_conductivity(fermi=fermis,nk=2,delta=delta,
            mode="singleshot",trace="exact")
    st = kpmkubo._Bounds(h,nk=2)
    eta = delta/2.
    L = lambda x: (eta/np.pi)/(x**2+eta**2)
    ref = np.zeros((len(fermis),3))
    for k in st.ks:
        es,U = np.linalg.eigh(st.ops.hamiltonian(k).toarray())
        for al in st.dirs:
            v = U.conj().T@st.ops.velocity(k,al).toarray()@U
            for j,E in enumerate(fermis):
                ref[j,al] += np.pi*np.sum(np.abs(v)**2*
                        np.outer(L(E-es),L(E-es)))
    ref /= st.V
    for al in st.dirs:
        assert np.max(np.abs(s[:,al,al]-ref[:,al])) < 1e-7*np.max(ref)
    assert np.all(np.isnan(s[:,0,1])) # not computed
    assert np.all(s[:,2,2]==0.) # no current along z
    # the random vectors are unbiased
    se,ee = h.get_dc_conductivity(fermi=fermis,nk=2,delta=delta,
            mode="singleshot",ntries=16,seed=3,return_error=True)
    assert np.all(np.abs(se[:,0,0]-s[:,0,0]) < 4.*ee[:,0,0])


def test_singleshot_drude_conductivity_of_a_clean_chain():
    """A clean ring has the Drude weight D = |v_F|/pi = 2/pi at half
    filling, and with each delta function a Lorentzian of half width
    delta/2 the single-shot conductivity is D/delta, the Drude peak of the
    band modes at w = 0. On 400 sites the level spacing, 0.03, is far below
    the half width 0.1, so the ring is the infinite chain"""
    g = geometry.chain().get_supercell(400)
    h = g.get_hamiltonian(has_spin=False,is_sparse=True)
    delta = 0.2
    s = h.get_dc_conductivity(fermi=0.,delta=delta,mode="singleshot",
            component="xx",trace="exact") # per unit length of the ring
    D = 2./np.pi # 2 Fermi points, |v_F| = 2 t, divided by 2 pi
    assert abs(s-D/delta) < 0.01*D/delta


def test_random_vectors_agree_with_the_exact_trace():
    """The stochastic trace is unbiased: with fixed seeds the estimate lies
    within a few standard errors of the exact trace, and the error bar
    shrinks with the number of random vectors"""
    h = _disordered_cell()
    ws = np.linspace(0.5,2.5,9)
    # the reactive part is linear in the moments, so it is unbiased when
    # the absorptive one is, and it is left out to keep the test fast
    kw = dict(energies=ws,nk=1,T=0.05,delta=0.2,npol=40,reactive=False)
    ws,ex = kpmkubo.optical_conductivity(h,trace="exact",**kw)
    errs = []
    for ntries in [4,16]:
        ws,s,e = kpmkubo.optical_conductivity(h,ntries=ntries,seed=7,
                return_error=True,**kw)
        d = np.abs(s[:,0,0]-ex[:,0,0])
        assert np.all(d < 4.*e[:,0,0] + 1e-3*np.max(ex[:,0,0].real))
        errs.append(np.mean(e[:,0,0]))
    assert errs[1] < errs[0]


@pytest.mark.parametrize("t2",[0.2,-0.2])
def test_kubo_bastin_hall_conductivity_is_the_chern_number(t2,tmp_path,
        monkeypatch):
    """The Kubo-Bastin DC tensor holds the Fermi sea, so in the gap of a
    Chern insulator sigma_xy 2 pi = -C, the sign of conductivity.py, for
    both signs of the flux, while sigma_xx vanishes. A scan over Fermi
    levels inside the gap gives the same plateau"""
    monkeypatch.chdir(tmp_path) # topology.chern writes *.OUT files to cwd
    h = _haldane(t2)
    c = round(topology.chern(h,nk=20))
    assert c != 0
    s = h.get_dc_conductivity(fermi=[-0.2,0.,0.2],nk=12,T=0.01,npol=200,
            trace="exact")
    assert np.allclose(2.*np.pi*s[:,0,1],-c,atol=5e-3)
    assert np.allclose(s[:,1,0],-s[:,0,1],atol=1e-6)
    assert np.max(np.abs(s[:,0,0])) < 1e-4


def test_kpm_sum_rule_weight():
    """The diamagnetic weight Tr[f(H) d2H/dK2]/V from a single Chebyshev
    expansion equals kubo.sum_rule_weight on the same k-mesh"""
    h = _disordered_cell()
    W0 = kubo.sum_rule_weight(h,nk=2,T=0.05)
    W1 = kpmkubo.sum_rule_weight(h,nk=2,T=0.05,trace="exact")
    assert np.allclose(W0,W1,atol=1e-4*np.max(np.abs(W0)))


def test_f_sum_rule_of_a_finite_flake():
    """For a finite system the velocity is v = i[H,X] and the f-sum rule
    int Re sigma_xx dw = pi Tr[f(H) (-[[H,X],X])] holds exactly, at any
    size, so the integral of the KPM absorption over all frequencies must
    give pi times the weight. (On a k-mesh the same identity needs the
    k-integral converged, which a supercell at the Gamma point reaches
    only when it is large.)"""
    rng = np.random.default_rng(2)
    g = geometry.honeycomb_lattice().get_supercell(3)
    g.dimensionality = 0 # an 18-site flake with open edges
    h = g.get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(0.5)
    h.intra = h.intra + np.diag(0.3*(rng.random(len(g.r))-0.5))
    W = kpmkubo.sum_rule_weight(h,T=0.05,trace="exact")[0,0]
    # the same weight from the eigenstates
    H = np.array(h.intra)
    x = np.array(g.r)[:,0]
    tau = -H*(x[None,:]-x[:,None])**2
    es,U = np.linalg.eigh(H)
    f = 1./(1.+np.exp(es/0.05))
    Wex = np.sum(f*np.real(np.diag(U.conj().T@tau@U)))
    assert abs(W-Wex) < 1e-4*abs(Wex)
    ws = np.linspace(0.,7.,281)
    ws,s = kpmkubo.optical_conductivity(h,energies=ws,T=0.05,delta=0.1,
            trace="exact",component="xx")
    weight = 2.*np.trapezoid(s.real,ws) # Re sigma_xx is even in w
    assert abs(weight-np.pi*Wex) < 0.01*np.pi*Wex


def test_unsupported_options_raise():
    h = _disordered_cell()
    with pytest.raises(ValueError,match="jackson"):
        h.get_optical_conductivity(integration="kpm",kernel="gauss")
    with pytest.raises(ValueError,match="exact"):
        h.get_optical_conductivity(integration="kpm",trace="hutch")
    with pytest.raises(NotImplementedError):
        h.get_optical_conductivity(integration="kpm",intraband=False)
    with pytest.raises(ValueError,match="singleshot"):
        h.get_dc_conductivity(mode="greenwood")
    with pytest.raises(ValueError,match="longitudinal"):
        h.get_dc_conductivity(mode="singleshot",component="xy")
    with pytest.raises(NotImplementedError,match="zero-temperature"):
        h.get_dc_conductivity(mode="singleshot",T=0.1)
    with pytest.raises(ValueError,match="npol"):
        h.get_dc_conductivity(mode="singleshot",npol=100)


def test_resolvent_batches_fit_the_memory_budget():
    """The single-shot resolvent holds a block of nb moments, three sets
    per energy of a batch (the sum, the product added to it and the batch
    the caller still holds) and four more sets, each set 16 n nc bytes.
    With 10^6 orbitals, nc = 3 and the default 2 GB, the block of 32
    vectors alone used to take 1.5 GB on top of the budget; now the total
    fits, and with room to spare the block and the batch are as before"""
    n, nc, memory = 10**6, 3, 2.
    unit = 16.*n*nc
    for nz in [1, 5, 100]:
        nb, nzb = kpmkubo._resolvent_batches(n, nc, nz, memory)
        assert nb >= 1 and 1 <= nzb <= nz
        assert (3*nzb + nb + 4)*unit <= memory*1e9
    assert kpmkubo._resolvent_batches(100, 3, 40, 2.) == (32, 40)


def test_resolvent_stays_within_its_memory_budget():
    """The peak of the arrays the single-shot resolvent allocates, traced
    by tracemalloc, stays inside the memory= budget, and a small budget,
    which runs the energies in several batches and the moments in small
    blocks, gives the same Im G D as an ample one. A budget of 30 sets of
    vectors used to hold the 32 stored vectors, the 30 energies and the
    product added to them, about three times the budget"""
    import tracemalloc
    import scipy.sparse as sp
    n = 8000
    H = kpmkubo._CSR(sp.diags([np.ones(n-1), np.ones(n-1)], [-1, 1],
            format="csr"))
    rng = np.random.default_rng(1)
    D = np.ascontiguousarray(rng.random((n, 3)) + 1j*rng.random((n, 3)))
    d0 = D[:, 0].copy()
    zs = np.linspace(-1., 1., 30) + 0.2j
    memory = 30*16.*n*3/1e9 # 30 sets of vectors

    def run(memory):
        out = np.zeros(len(zs), dtype=np.complex128)
        for j0, Y in kpmkubo._imaginary_resolvent(H, D, zs, 2.1, 0., memory):
            for j in range(len(Y)):
                out[j0+j] = np.vdot(d0, Y[j][:, 0])
        return out
    ref = run(10.) # every energy in one batch, and the kernels compiled
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        out = run(memory)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1.5*memory*1e9, (peak, memory*1e9)
    assert np.max(np.abs(out-ref)) < 1e-10*np.max(np.abs(ref))


def test_singleshot_does_not_depend_on_the_memory_budget():
    """A budget too small for one batch of all the Fermi levels runs them
    in several batches, with the same single-shot conductivity"""
    g = geometry.chain().get_supercell(60)
    h = g.get_hamiltonian(has_spin=False, is_sparse=True)
    fermis = np.linspace(-1.5, 1.5, 7)
    kw = dict(fermi=fermis, delta=0.3, mode="singleshot", component="xx",
            trace="exact")
    s0 = h.get_dc_conductivity(**kw)
    # the exact trace runs the 60 basis vectors as one block, so the
    # recursion carries v X and X, 120 columns: 20 sets of 60 x 120
    # vectors give blocks of 4 moments and batches of 4 Fermi levels
    memory = 20.5*16.*60*120/1e9 # 20 whole sets, away from a rounding
    assert kpmkubo._resolvent_batches(60, 120, len(fermis), memory) == (4, 4)
    s1 = h.get_dc_conductivity(memory=memory, **kw)
    assert np.max(np.abs(s1-s0)) < 1e-10*np.max(np.abs(s0))


@pytest.mark.parametrize("fermi,w", [(0.5, 0.01), (0.5, -0.01), (0.5, 1.3),
        (-1.69, 0.05), (0.5, 0.)])
def test_box_weights_integrate_the_chebyshev_series_exactly(fermi, w):
    """At T=0 the occupation factor is a box, 1/|w| between fermi-w and
    fermi, and its weights on the Chebyshev-Gauss nodes integrate every
    cos(k theta), with k well below the number of nodes, over the box in
    closed form, for a box far narrower than the node spacing, a box of
    the other sign, a wide box, a box cut at the edge of the rescaled
    spectrum, and the w -> 0 limit, the value at the Fermi level divided
    by dE/dtheta = a sin(theta)"""
    ne, a, b = 256, 2., 0.3
    q = kpmkubo._box_weights(ne, fermi, w, a, b)
    th = np.pi*(np.arange(ne)+0.5)/ne
    x = lambda E: np.clip((E-b)/a, -1., 1.)
    for k in [0, 1, 7, 40]:
        if w == 0.:
            t0 = np.arccos(x(fermi))
            ref = np.cos(k*t0)/(a*np.sin(t0))
        else:
            lo, hi = fermi-max(w, 0.), fermi-min(w, 0.)
            t1, t2 = np.arccos(x(hi)), np.arccos(x(lo))
            ref = (t2-t1) if k == 0 else (np.sin(k*t2)-np.sin(k*t1))/k
            ref = ref/abs(w)
        assert abs(np.sum(q*np.cos(k*th))-ref) < 1e-8, (k, ref)


def _box_reference(st, mu, w, fermi, n=200):
    """pi/|w| times the integral of rho_ab(E,E+w) over the box between
    fermi-w and fermi, with rho_ab(E,E') = sum_nm mu_nm phi_n(E') phi_m(E)
    the double series of the module docstring, by Gauss-Legendre on the box
    itself, so no node of absorptive() is used"""
    a, b, M = st.a, st.b, st.M
    lo, hi = fermi-max(w, 0.), fermi-min(w, 0.)
    xg, wg = np.polynomial.legendre.leggauss(n)
    E = (hi-lo)*xg/2. + (hi+lo)/2.

    def phi(E): # phi_n(E), (M,len(E))
        x = (E-b)/a
        return kpmkubo._phi(x, M)/(np.pi*a*np.sqrt(1.-x**2))
    rho = np.einsum("abnm,nj,mj->abj", mu, phi(E+w), phi(E))
    return np.pi*(rho@(wg*(hi-lo)/2.))/abs(w)


@pytest.mark.parametrize("on_node", [True, False])
def test_zero_temperature_absorption_is_the_box_integral(on_node):
    """At T=0 the absorptive part is pi/|w| times the integral of the
    double Chebyshev series rho(E,E+w) over the box between fermi-w and
    fermi. On a metallic cell with exact moments it agrees with that
    integral done by Gauss-Legendre on the box, for boxes 20 and 3 times
    narrower than the node spacing, a few spacings wide, much wider, and
    of the other sign, with the Fermi level on a node and between two.

    The box used to be sampled on the nodes, so a box narrower than their
    spacing held zero or one of them and gave 0 or several times the
    integral"""
    h = _disordered_cell()
    h.shift_fermi(1.2) # the Fermi level in the valence band, a metal
    st = kpmkubo._Setup(h, nk=2, npol=60)
    mu = kpmkubo.exact_moments(st)
    ne = 4*st.M
    E = st.a*np.cos(np.pi*(np.arange(ne)+0.5)/ne) + st.b
    j = np.argmin(np.abs(E-0.05))
    spacing = abs(E[j+1]-E[j])
    fermi = E[j] if on_node else E[j] + 0.37*(E[j+1]-E[j])
    ws = np.array([spacing/20., spacing/3., 2.5*spacing, 0.4, -spacing/7.])
    out = kpmkubo.absorptive(st, mu, ws, fermi=fermi, T=0.)
    ref = [_box_reference(st, mu, w, fermi) for w in ws]
    scale = max(np.max(np.abs(r)) for r in ref)
    assert scale > 1e-3 # a real signal
    d = st.dirs
    for iw in range(len(ws)):
        assert np.max(np.abs(out[iw][np.ix_(d, d)]-ref[iw])) < 1e-6*scale, \
                ws[iw]


def test_zero_temperature_drude_peak_is_smooth_and_matches_the_bands():
    """A clean ring of 400 sites with the Fermi level at 0.1 in the band
    has a Drude peak D delta/(w^2 + delta^2), with D = sqrt(4 - mu^2)/pi.
    At T=0 the KPM peak is a smooth function of w down to w = 0, its top,
    so a frequency 64 times below delta, far below the spacing of the
    energy nodes (about delta/7), gives the w = 0 value to 1e-3, and the
    whole peak follows the band calculation (integration="grid" at
    T=0.01 on 4000 k-points, the chain itself) to the few percent of the
    Lorentz kernel. The band calculation takes the line width the KPM
    realizes with its integer number of polynomials, 0.197 for 0.2, and
    the Fermi level is near the center, where the kernel is not narrowed.

    The occupation box used to be sampled on the nodes: zero at w = 0,
    and 0 or several times the peak below the node spacing"""
    fermi = 0.1
    g = geometry.chain().get_supercell(400)
    h = g.get_hamiltonian(has_spin=False, is_sparse=True)
    st = kpmkubo._Setup(h, delta=0.2, kernel="lorentz")
    delta = st.delta # the line width of npol polynomials
    assert abs(delta-0.2) < 0.01
    ws = delta*np.array([0., 1./64, 1./32, 1./16, 1./8, 1./2, 1., 2.])
    ws, s = conductivity.optical_conductivity(h, ws, integration="kpm",
            T=0., npol=st.npol, fermi=fermi, kernel="lorentz",
            trace="exact", reactive=False, component="xx")
    s = s.real
    h0 = geometry.chain().get_hamiltonian(has_spin=False)
    h0.shift_fermi(-fermi) # the band path occupies the states below zero
    ws, s0 = conductivity.optical_conductivity(h0, ws, nk=4000, T=0.01,
            delta=delta, component="xx")
    s0 = s0.real
    D = np.sqrt(4.-fermi**2)/np.pi
    assert abs(s0[0]-D/delta) < 0.02*D/delta # the reference is the peak
    assert abs(s[1]-s[0]) < 1e-3*s[0] # w = delta/64 against w = 0
    assert np.all(np.diff(s) < 0.) # the peak falls monotonically
    assert np.max(np.abs(s-s0)) < 0.05*s0[0]
