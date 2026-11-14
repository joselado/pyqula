import numpy as np
import pytest

from pyqula import geometry
from pyqula import sculpt
from pyqula import spectrum


def _folded_dirac_subspace(h0):
    """Return an orthonormal basis for the 4-dimensional zero-energy
    subspace of a Kekule-commensurate honeycomb Hamiltonian at Gamma (the
    folded K,K' Dirac points)"""
    hk0 = h0.get_hk_gen()([0.,0.,0.])
    (es,vs) = np.linalg.eigh(hk0)
    inds = np.where(np.abs(es)<1e-6)[0]
    assert len(inds)==4
    return vs[:,inds]


def test_inplane_valley_pseudospin_algebra():
    """tau_x, tau_y (get_operator("valley_x"/"valley_y")) and tau_z
    (get_operator("valley")), projected onto the low-energy (folded
    K,K' Dirac point) subspace of a Kekule-commensurate honeycomb cell,
    must satisfy the expected Pauli pseudospin algebra: each squares to
    the identity, and any two of them anticommute."""
    g3 = geometry.honeycomb_lattice().supercell(3)
    h0 = g3.get_hamiltonian(has_spin=False)
    h0.turn_multicell()
    V = _folded_dirac_subspace(h0)

    opx = h0.get_operator("valley_x")
    opy = h0.get_operator("valley_y")
    opz = h0.get_operator("valley")
    k = [0.,0.,0.]
    Mx = V.conj().T@opx.m(None,k=k)@V
    My = V.conj().T@opy.m(None,k=k)@V
    Mz = V.conj().T@opz.m(None,k=k)@V

    I4 = np.eye(4)
    assert np.max(np.abs(Mx@Mx-I4))<1e-8
    assert np.max(np.abs(My@My-I4))<1e-8
    assert np.max(np.abs(Mz@Mz-I4))<1e-8
    assert np.max(np.abs(Mx@Mz+Mz@Mx))<1e-8
    assert np.max(np.abs(My@Mz+Mz@My))<1e-8
    assert np.max(np.abs(Mx@My+My@Mx))<1e-8


def test_kekule_gapped_states_are_valley_x_eigenstates():
    """Gapping the Dirac points (exactly at the folded K,K' point, k=Gamma
    of the tripled cell) with a Kekule mass added through
    add_valley_exchange (a first-neighbor hopping modulated by a plane
    wave of wavevector K-K', see operatortk/inplane_valley.py) must
    produce the gap-edge eigenstates as exact +-1 eigenvectors of the
    tau_x ("valley_x") operator."""
    g3 = geometry.honeycomb_lattice().supercell(3)
    h = g3.get_hamiltonian(has_spin=False)
    h.add_valley_exchange([0.4,0.,0.])
    h.turn_multicell()
    hk = h.get_hk_gen()([0.,0.,0.])
    (es,vs) = np.linalg.eigh(hk)
    near_gap = np.abs(np.abs(es)-0.4)<1e-6 # the induced gap is exactly 0.4
    assert np.sum(near_gap)==4 # the 4 folded Dirac states

    opx = h.get_operator("valley_x")
    Mx = opx.m(None,k=[0.,0.,0.])
    for i in np.where(near_gap)[0]:
        w = vs[:,i]
        ev = np.real(np.conj(w)@Mx@w) # expectation value, should be +-1
        assert np.isclose(abs(ev),1.0,atol=1e-6)


def test_add_valley_exchange_matches_operator_convention():
    """add_valley_exchange([vx,vy,vz]) must open a gap of exactly
    |v|=sqrt(vx^2+vy^2+vz^2) at the folded Dirac point, with the same
    (tau_x,tau_y,tau_z) normalization as get_operator("valley_x"/
    "valley_y"/"valley"), for an arbitrary (not just axis-aligned)
    direction in valley pseudospin space."""
    v = np.array([0.1,0.15,0.25])
    g3 = geometry.honeycomb_lattice().supercell(3)
    h = g3.get_hamiltonian(has_spin=False)
    h.add_valley_exchange(list(v))
    h.turn_multicell()
    es = np.linalg.eigvalsh(h.get_hk_gen()([0.,0.,0.]))
    gap = np.linalg.norm(v)
    near_gap = np.abs(np.abs(es)-gap)<1e-6
    assert np.sum(near_gap)==4 # the 4 folded Dirac states


# what follows pins the properties that the pseudospin algebra at the
# Dirac point cannot see: the three tests above pass for any pair of
# operators that flips the valley, also for one that breaks time reversal
# or that is wrong at the edges of a finite system


def _valley_matrices(h,k=[0.,0.,0.]):
    """Return (tau_x,tau_y,tau_z) of h at k as dense arrays"""
    names = ["valley_x","valley_y","valley"]
    if h.dimensionality==0:
        return [np.array(h.get_operator(n).matrix) for n in names]
    return [np.array(h.get_operator(n).m(None,k=k)) for n in names]


def _texture(h):
    """Return tau_x + i tau_y on every site, summed over the occupied
    states"""
    vx = spectrum.real_space_vev(h,operator=h.get_operator("valley_x"),
            write=False)
    vy = spectrum.real_space_vev(h,operator=h.get_operator("valley_y"),
            write=False)
    return vx + 1j*vy


def _disk(radius,center_sublattice=1):
    """Return a honeycomb disk centred on an atom of the given sublattice,
    with that atom at the origin, and its index. The squared distances
    between sites are integers, so a radius whose square is not one leaves
    no site on the rim for the rounding to decide"""
    g = geometry.honeycomb_lattice().supercell(int(2*radius)+4)
    g.set_finite() ; g.center()
    i0 = [i for i in g.get_central(n=4)
            if g.sublattice[i]==center_sublattice][0]
    sculpt.shift(g,r=g.r[i0].copy())
    g = sculpt.intersec(g,lambda r: r.dot(r)<radius**2)
    g = sculpt.remove_unibonded(g,iterative=True)
    return g,int(np.argmin(np.linalg.norm(g.r,axis=1)))


def _shuffled(g,seed=0):
    """Return the same geometry with its atoms in a random order, and the
    permutation"""
    p = np.random.default_rng(seed).permutation(len(g.r))
    go = g.copy()
    go.r = g.r[p].copy() ; go.r2xyz()
    go.sublattice = np.array(g.sublattice)[p]
    go.has_fractional = False
    return go,p


def test_inplane_valley_is_time_reversal_even():
    """Time reversal exchanges K and K', so it flips tau_z and leaves
    tau_x and tau_y: for a spinless real Hamiltonian they are real
    symmetric hoppings, tau(-k) = conj(tau(k)), and a valley exchange in
    the plane keeps the Hamiltonian real."""
    g3 = geometry.honeycomb_lattice().supercell(3)
    h = g3.get_hamiltonian(has_spin=False)
    k = np.array([0.13,0.31,0.])
    for name in ["valley_x","valley_y"]:
        op = h.get_operator(name)
        m0 = np.array(op.m(None,k=[0.,0.,0.]))
        assert np.max(np.abs(m0.imag))<1e-12
        assert np.max(np.abs(m0-m0.T))<1e-12
        mk,mmk = np.array(op.m(None,k=k)),np.array(op.m(None,k=-k))
        assert np.max(np.abs(mmk-mk.conj()))<1e-12
    h.add_valley_exchange([0.2,0.3,0.])
    hk,hmk = np.array(h.get_hk_gen()(k)),np.array(h.get_hk_gen()(-k))
    assert np.max(np.abs(hmk-hk.conj()))<1e-12


def test_inplane_valley_is_right_handed_for_any_atom_order():
    """tau_x tau_y = i tau_z on the Dirac point, the full Pauli algebra and
    not only the anticommutators, and the operators of a geometry with its
    atoms in another order are the same operators with their rows
    permuted: neither the handedness nor the axes follow which atom
    happens to come first."""
    g3 = geometry.honeycomb_lattice().supercell(3)
    h0 = g3.get_hamiltonian(has_spin=False)
    ms0 = _valley_matrices(h0)
    for seed in range(3):
        g,p = _shuffled(g3,seed=seed)
        h = g.get_hamiltonian(has_spin=False)
        V = _folded_dirac_subspace(h.get_multicell())
        ms = _valley_matrices(h)
        (Mx,My,Mz) = [V.conj().T@m@V for m in ms]
        assert np.max(np.abs(Mx@My-1j*Mz))<1e-8
        for (m,m0) in zip(ms,ms0):
            assert np.max(np.abs(m-m0[np.ix_(p,p)]))<1e-10


def test_inplane_valley_exchange_gaps_the_whole_brillouin_zone():
    """v_x tau_x + v_y tau_y is a mass of the Dirac equation, the Kekule
    mass: the gap is |v| at the Dirac point and nowhere smaller. An
    in-plane valley operator that commutes with the Dirac Hamiltonian
    instead passes every test at the Dirac point and closes the gap away
    from it."""
    g3 = geometry.honeycomb_lattice().supercell(3)
    h = g3.get_hamiltonian(has_spin=False)
    h.add_valley_exchange([0.18,0.24,0.]) # |v| = 0.3
    hk = h.get_hk_gen()
    gap = min([np.min(np.abs(np.linalg.eigvalsh(hk(k))))
            for k in g3.get_kmesh(nk=12)])
    assert abs(gap-0.3)<1e-8


def test_kekule_distortion_is_an_inplane_valley_exchange():
    """A plain Kekule distortion of strength t, h.add_kekule(t), built by
    an independent routine that paints the bonds of one hexagon out of
    three, is a uniform first-neighbor hopping 2t/3 plus an in-plane valley
    exchange of modulus |t| exactly, on every bond."""
    t = 0.3
    g3 = geometry.honeycomb_lattice().supercell(3)
    h0 = g3.get_hamiltonian(has_spin=False).get_multicell()
    h = h0.copy() ; h.add_kekule(t)
    k = [0.07,0.19,0.]
    (mx,my,mz) = _valley_matrices(h0,k=k)
    hk0 = np.array(h0.get_hk_gen()(k))
    dm = np.array(h.get_hk_gen()(k)) - (1.+2.*t/3.)*hk0 # the modulation
    vx = np.trace(mx@dm).real/np.trace(mx@mx).real
    vy = np.trace(my@dm).real/np.trace(my@my).real
    assert np.max(np.abs(dm-vx*mx-vy*my))<1e-10
    assert abs(np.sqrt(vx**2+vy**2)-abs(t))<1e-10


def test_translation_rotates_the_inplane_valley_by_120_degrees():
    """A translation by a lattice vector of the primitive cell is a
    rotation of (tau_x,tau_y) by 120 degrees, since the operators carry
    the momentum K-K': shifting the atoms of a Kekule cell by one
    primitive vector gives tau_x cos(2 pi/3) +- tau_y sin(2 pi/3)."""
    g0 = geometry.honeycomb_lattice()
    g3 = g0.supercell(3)
    gs = g3.copy() ; sculpt.shift(gs,r=g0.a1)
    (mx,my,mz) = _valley_matrices(g3.get_hamiltonian(has_spin=False))
    (sx,sy,sz) = _valley_matrices(gs.get_hamiltonian(has_spin=False))
    (c,s) = (np.cos(2.*np.pi/3.),np.sin(2.*np.pi/3.))
    d = [np.max(np.abs(sx-c*mx-sign*s*my)) for sign in [1.,-1.]]
    assert min(d)<1e-10 and max(d)>1e-2
    assert np.max(np.abs(sz-mz))<1e-10 # tau_z does not see the shift


def test_small_shift_of_the_geometry_keeps_the_valley_axes():
    """The phase of the operators is measured from the lattice site
    closest to the origin of coordinates, so a shift of the geometry that
    is small on the scale of the bond changes nothing. The flakes tested
    have their origin where it usually is, at the same distance from
    several sites (a bond midpoint, a hexagon center), which is where a
    rule that took the closest site to the last digit would jump by 120
    degrees. Recentering a flake after removing an atom is such a shift."""
    from pyqula import islands
    for rot in [0.,np.pi/6.]:
        g = islands.get_geometry(name="honeycomb",n=4,nedges=6,rot=rot)
        (mx,my,mz) = _valley_matrices(g.get_hamiltonian(has_spin=False))
        rng = np.random.default_rng(1)
        for i in range(4):
            d = 0.1*rng.uniform(-1.,1.,3) ; d[2] = 0.
            gs = g.copy() ; sculpt.shift(gs,r=d)
            (sx,sy,sz) = _valley_matrices(gs.get_hamiltonian(has_spin=False))
            assert np.max(np.abs(sx-mx))<1e-10
            assert np.max(np.abs(sy-my))<1e-10
        i0 = int(g.get_central()[0])
        gv = g.remove(i0)
        gc = gv.copy() ; gc.center() # the center of mass moved
        assert 1e-4<np.max(np.abs(gc.r-gv.r))<0.1
        keep = [i for i in range(len(g.r)) if i!=i0]
        (vx,vy,vz) = _valley_matrices(gc.get_hamiltonian(has_spin=False))
        assert np.max(np.abs(vx-mx[np.ix_(keep,keep)]))<1e-10
        assert np.max(np.abs(vy-my[np.ix_(keep,keep)]))<1e-10


def test_flake_operator_is_the_bulk_operator_on_its_bonds():
    """The operators of a finite flake are those of the infinite lattice
    restricted to the bonds the flake has: T = tau_x + i tau_y has modulus
    2/3 on every bond, also on those at the edge, and cutting a flake out
    of a larger one cuts the operator with it."""
    (gb,ib) = _disk(8.2)
    gs = sculpt.intersec(gb,lambda r: r.dot(r)<5.2**2)
    gs = sculpt.remove_unibonded(gs,iterative=True)
    idx = [int(np.argmin(np.linalg.norm(gb.r-r,axis=1))) for r in gs.r]
    assert np.max(np.abs(gb.r[idx]-gs.r))<1e-8
    (bx,by,bz) = _valley_matrices(gb.get_hamiltonian(has_spin=False))
    hs = gs.get_hamiltonian(has_spin=False)
    (mx,my,mz) = _valley_matrices(hs)
    assert np.max(np.abs(mx-bx[np.ix_(idx,idx)]))<1e-10
    assert np.max(np.abs(my-by[np.ix_(idx,idx)]))<1e-10
    bonds = np.abs(np.array(hs.intra))>0.5 # the first neighbors
    T = mx + 1j*my
    assert np.max(np.abs(np.abs(T[bonds])-2./3.))<1e-10
    assert np.max(np.abs(T[~bonds]))<1e-12


@pytest.mark.parametrize("sublattice,winding",[(1,-1),(-1,1)])
def test_vacancy_creates_a_vortex_of_the_inplane_valley(sublattice,winding):
    """A vacancy scatters between the valleys and creates a vortex of
    (tau_x,tau_y). On a disk centred on the vacancy the texture it
    induces, f = tau_x + i tau_y, obeys f(C3 r) = exp(i 2 pi m/3) f(r)
    exactly, which leaves the angular harmonics m, m+-3, m+-6..., and far
    from the vacancy the harmonic exp(i m theta) dominates, with m=-1 for
    a vacancy on sublattice +1 and m=+1 on sublattice -1: the two
    sublattices see the Dirac points with opposite chirality."""
    (g,i0) = _disk(11.2,center_sublattice=sublattice)
    gv = g.remove(i0) # same coordinates, so the same valley axes
    f0 = _texture(g.get_hamiltonian(has_spin=False))
    f = _texture(gv.get_hamiltonian(has_spin=False))
    df = f - np.delete(f0,i0) # the part created by the vacancy
    c3 = np.array([[-0.5,-np.sqrt(3.)/2.,0.],[np.sqrt(3.)/2.,-0.5,0.],
        [0.,0.,1.]])
    perm = [int(np.argmin(np.linalg.norm(gv.r-r,axis=1))) for r in gv.r@c3.T]
    assert np.max(np.abs(gv.r[perm]-gv.r@c3.T))<1e-8
    assert np.max(np.abs(df[perm]-np.exp(2j*np.pi*winding/3.)*df))<1e-8
    rr = np.linalg.norm(gv.r,axis=1)
    theta = np.arctan2(gv.r[:,1],gv.r[:,0])
    assert np.max(np.abs(df[rr<1.1]))>0.1 # large next to the vacancy
    ring = (rr>6.2)&(rr<9.1) # far from the vacancy and from the edge
    cm = {m: abs(np.mean(df[ring]*np.exp(-1j*m*theta[ring])))
            for m in range(-5,6)}
    others = max([cm[m] for m in cm if m!=winding])
    assert cm[winding]>5e-4
    assert others<cm[winding]/3.
    forbidden = max([cm[m] for m in cm if (m-winding)%3!=0])
    assert forbidden<1e-8 # the harmonics that C3 does not allow


def test_zigzag_edge_has_no_net_inplane_valley_and_armchair_does():
    """An armchair edge mixes the two valleys and a zigzag edge does not.
    In a zigzag ribbon the texture rotates by 120 degrees from one cell to
    the next, so it adds to zero over the three cells of the Kekule
    period, while its modulus at the edge is large. In an armchair ribbon
    it is the same in every cell, large at the edge, and of the same
    modulus at the two edges, which a rotation by 180 degrees maps on each
    other."""
    gz = geometry.honeycomb_zigzag_ribbon(4)
    nz = len(gz.r)
    fz = _texture(gz.supercell(6).get_hamiltonian(has_spin=False))
    fz = fz.reshape((6,nz)) # cell, site in the cell
    assert np.max(np.abs(fz))>0.1
    assert np.max(np.abs(fz[0]+fz[1]+fz[2]))<1e-8
    rot = np.abs(fz[1]/fz[0]-np.exp(2j*np.pi/3.))<1e-6
    rotm = np.abs(fz[1]/fz[0]-np.exp(-2j*np.pi/3.))<1e-6
    assert np.all(rot) or np.all(rotm)
    ga = geometry.honeycomb_armchair_ribbon(6)
    na = len(ga.r)
    gas = ga.supercell(4)
    fa = _texture(gas.get_hamiltonian(has_spin=False)).reshape((4,na))
    assert np.max(np.abs(fa-fa[0][None,:]))<1e-8 # the same in every cell
    across = ga.r[:,int(np.argmin(np.abs(ga.a1[:2])))] # across the ribbon
    across = across - np.mean(across)
    top = np.abs(across-np.max(across))<1e-3
    bottom = np.abs(across-np.min(across))<1e-3
    assert np.sum(top)==np.sum(bottom) and np.sum(top)>0
    assert np.min(np.abs(fa[0][top]))>0.1
    assert abs(np.mean(np.abs(fa[0][top]))-np.mean(np.abs(fa[0][bottom])))<1e-8


def _aa_bilayer():
    from pyqula import specialgeometry
    return specialgeometry.multilayer_graphene(l=[0,0])


def _ab_bilayer():
    from pyqula import specialgeometry
    return specialgeometry.multilayer_graphene(l=[0,1])


@pytest.mark.parametrize("builder,nlayers",[
    (geometry.buckled_honeycomb_lattice,1),(_aa_bilayer,2),(_ab_bilayer,2)])
def test_inplane_valley_on_buckled_and_stacked_honeycombs(builder,nlayers):
    """Each bond takes K from its own in-plane direction and from the
    sublattice labels of its two sites, so a buckled honeycomb and every
    layer of a stack get the full Pauli algebra with the tau_z of the same
    labels, tau_x tau_y = i tau_z on the Dirac points, and v.tau opens the
    gap |v|. The AA stack labels its second layer the other way around,
    which exchanges the valleys there for tau_z and tau_y alike. The
    buckled lattice has a shorter in-plane bond, which the normalization
    of tau_z has to follow: with the one of the flat lattice its
    eigenvalues were +-8/9."""
    g3 = builder().supercell(3)
    h = g3.get_hamiltonian(has_spin=False)
    hk0 = np.array(h.get_hk_gen()([0.,0.,0.]))
    (es,vs) = np.linalg.eigh(hk0)
    V = vs[:,np.abs(es)<1e-6]
    assert V.shape[1]==4*nlayers # the layers are not coupled
    (Mx,My,Mz) = [V.conj().T@m@V for m in _valley_matrices(h)]
    I = np.eye(V.shape[1])
    assert np.max(np.abs(Mx@Mx-I))<1e-8
    assert np.max(np.abs(My@My-I))<1e-8
    assert np.max(np.abs(Mz@Mz-I))<1e-8
    assert np.max(np.abs(Mx@My-1j*Mz))<1e-8
    v = np.array([0.1,0.15,0.25])
    h.add_valley_exchange(list(v))
    es = np.linalg.eigvalsh(np.array(h.get_hk_gen()([0.,0.,0.])))
    assert abs(np.min(np.abs(es))-np.linalg.norm(v))<1e-8


def _pulled_apart(g):
    """The same geometry with the sites above the middle height moved up by
    two bond lengths, so that the vertical first-neighbor bonds between the
    layers are gone and every in-plane bond is the same"""
    go = g.copy()
    z = np.array(go.r)[:,2]
    go.r = np.array(go.r) + 2.*np.outer(z>np.mean(z),[0.,0.,1.])
    go.r2xyz()
    return go


@pytest.mark.parametrize("builder",["aa_dz1","buckled_film"])
def test_vertical_bonds_carry_no_inplane_valley(builder):
    """A vertical bond as long as the in-plane one, between two layers on
    top of each other, has no in-plane direction to take K from. It gave
    K = 0/0, and the operators came back with NaN entries and no error.
    The plane wave lives on the bonds of each layer, so the operators of
    the stack are those of the same layers pulled apart, where the
    vertical bonds are not first neighbors any more. The AA stack of
    multilayer at its default dz=1 and the (111) diamond film of two
    buckled layers both have such bonds"""
    from pyqula import specialgeometry
    if builder=="aa_dz1":
        g = specialgeometry.multilayer(geometry.honeycomb_lattice(),l=[0,0])
    else: g = geometry.buckled_honeycomb_lattice(2)
    g = g.supercell(3)
    go = _pulled_apart(g)
    assert np.max(np.abs(np.array(g.r)[:,:2]-np.array(go.r)[:,:2]))<1e-12
    k = [0.1,0.2,0.]
    (h,ho) = [x.get_hamiltonian(has_spin=False) for x in (g,go)]
    for name in ["valley_x","valley_y"]:
        (m,mo) = [_dense(x.get_operator(name).m(None,k=k)) for x in (h,ho)]
        assert np.all(np.isfinite(m))
        assert np.max(np.abs(m-mo))<1e-12
        assert np.max(np.abs(mo))>0.1 # really there
    h = g.get_hamiltonian(has_spin=False)
    h.add_valley_exchange([0.1,0.2,0.])
    assert np.all(np.isfinite(np.array(h.get_hk_gen()(k))))


def test_bond_without_inplane_direction_cannot_pass_the_guards(monkeypatch):
    """The guards of the geometry are written so that a NaN fails them.
    With the vertical bonds let through again, the AA stack at dz=1 has
    bonds with K = 0/0, and it has to raise and not hand back operators
    with NaN entries, which is what the guards written as max(...)>tol
    did"""
    from pyqula import specialgeometry
    from pyqula.operatortk import inplane_valley
    monkeypatch.setattr(inplane_valley,"_VERTICAL",-1.) # keep every bond
    g = specialgeometry.multilayer(geometry.honeycomb_lattice(),l=[0,0])
    h = g.supercell(3).get_hamiltonian(has_spin=False)
    for name in ["valley_x","valley_y"]:
        with np.errstate(invalid="ignore",divide="ignore"):
            with pytest.raises(ValueError,match="120 degrees apart"):
                h.get_operator(name)
    with np.errstate(invalid="ignore",divide="ignore"):
        with pytest.raises(ValueError,match="120 degrees apart"):
            h.add_valley_exchange([0.1,0.2,0.])


def test_inplane_valley_of_a_twisted_bilayer_needs_a_tripled_cell():
    """In a twisted bilayer each layer has its own K. The moire cell does
    not hold a period of the plane wave K-K' of the layers, so it is
    refused, and three moire cells do: there the two decoupled layers get
    the full Pauli algebra, each with the K of its own orientation."""
    from pyqula import specialgeometry
    g = specialgeometry.twisted_bilayer(1)
    with pytest.raises(ValueError,match="multiple of 3"):
        g.get_hamiltonian(has_spin=False).get_operator("valley_x")
    g3 = g.get_supercell(3)
    h = g3.get_hamiltonian(has_spin=False) # first neighbors: decoupled layers
    hk0 = np.array(h.get_hk_gen()([0.,0.,0.]))
    (es,vs) = np.linalg.eigh(hk0)
    V = vs[:,np.abs(es)<1e-6]
    assert V.shape[1]==8 # the Dirac points of the two layers
    (Mx,My,Mz) = [V.conj().T@m@V for m in _valley_matrices(h)]
    I = np.eye(8)
    assert np.max(np.abs(Mx@Mx-I))<1e-8
    assert np.max(np.abs(My@My-I))<1e-8
    assert np.max(np.abs(Mx@My-1j*Mz))<1e-8


def _route_geometry(name):
    """Small honeycomb geometries of every dimensionality"""
    from pyqula import islands
    if name=="island": return islands.get_geometry(name="honeycomb",n=2,
            nedges=6)
    if name=="zigzag": return geometry.honeycomb_zigzag_ribbon(2).supercell(3)
    if name=="armchair": return geometry.honeycomb_armchair_ribbon(2)
    if name=="cell": return geometry.honeycomb_lattice().supercell(3)
    if name=="stack": # the cell repeated along z, ten bonds apart
        g = geometry.honeycomb_lattice().supercell(3)
        g.dimensionality = 3
        return g


def _route_hamiltonian(g,space,sparse):
    """A spinless, spinful or Nambu Hamiltonian of g, sparse or dense"""
    h = g.get_hamiltonian(has_spin=(space!="spinless"))
    if sparse: h.turn_sparse()
    if space=="nambu": h.setup_nambu_spinor()
    return h


def _dense(m):
    from scipy.sparse import issparse
    return m.toarray() if issparse(m) else np.array(m)


@pytest.mark.parametrize("sparse",[False,True])
@pytest.mark.parametrize("space",["spinless","spinful","nambu"])
@pytest.mark.parametrize("name",["island","zigzag","armchair","cell","stack"])
def test_inplane_valley_through_every_route(name,space,sparse):
    """The operators and the valley exchange are the same physics through
    every way of storing the Hamiltonian: a flake, a ribbon, a cell and a
    stack of cells, spinless, spinful and with the electron-hole degree of
    freedom, dense and sparse. A sparse Hamiltonian gets sparse matrices
    equal to the dense ones, a spinful one the spinless operator on each
    spin, the Nambu one the Nambu form of the spinful one with the same
    expectation values, and an in-plane angle the combination of tau_x and
    tau_y."""
    from scipy.sparse import issparse
    from pyqula.operatortk.inplane_valley import get_inplane_valley
    g = _route_geometry(name)
    k = [[0.,0.,0.],[0.13,0.,0.],[0.13,0.21,0.],[0.13,0.21,0.34]][
            g.dimensionality]
    v = [0.2,0.1,0.15]
    def matrix(h,opname):
        op = h.get_operator(opname)
        return op.matrix if h.dimensionality==0 else op.m(None,k=k)
    h = _route_hamiltonian(g,space,sparse)
    hd = _route_hamiltonian(g,space,False) # the dense reference
    h0 = _route_hamiltonian(g,"spinless",False)
    ms = {}
    for opname in ["valley_x","valley_y"]:
        m = matrix(h,opname)
        assert issparse(m)==sparse
        ms[opname] = _dense(m)
        assert np.max(np.abs(ms[opname]-_dense(matrix(hd,opname))))<1e-12
        assert np.max(np.abs(ms[opname]-ms[opname].conj().T))<1e-12
        if space=="spinful": # the spinless operator on each spin
            m0 = np.kron(_dense(matrix(h0,opname)),np.eye(2))
            assert np.max(np.abs(ms[opname]-m0))<1e-12
    ma = get_inplane_valley(h,angle=0.7) # a generic in-plane direction
    if h.dimensionality>0: ma = ma(None,k=k)
    comb = np.cos(0.7)*ms["valley_x"] + np.sin(0.7)*ms["valley_y"]
    assert np.max(np.abs(_dense(ma)-comb))<1e-12
    # a valley exchange, in the Hamiltonian
    h.add_valley_exchange(v) ; hd.add_valley_exchange(v)
    assert h.is_sparse==sparse and issparse(h.intra)==sparse
    d = _dense(h.get_hk_gen()(k)) - _dense(hd.get_hk_gen()(k))
    assert np.max(np.abs(d))<1e-12
    hn = _route_hamiltonian(g,"spinful" if space=="nambu" else space,False)
    hn.add_valley_exchange(v) # the normal state with the same exchange
    if space=="nambu": # the Nambu form of the normal Hamiltonian
        ref = hn.copy() ; ref.setup_nambu_spinor()
        d = _dense(h.get_hk_gen()(k)) - _dense(ref.get_hk_gen()(k))
        assert np.max(np.abs(d))<1e-12
    if name=="stack": return # the texture is the one of the cell
    # and the texture it creates
    (f,fn) = (_texture(h),_texture(hn))
    assert np.max(np.abs(fn))>0.02
    assert np.max(np.abs(f-fn))<1e-8


@pytest.mark.parametrize("name",["island","cell"])
def test_sharpened_valley_is_the_same_dense_and_sparse(name):
    """delta= renormalizes the eigenvalues e of the valley operators to
    e/(|e|+delta). On a sparse Hamiltonian it was dropped without a word,
    since the sharpening returned a sparse matrix untouched; now a sparse
    and a dense Hamiltonian of the same geometry give the same sharpened
    operator, in the storage of each"""
    from scipy.sparse import issparse
    g = _route_geometry(name)
    k = [[0.,0.,0.],None,[0.13,0.21,0.]][g.dimensionality]
    delta = 0.1
    def matrix(h,opname,**kw):
        op = h.get_operator(opname,**kw)
        return op.matrix if h.dimensionality==0 else op.m(None,k=k)
    hd = _route_hamiltonian(g,"spinless",False)
    hs = _route_hamiltonian(g,"spinless",True)
    for opname in ["valley_x","valley_y","valley"]:
        raw = _dense(matrix(hd,opname))
        md = matrix(hd,opname,delta=delta)
        ms = matrix(hs,opname,delta=delta)
        assert issparse(ms)==issparse(matrix(hs,opname)) # same storage
        assert not issparse(md)
        assert np.max(np.abs(_dense(ms)-_dense(md)))<1e-12
        (e0,v0) = np.linalg.eigh(raw) # the same eigenvectors, renormalized
        ref = (v0*(e0/(np.abs(e0)+delta))[None,:])@v0.conj().T
        assert np.max(np.abs(_dense(md)-ref))<1e-10
        assert np.max(np.abs(_dense(md)-raw))>1e-2 # delta does something


@pytest.mark.parametrize("delta",[None,0.1])
def test_operator_kept_between_calls_is_the_one_of_each_k_point(delta):
    """The operator keeps its matrix at the last k-point, since the states
    of one k-point are visited one after the other. Going back and forth
    between k-points, and writing into a returned matrix, must give the
    matrix that a new operator builds at that k-point"""
    from pyqula.operatortk.inplane_valley import get_inplane_valley
    h = geometry.honeycomb_lattice().supercell(3).get_hamiltonian(
            has_spin=False)
    op = get_inplane_valley(h,angle=0.4,delta=delta)
    ks = [[0.1,0.2,0.],[0.31,-0.12,0.],[0.1,0.2,0.],np.array([0.1,0.2,0.]),
          [0.31,-0.12,0.]]
    v = np.random.default_rng(1).normal(size=h.intra.shape[0])
    for k in ks:
        ref = get_inplane_valley(h,angle=0.4,delta=delta)(None,k=k)
        m = op(None,k=k)
        assert np.max(np.abs(m-ref))<1e-12
        assert np.max(np.abs(op(v,k=k)-ref@v))<1e-12
        m[:,:] = 0. # the caller owns the matrix it gets
    m1 = op(None,k=ks[0])
    assert np.max(np.abs(m1-op(None,k=ks[1])))>1e-2 # it does depend on k


def test_stack_of_cells_has_the_operators_of_one_layer():
    """Layers ten bonds apart are not coupled, so the operators of the
    three-dimensional stack are those of the two-dimensional cell, at any
    momentum along the stack."""
    g2 = _route_geometry("cell")
    g3 = _route_geometry("stack")
    (h2,h3) = [g.get_hamiltonian(has_spin=False) for g in (g2,g3)]
    for kz in [0.,0.37]:
        m2 = _valley_matrices(h2,k=[0.13,0.21,0.])
        m3 = _valley_matrices(h3,k=[0.13,0.21,kz])
        for (a,b) in zip(m2,m3): assert np.max(np.abs(a-b))<1e-12


def test_inplane_valley_in_nambu_space_is_the_one_of_the_normal_state():
    """With the electron-hole degree of freedom and no pairing nothing
    physical changes: a valley exchange added to the Nambu Hamiltonian is
    the Nambu form of the normal Hamiltonian that has it, and the texture
    it creates is the same."""
    g3 = geometry.honeycomb_lattice().supercell(3)
    v = [0.2,0.1,0.15]
    hn = g3.get_hamiltonian(has_spin=True)
    hn.add_valley_exchange(v)
    hs = g3.get_hamiltonian(has_spin=True)
    hs.setup_nambu_spinor()
    hs.add_valley_exchange(v)
    ref = hn.copy() ; ref.setup_nambu_spinor()
    k = [0.07,0.19,0.]
    d = np.array(hs.get_hk_gen()(k)) - np.array(ref.get_hk_gen()(k))
    assert np.max(np.abs(d))<1e-10
    fn,fs = _texture(hn),_texture(hs)
    assert np.max(np.abs(fn))>0.1
    assert np.max(np.abs(fn-fs))<1e-8


def test_inplane_valley_refuses_a_lattice_that_is_not_one_honeycomb():
    """Two sublattices whose bonds do not point in three directions 120
    degrees apart have no K point to take the wavevector from"""
    g = geometry.square_lattice().get_supercell(2)
    g.get_sublattice()
    g.set_finite()
    h = g.get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError,match="honeycomb"):
        h.get_operator("valley_x")
    with pytest.raises(ValueError,match="honeycomb"):
        h.add_valley_exchange([0.1,0.,0.])


@pytest.mark.parametrize("delta",[None,0.1])
def test_tauz_kept_between_calls_is_the_one_of_each_k_point(delta):
    """tau_z keeps its matrix at the last k-point, as the in-plane
    operators do. Going back and forth between k-points, and writing into
    a returned matrix, must give the matrix that a new operator builds at
    that k-point"""
    from pyqula.operatortk.valley import get_valley
    h = geometry.honeycomb_lattice().supercell(3).get_hamiltonian(
            has_spin=False)
    op = get_valley(h,delta=delta)
    ks = [[0.1,0.2,0.],[0.31,-0.12,0.],[0.1,0.2,0.],np.array([0.1,0.2,0.]),
          [0.31,-0.12,0.]]
    v = np.random.default_rng(2).normal(size=h.intra.shape[0])
    for k in ks:
        ref = _dense(get_valley(h,delta=delta)(None,k=k))
        m = op(None,k=k)
        assert np.max(np.abs(_dense(m)-ref))<1e-12
        assert np.max(np.abs(np.asarray(op(v,k=k)).ravel()-ref@v))<1e-12
        m[:,:] = 0. # the caller owns the matrix it gets
    m1 = _dense(op(None,k=ks[0]))
    assert np.max(np.abs(m1-_dense(op(None,k=ks[1]))))>1e-2 # it depends on k


def test_velocity_kept_between_calls_is_the_one_of_each_k_point():
    """The velocity of a 2D Hamiltonian keeps the velocity matrices of the
    last k-point. The speed of several states, going back and forth
    between k-points, must be the one that a new operator gives"""
    h = geometry.honeycomb_lattice().supercell(2).get_hamiltonian()
    h.add_rashba(0.2)
    op = h.get_operator("velocity")
    ks = [[0.1,0.2,0.],[0.31,-0.12,0.],[0.1,0.2,0.],[0.31,-0.12,0.]]
    rng = np.random.default_rng(3)
    for k in ks:
        ref = h.get_operator("velocity")
        for i in range(3):
            w = rng.normal(size=h.intra.shape[0]) \
                    + 1j*rng.normal(size=h.intra.shape[0])
            w = w/np.linalg.norm(w)
            got = np.asarray(op.m(w,k=k)).ravel()
            exp = np.asarray(ref.m(w,k=k)).ravel()
            assert np.max(np.abs(got-exp))<1e-12


def test_vertical_bonds_do_not_change_tauz():
    """tau_z is normalized with the in-plane bond length, which a vertical
    bond between two buckled layers does not have. Counted as a bond of
    in-plane length 0 it made the bond lengths look different, and tau_z
    fell back to the normalization of a flat lattice, with eigenvalues
    +-8/9 on the Dirac states of the (111) diamond film. The tau_z of the
    film must be the one of the same layers pulled apart"""
    g = geometry.buckled_honeycomb_lattice(2).supercell(3)
    go = _pulled_apart(g)
    k = [0.1,0.2,0.]
    (h,ho) = [x.get_hamiltonian(has_spin=False) for x in (g,go)]
    (m,mo) = [_dense(x.get_operator("valley").m(None,k=k)) for x in (h,ho)]
    assert np.max(np.abs(mo))>0.1 # really there
    assert np.max(np.abs(m-mo))<1e-12
