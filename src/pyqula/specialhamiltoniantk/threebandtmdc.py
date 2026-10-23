"""Three-band tight-binding model of the group-VIB transition metal
dichalcogenide monolayers MX2 (M = Mo, W; X = S, Se, Te).

Reference
---------
G.-B. Liu, W.-Y. Shan, Y. Yao, W. Yao and D. Xiao, "Three-band tight-binding
model for monolayers of group-VIB transition metal dichalcogenides",
PRB 88, 085433 (2013), arXiv:1305.6089.

The basis is the d_z2, d_xy and d_x2-y2 orbitals of the metal atom on a
triangular lattice. The model with first-neighbor hoppings ("NN") is Eq.
(HNNk) of the paper with the parameters of its Table II, and the model with
up to third-neighbor hoppings ("TNN") is Eq. (HTNNk) with Table III. Both
come in a GGA and an LDA fit. The spin-orbit coupling is the onsite
lambda L.S term of Eq. (Hsok), lambda/2 L_z s_z, with lambda from Table IV
(fitted to GGA bands; it is used for the LDA parameters as well).

How the model is built
----------------------
The paper gives H(k) in closed form, H(k) = sum_R exp(i k.R) E(R), which is
a finite Fourier series over the lattice vectors R up to the third
neighbors. The real-space hoppings E(R) are recovered exactly from it by a
discrete Fourier transform on a mesh finer than their range, so that
_hk_liu (a transcription of the paper's equations) is the single source of
the model. tests/specialhamiltonian checks the resulting Hamiltonian
against _hk_liu at random k-points and against the closed-form energies at
Gamma, K and M of the paper's Table I.

Each orbital is stored as its own site, all three at the position of the
metal atom, so that every per-site tool of pyqula (onsite energies,
exchange fields, mean-field terms) acts on orbitals, and the dipole between
two orbitals of the same atom is zero, as in the paper. The basis order is
(orbital, spin) with the orbital slowest: d_z2, d_xy, d_x2-y2.

Units: energies in eV, lengths in Angstrom, with the lattice constant a of
the paper's Table II. The nearest-neighbor vector R1 = (a,0,0) points along
x, so K = (4 pi/(3a),0) and x is the zigzag direction of the monolayer.
"""
import numpy as np


# Table II: a (Angstrom), z_XX (Angstrom), eps1, eps2, t0, t1, t2, t11, t12, t22
_nn = {"GGA":{
    "MoS2": [3.190,3.130,1.046,2.104,-0.184,0.401,0.507,0.218,0.338,0.057],
    "WS2":  [3.191,3.144,1.130,2.275,-0.206,0.567,0.536,0.286,0.384,-0.061],
    "MoSe2":[3.326,3.345,0.919,2.065,-0.188,0.317,0.456,0.211,0.290,0.130],
    "WSe2": [3.325,3.363,0.943,2.179,-0.207,0.457,0.486,0.263,0.329,0.034],
    "MoTe2":[3.557,3.620,0.605,1.972,-0.169,0.228,0.390,0.207,0.239,0.252],
    "WTe2": [3.560,3.632,0.606,2.102,-0.175,0.342,0.410,0.233,0.270,0.190]},
  "LDA":{
    "MoS2": [3.129,3.115,1.238,2.366,-0.218,0.444,0.533,0.250,0.360,0.047],
    "WS2":  [3.132,3.126,1.355,2.569,-0.238,0.626,0.557,0.324,0.405,-0.076],
    "MoSe2":[3.254,3.322,1.001,2.239,-0.222,0.350,0.488,0.244,0.314,0.129],
    "WSe2": [3.253,3.338,1.124,2.447,-0.242,0.506,0.514,0.305,0.353,0.025],
    "MoTe2":[3.472,3.598,0.618,2.126,-0.202,0.254,0.423,0.241,0.263,0.269],
    "WTe2": [3.476,3.611,0.623,2.251,-0.209,0.388,0.442,0.272,0.295,0.200]}}

# Table III: eps1, eps2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12,
# u0, u1, u2, u11, u12, u22
_tnn = {"GGA":{
    "MoS2": [0.683,1.707,-0.146,-0.114,0.506,0.085,0.162,0.073,0.060,-0.236,
             0.067,0.016,0.087,-0.038,0.046,0.001,0.266,-0.176,-0.150],
    "WS2":  [0.717,1.916,-0.152,-0.097,0.590,0.047,0.178,0.016,0.069,-0.261,
             0.107,-0.003,0.109,-0.054,0.045,0.002,0.325,-0.206,-0.163],
    "MoSe2":[0.684,1.546,-0.146,-0.130,0.432,0.144,0.117,0.075,0.039,-0.209,
             0.069,0.052,0.060,-0.042,0.036,0.008,0.272,-0.172,-0.150],
    "WSe2": [0.728,1.655,-0.146,-0.124,0.507,0.117,0.127,0.015,0.036,-0.234,
             0.107,0.044,0.075,-0.061,0.032,0.007,0.329,-0.202,-0.164],
    "MoTe2":[0.588,1.303,-0.226,-0.234,0.036,0.400,0.098,0.017,0.003,-0.025,
             -0.169,0.082,0.051,0.057,0.103,0.187,-0.045,-0.141,0.087],
    "WTe2": [0.697,1.380,-0.109,-0.164,0.368,0.204,0.093,0.038,-0.015,-0.209,
             0.107,0.115,0.009,-0.066,0.011,-0.013,0.312,-0.177,-0.132]},
  "LDA":{
    "MoS2": [0.820,1.931,-0.176,-0.101,0.531,0.084,0.169,0.070,0.070,-0.252,
             0.084,0.019,0.093,-0.043,0.047,0.005,0.304,-0.192,-0.162],
    "WS2":  [0.905,2.167,-0.175,-0.090,0.611,0.043,0.181,0.008,0.075,-0.282,
             0.127,0.001,0.114,-0.063,0.047,0.004,0.374,-0.224,-0.177],
    "MoSe2":[0.715,1.687,-0.154,-0.134,0.437,0.124,0.119,0.072,0.048,-0.248,
             0.090,0.066,0.045,-0.067,0.041,0.005,0.327,-0.194,-0.151],
    "WSe2": [0.860,1.892,-0.152,-0.125,0.508,0.094,0.129,0.009,0.044,-0.278,
             0.129,0.059,0.058,-0.090,0.039,0.001,0.392,-0.224,-0.165],
    "MoTe2":[0.574,1.410,-0.148,-0.173,0.333,0.203,0.186,0.127,0.007,-0.280,
             0.067,0.073,0.081,-0.054,0.008,0.037,0.145,-0.078,0.035],
    "WTe2": [0.675,1.489,-0.124,-0.159,0.362,0.196,0.101,0.044,-0.009,-0.250,
             0.129,0.131,-0.007,-0.086,0.012,-0.020,0.361,-0.193,-0.129]}}

# Table IV: the spin-orbit parameter lambda (eV)
_soc = {"MoS2":0.073,"WS2":0.211,"MoSe2":0.091,"WSe2":0.228,
        "MoTe2":0.107,"WTe2":0.237}


def get_material_names(): return sorted(_soc)
def get_model_names(): return ["NN","TNN"]
def get_functional_names(): return sorted(_nn)


def _parameters(material,model,functional):
    """Lattice constant and the parameter dict of one fit"""
    if material not in _soc: raise ValueError("unknown material "
            +repr(material)+", accepted values are "
            +", ".join(get_material_names()))
    if functional not in _nn: raise ValueError("unknown functional "
            +repr(functional)+", accepted values are "
            +", ".join(get_functional_names()))
    nn = _nn[functional][material]
    a = nn[0]
    if model=="NN":
        names = ["e1","e2","t0","t1","t2","t11","t12","t22"]
        p = dict(zip(names,nn[2:]))
        for n in ["r0","r1","r2","r11","r12","u0","u1","u2","u11","u12","u22"]:
            p[n] = 0. # no hoppings beyond the first neighbors
    elif model=="TNN":
        names = ["e1","e2","t0","t1","t2","t11","t12","t22","r0","r1","r2",
                 "r11","r12","u0","u1","u2","u11","u12","u22"]
        p = dict(zip(names,_tnn[functional][material]))
    else: raise ValueError("unknown model "+repr(model)
            +", accepted values are "+", ".join(get_model_names()))
    return a,p


def _hk_liu(k,a,p):
    """The paper's spinless H(k), Eq. (HTNNk), at the Cartesian momentum k
    (the NN model, Eq. (HNNk), is the same with every r and u zero)"""
    al,be = k[0]*a/2.,np.sqrt(3.)*k[1]*a/2.
    c,s = np.cos,np.sin
    sq3 = np.sqrt(3.)
    e1,e2,t0,t1,t2,t11,t12,t22 = [p[n] for n in
            ["e1","e2","t0","t1","t2","t11","t12","t22"]]
    r0,r1,r2,r11,r12 = [p[n] for n in ["r0","r1","r2","r11","r12"]]
    u0,u1,u2,u11,u12,u22 = [p[n] for n in
            ["u0","u1","u2","u11","u12","u22"]]
    V0 = (e1 + 2*t0*(2*c(al)*c(be)+c(2*al))
          + 2*r0*(2*c(3*al)*c(be)+c(2*be))
          + 2*u0*(2*c(2*al)*c(2*be)+c(4*al)))
    ReV1 = (-2*sq3*t2*s(al)*s(be) + 2*(r1+r2)*s(3*al)*s(be)
            - 2*sq3*u2*s(2*al)*s(2*be))
    ImV1 = (2*t1*s(al)*(2*c(al)+c(be)) + 2*(r1-r2)*s(3*al)*c(be)
            + 2*u1*s(2*al)*(2*c(2*al)+c(2*be)))
    ReV2 = (2*t2*(c(2*al)-c(al)*c(be))
            - 2./sq3*(r1+r2)*(c(3*al)*c(be)-c(2*be))
            + 2*u2*(c(4*al)-c(2*al)*c(2*be)))
    ImV2 = (2*sq3*t1*c(al)*s(be)
            + 2./sq3*s(be)*(r1-r2)*(c(3*al)+2*c(be))
            + 2*sq3*u1*c(2*al)*s(2*be))
    V11 = (e2 + (t11+3*t22)*c(al)*c(be) + 2*t11*c(2*al)
           + 4*r11*c(3*al)*c(be) + 2*(r11+sq3*r12)*c(2*be)
           + (u11+3*u22)*c(2*al)*c(2*be) + 2*u11*c(4*al))
    ReV12 = (sq3*(t22-t11)*s(al)*s(be) + 4*r12*s(3*al)*s(be)
             + sq3*(u22-u11)*s(2*al)*s(2*be))
    ImV12 = (4*t12*s(al)*(c(al)-c(be))
             + 4*u12*s(2*al)*(c(2*al)-c(2*be)))
    V22 = (e2 + (3*t11+t22)*c(al)*c(be) + 2*t22*c(2*al)
           + 2*r11*(2*c(3*al)*c(be)+c(2*be))
           + 2./sq3*r12*(4*c(3*al)*c(be)-c(2*be))
           + (3*u11+u22)*c(2*al)*c(2*be) + 2*u22*c(4*al))
    V1 = ReV1 + 1j*ImV1
    V2 = ReV2 + 1j*ImV2
    V12 = ReV12 + 1j*ImV12
    return np.array([[V0,V1,V2],
                     [np.conj(V1),V11,V12],
                     [np.conj(V2),np.conj(V12),V22]])


def _lattice(a):
    """Lattice vectors of the paper, R1 = a(1,0) and R2 = a(1/2,sqrt3/2)
    (pyqula's triangular lattice uses a2 = R2 - R1, the same lattice)"""
    a1 = a*np.array([1.,0.,0.])
    a2 = a*np.array([-0.5,np.sqrt(3.)/2.,0.])
    return a1,a2


def _real_space_hoppings(a,p,nmax=2,nmesh=8):
    """Hoppings E(R), R = n1 a1 + n2 a2, recovered from the paper's H(k) by
    a discrete Fourier transform. With the convention H(k) = sum_R
    exp(i k.R) E(R), the inverse is E(R) = (1/N) sum_k exp(-i k.R) H(k)
    over an N = nmesh^2 mesh, exact as long as nmesh > 2*nmax, since every
    hopping of the TNN model has |n1|,|n2| <= nmax = 2"""
    a1,a2 = _lattice(a)
    A = np.array([a1[:2],a2[:2]])
    B = 2.*np.pi*np.linalg.inv(A).T # rows are the reciprocal vectors
    fr = np.arange(nmesh)/nmesh
    hks = {}
    for k1 in fr:
        for k2 in fr:
            hks[(k1,k2)] = _hk_liu(k1*B[0]+k2*B[1],a,p)
    out = {}
    for n1 in range(-nmax,nmax+1):
        for n2 in range(-nmax,nmax+1):
            m = sum(np.exp(-2j*np.pi*(k1*n1+k2*n2))*hk
                    for ((k1,k2),hk) in hks.items())/nmesh**2
            if np.max(np.abs(m))>1e-10: out[(n1,n2,0)] = m
    return out


def three_band_tmdc(material="MoS2",model="TNN",functional="GGA",soc=True,
        fermi="midgap"):
    """Three-band tight-binding Hamiltonian of a MX2 monolayer, see the
    module docstring and specialhamiltonian.three_band_tmdc"""
    from .. import geometry
    from ..multihopping import MultiHopping
    a,p = _parameters(material,model,functional)
    g = geometry.triangular_lattice()
    g.x = np.zeros(3) ; g.y = np.zeros(3) ; g.z = np.zeros(3)
    g.xyz2r() # three orbitals on the metal atom
    g.a1,g.a2 = _lattice(a)
    g.a3 = np.array([0.,0.,20.])
    g.has_sublattice = False
    g.update_reciprocal()
    g.get_fractional()
    h = g.get_hamiltonian(has_spin=False,is_multicell=True,
            tij=lambda r1,r2: 0.)
    h.set_multihopping(MultiHopping(_real_space_hoppings(a,p)))
    if soc is not False and soc is not None: # spinful, with lambda L.S
        lamb = _soc[material] if soc is True else float(soc)
        h.turn_spinful()
        Lz = np.array([[0.,0.,0.],[0.,0.,2j],[0.,-2j,0.]])
        sz = np.diag([1.,-1.])
        h.intra = h.intra + lamb/2.*np.kron(Lz,sz) # basis (orbital,spin)
    if fermi=="midgap": h.shift_fermi(-_midgap(h))
    elif fermi is not None: h.shift_fermi(-float(fermi))
    return h


def _midgap(h,nk=36):
    """Middle of the gap between the lowest band (of each spin) and the
    rest, the Fermi level of the undoped MX2 monolayer (two electrons in
    the three d orbitals)"""
    from ..klist import kmesh
    hkgen = h.get_hk_gen()
    nocc = h.intra.shape[0]//3 # the lowest band, one per spin
    ks = list(kmesh(2,nk=nk)) + [[1./3.,1./3.,0.],[2./3.,2./3.,0.]] # the
                                     # mesh and the two valleys -K and K
    es = np.array([np.linalg.eigvalsh(np.asarray(
        hkgen(k).todense() if hasattr(hkgen(k),"todense") else hkgen(k)))
        for k in ks])
    vbm,cbm = np.max(es[:,:nocc]),np.min(es[:,nocc:])
    return (vbm+cbm)/2.


_stackings = ["H","R"] # stacking of one layer on the next, see multilayer


def three_band_tmdc_multilayer(layers,t_perp,stacking="H",model="TNN",
        functional="GGA",soc=True,onsite=None,vdw_gap=3.0,fermi="midgap"):
    """Multilayer of three-band MX2 monolayers, see
    specialhamiltonian.three_band_tmdc_multilayer for the documentation.

    Layer l has its metal atoms at the lateral position c_l and height z_l,
    and an orientation s_l = +-1. An "H" interface puts the next layer at
    c_l + delta with the opposite orientation, so the pair is the
    inversion image of itself through the midpoint (the 2H stacking, with
    the metal atoms of one layer above the chalcogens of the other); an
    "R" interface puts it at c_l + delta with the same orientation (the R,
    or 3R, stacking). delta = (a1 + 2 a2')/3 with a2' = R2 the second
    lattice vector of the paper, |delta| = a/sqrt3. Since the d orbitals are
    even under inversion, a layer with s = -1 has the hoppings E(-R) of the
    paper, and its spin-orbit term, lambda/2 Lz sz, is unchanged.

    The interlayer hopping t_perp couples the d_z2 orbitals of the three
    nearest metal atoms of two neighboring layers, at lateral distance
    a/sqrt3. Their phases add to zero at K and K', so the valleys are not
    coupled by it, while the d_z2 valence band at Gamma of a homobilayer is
    split by exactly 6|t_perp|."""
    from .. import geometry
    from ..multihopping import MultiHopping
    if isinstance(stacking,str): stacking = [stacking]*(len(layers)-1)
    if len(stacking)!=len(layers)-1: raise ValueError("stacking needs one "
            "value per interface, "+str(len(layers)-1)+" here, got "
            +str(len(stacking)))
    for s in stacking:
        if s not in _stackings: raise ValueError("unknown stacking "
                +repr(s)+", accepted values are "+", ".join(_stackings))
    if np.isscalar(t_perp): t_perp = [t_perp]*(len(layers)-1)
    if len(t_perp)!=len(layers)-1: raise ValueError("t_perp needs one value "
            "per interface, "+str(len(layers)-1)+" here, got "+str(len(t_perp)))
    if onsite is None: onsite = [0.]*len(layers)
    if len(onsite)!=len(layers): raise ValueError("onsite needs one energy "
            "per layer, "+str(len(layers))+" here, got "+str(len(onsite)))
    pars = [_parameters(m,model,functional) for m in layers]
    a = np.mean([ap[0] for ap in pars]) # common lattice constant
    a1,a2 = _lattice(a)
    R2 = a1 + a2 # the paper's second lattice vector
    delta = (a1 + R2)/3. # lateral shift between stacked layers
    # orientation, lateral position and height of each layer
    orient,c,z = [1],[np.zeros(3)],[0.]
    for (l,s) in enumerate(stacking):
        orient.append(-orient[-1] if s=="H" else orient[-1])
        c.append(c[-1] + delta)
        zxx = (_nn[functional][layers[l]][1] + _nn[functional][layers[l+1]][1])/2.
        z.append(z[-1] + zxx + vdw_gap) # metal-metal distance
    nl = len(layers)
    g = geometry.triangular_lattice()
    r = np.array([c[l] + np.array([0.,0.,z[l]]) for l in range(nl)
                  for i in range(3)]) # three orbitals per metal atom
    r[:,2] = r[:,2] - np.mean(z) # center the stack at z = 0
    g.x,g.y,g.z = r[:,0],r[:,1],r[:,2]
    g.xyz2r()
    g.a1,g.a2 = a1,a2
    g.a3 = np.array([0.,0.,np.max(r[:,2])-np.min(r[:,2])+20.])
    g.has_sublattice = False
    g.update_reciprocal()
    g.get_fractional()
    # intralayer hoppings, block diagonal
    hop = {}
    def block(key):
        if key not in hop: hop[key] = np.zeros((3*nl,3*nl),dtype=np.complex128)
        return hop[key]
    for l in range(nl):
        el = _real_space_hoppings(pars[l][0],pars[l][1])
        for (key,m) in el.items():
            k2 = key if orient[l]==1 else (-key[0],-key[1],0)
            b = block(k2)
            b[3*l:3*l+3,3*l:3*l+3] += m
        block((0,0,0))[3*l:3*l+3,3*l:3*l+3] += onsite[l]*np.eye(3)
    # interlayer d_z2 hoppings to the three nearest metal atoms
    for l in range(nl-1):
        i,j = 3*l,3*(l+1) # the d_z2 orbitals of the two layers
        for n1 in range(-2,3):
            for n2 in range(-2,3):
                d = c[l+1] + n1*a1 + n2*a2 - c[l] # lateral bond, cell (n1,n2)
                if abs(np.linalg.norm(d)-a/np.sqrt(3.))>1e-6: continue
                block((n1,n2,0))[i,j] += t_perp[l]
                block((-n1,-n2,0))[j,i] += np.conj(t_perp[l])
    h = g.get_hamiltonian(has_spin=False,is_multicell=True,
            tij=lambda r1,r2: 0.)
    h.set_multihopping(MultiHopping(hop))
    if soc is not False and soc is not None:
        lambs = [_soc[m] if soc is True else float(soc) for m in layers]
        h.turn_spinful()
        Lz = np.array([[0.,0.,0.],[0.,0.,2j],[0.,-2j,0.]])
        L = np.zeros((3*nl,3*nl),dtype=np.complex128)
        for l in range(nl): L[3*l:3*l+3,3*l:3*l+3] = lambs[l]/2.*Lz
        h.intra = h.intra + np.kron(L,np.diag([1.,-1.]))
    if fermi=="midgap": h.shift_fermi(-_midgap(h))
    elif fermi is not None: h.shift_fermi(-float(fermi))
    return h
