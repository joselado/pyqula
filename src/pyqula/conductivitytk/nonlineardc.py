"""Second- and third-order dc conductivity of a metal, decomposed in powers
of the relaxation time: at second order the nonlinear Drude term (tau^2),
the Berry curvature dipole (tau^1) and the quantum-metric terms (tau^0); at
third order the nonlinear Drude term (tau^3) and the Berry connection
polarizability term (tau^1).

The derivations, the sign chain and the comparison of the conventions in
use are in future_development/nonlinear_response.md, section "The
second-order dc conductivity in powers of tau"; this module holds the
machinery and the summary below.

Conventions
-----------
hbar = 1 and the carriers are electrons, q = -1, so that e^3 = 1 in the
formulas of the papers (written with e > 0). The chemical potential is
zero (h.shift_fermi(-mu) puts it at mu). The current is
J_a = sigma_abc E_b E_c with sigma symmetric in bc, the first index the
current, and sigma is per unit cell volume (area in 2D, length in 1D):
<X> = (1/(N_k V_cell)) sum_k sum_n X_n, as in kubo.py.

Building blocks
---------------
Everything comes from the band energies and the eigenbasis matrix elements
of the first and second Cartesian derivatives of the Peierls-coupled Bloch
Hamiltonian, h^a (the velocity, atomic gauge) and h^ab, from
shg._eigenbasis_derivatives. No k-derivative of a geometric quantity is
taken. With eps_nm = eps_n - eps_m and P(n) the degenerate multiplet of n,
for m not in P(n):

  Q^nm_ab    = h^a_nm h^b_mn/eps_nm^2         pair quantum geometric tensor
  G^n_ab     = sum_m Re Q^nm_ab                quantum metric
  Omega^n_ab = -2 sum_m Im Q^nm_ab             Berry curvature
  Gt^n_ab    = 2 sum_m Re Q^nm_ab/eps_nm       band-normalized metric, the
                                               Berry connection polarizability
  u^n_a      = h^a_nn                          band velocity
  u^n_ab     = h^ab_nn + 2 sum_m Re(h^a_nm h^b_mn)/eps_nm   inverse mass
  w_n        = -f'(eps_n) = f(1-f)/T           Fermi-surface weight

and the five terms are

  sigma^ND_abc    = -tau^2 < w u_a u_bc >
  sigma^BCD_abc   = (tau/2) < w (u_c Omega_ab + u_b Omega_ac) >
  sigma^hall_abc  = - < w [u_a Gt_bc - (1/2)(u_b Gt_ca + u_c Gt_ab)] >
  sigma^disp_abc  = - < w u_a Gt_bc >
  sigma^intra_abc = -(1/2) < w d_a G_bc > = -(1/2) < f'' u_a G_bc >

with f'' = f(1-f)(1-2f)/T^2. The nonlinear Drude term is the Fermi-sea
form -tau^2 <f d_a d_b d_c eps> integrated by parts once (Shibata,
arXiv:2606.22359, Eq. (ND); Kaplan, Holder and Yan, arXiv:2211.17213),
the same number as nonlineardrude._sigma at l = 2. The Berry curvature
dipole term is Sodemann and Fu (arXiv:1508.00571) with their dc factor 2;
in 2D sigma_xyy = tau D_y and sigma_yxx = -tau D_x, with the dipole
D_a = <w u_a Omega_xy> returned by berry_curvature_dipole. The intra term
is used in its f'' form, one more integration by parts on the mesh, so
that no derivative of the metric is needed; the covariant-derivative form
_intra_covariant is the check of it.

The quantum-metric (tau^0) term has three forms in use, which differ by
gauge-invariant pieces, so the pieces are exposed and qmd= names the
convention:

  "shibata" (default)  hall + intra  Shibata, Eqs. (intraQMD-band) and
                                     (interQMD-band), the clean limit of the
                                     velocity-gauge Keldysh calculation
  "gao"                hall          Gao, Yang and Niu, arXiv:1402.2538;
                                     Wang et al., arXiv:2106.12695
  "kaplan"             hall + disp   Kaplan, Holder and Yan, Eq.
                                     (fullcurrent2-metric), as written

The hall piece is purely transverse, sigma^hall_aaa = 0. The disp piece
is Kaplan-Holder-Yan's dispersion current with their literal coefficient.
The second-order Stark shift of a band is (q^2/2) E.Gt.E, half of their
Eq. (energy_renorm), so with an unshifted f0 the physical dispersion current
is half of sigma^disp, and with the shifted f0 it cancels as a total
derivative. It is kept as the paper writes it, with this factor recorded
here.

Degenerate multiplets
---------------------
Every sum over m leaves out the multiplet P(n), so that the sums over a
multiplet of Omega, G and Gt are its non-Abelian traces. A product of two
band-diagonal quantities is not invariant under a rotation of the basis
inside the multiplet, so it is replaced by the block trace,
sum_{n in P} w u^n_a X^n_bc -> w Tr_P[U_a X_bc], with U_a = P h^a P and
X_bc the block matrix whose diagonal is u_bc, G_bc, Gt_bc or Omega_bc (the
same sums over m not in P, between two states of P). For a nondegenerate
spectrum this is the formula above, and for an exact spin degeneracy
without spin-orbit coupling it is the sum of the two spin copies. In
general it is a regularization, tested by rotating the basis inside every
multiplet by a random unitary, not a non-Abelian derivation.

Third order
-----------
J_a = sigma_abcd E_b E_c E_d with sigma symmetric in bcd. Two terms are
built, the tau^3 and the tau^1 one; the Berry curvature quadrupole (tau^2)
and the tau^0 term are not (see future_development/nonlinear_response.md).

  sigma^ND3_abcd = tau^3 < f d_a d_b d_c d_d eps >

is the prefactor chain of nonlineardrude._sigma, (-1)^(l+1) tau^l = +tau^3
at l = 3, with the fourth band derivative from the Rayleigh-Schroedinger
recursion of nonlineardrude._Channel on the whole Hamiltonian, so it holds
for any multiband model in the charge channel; in the Kato block form the
recursion gives the multiplet trace exactly. The recursion runs over the
three Cartesian indices, so this term is built in 1D, 2D and 3D.

The tau^1 term is the Berry connection polarizability (BCP) term of Liu et
al., arXiv:2106.04931 (Lai et al., Nat. Nanotechnol. 16, 869 (2021)). Their
BCP G_ab = 2 Re sum_m A^a_nm A^b_mn/(eps_n - eps_m) is Gt above, and their
units (e = hbar = 1, k' = -E, j = -int r' f) are ours. In the semiclassics
of Gao, Yang and Niu (arXiv:1402.2538) the field shifts the wave-packet
center by A^(1)_a = Gt_ab E_b, which gives the field-induced curvature
Omega^(1)_ab = d_a A^(1)_b - d_b A^(1)_a, and corrects the band energy by
eps^(2) = -(1/2) E.Gt.E. Note the sign: the Stark shift of the state is
+(1/2) E.Gt.E (the static polarizability check of the second order), and
the semiclassical band energy leaves out the potential energy E.A^(1) =
E.Gt.E of the shifted center. With f = sum_p (tau E.d)^p f0(eps~), the
terms linear in tau are their Eq. (j3),

  j = -tau int [ v (E.d)(eps^(2) f0') + d eps^(2) (E.d) f0
                 + E x Omega^(1) (E.d) f0 ],

and after the symmetrization over bcd two of the three Fermi-sea pieces of
their Eq. (chiI) cancel, leaving (with Gt = G)

  sigma^BCP_abcd = tau Sym_bcd [ -< f d_b d_c Gt_ad >
                                 + (1/2) < f'' u_a u_b Gt_cd > ]
                 = tau Sym_bcd [ -< f'' u_b u_c Gt_ad > + < w u_bc Gt_ad >
                                 + (1/2) < f'' u_a u_b Gt_cd > ],

the second line after two integrations by parts on the mesh, which is the
form computed here, from es, h1 and h2 only. At tau^0 the same semiclassics
gives the second-order qmd="gao" term. The relaxation goes to f0(eps~),
the choice of Liu et al., which is what gives the first term of Eq. (j3);
relaxing to f0(eps) instead would drop it. Products of three
band-diagonal quantities in a multiplet become the block trace
Tr_P[(1/2){U_x,U_y} Gt_zt], with the bcd symmetrization on top: basis
independent, and the sum of the copies for an exact spin degeneracy.

The third-order recursion of nonlinearkernel does not check the BCP term
through a Laurent coefficient: a one-pole term carries 1/(p delta) with
p = 1..3 at third order, and the coefficients of 1/delta mix several tau
structures. Only A_-3 = sigma^ND3/(6 tau^3) is asserted there.
"""
import numpy as np

from .. import parallel
from ..dostk.simplexdos import uniform_mesh
from . import kubo
from . import shg
from . import nonlineardrude


def _drude(t): return -t["uu"]

def _bcd(t): # (1/2)(<w u_c Omega_ab> + <w u_b Omega_ac>), t["uOmega"] is [c,a,b]
    return 0.5*(np.transpose(t["uOmega"],(1,2,0))
               +np.transpose(t["uOmega"],(1,0,2)))

def _qmd_hall(t): # t["uGt"][a,b,c] = <w u_a Gt_bc>
    g = t["uGt"]
    return -(g - 0.5*(np.transpose(g,(2,0,1)) + np.transpose(g,(1,2,0))))

def _qmd_disp(t): return -t["uGt"]

def _qmd_intra(t): return -0.5*t["uG2"]


# term= registry: the function giving the term without its power of tau,
# and that power
_terms = {
    "drude": (_drude,2),
    "bcd": (_bcd,1),
    "qmd_hall": (_qmd_hall,0),
    "qmd_disp": (_qmd_disp,0),
    "qmd_intra": (_qmd_intra,0),
    }

# qmd= registry: the quantum-metric pieces of each convention
_qmd = {
    "shibata": ("qmd_hall","qmd_intra"),
    "gao": ("qmd_hall",),
    "kaplan": ("qmd_hall","qmd_disp"),
    }

_channels = ("charge","spin","up","dn") # channel=, see nonlineardrude._combine


def _bcp(t):
    """The tau^1 term at third order, sigma[a,b,c,d] symmetric in bcd, from
    t["M"][x,y,z,s] = <f'' Tr[(1/2){U_x,U_y} Gt_zs]> and
    t["N"][x,y,z,s] = <w Tr[u_xy Gt_zs]>"""
    M,N = t["M"],t["N"]
    # -<f'' u_b u_c Gt_ad> + <w u_bc Gt_ad> + (1/2)<f'' u_a u_b Gt_cd>
    s = (-np.transpose(M,(2,0,1,3)) + np.transpose(N,(2,0,1,3))
         + 0.5*M)
    return _symmetrize_fields(s)


def _drude3(t): return t["dddd"]


# term= registry at third order, as _terms
_terms3 = {
    "drude": (_drude3,3),
    "bcp": (_bcp,1),
    }

_orders = (2,3) # order=


def get_term_names(order=2):
    """Accepted values of term= in nonlinear_dc_conductivity at order="""
    if order==3: return list(_terms3)+["all"]
    return ["drude","bcd","qmd"]+[t for t in _terms if t.startswith("qmd_")]\
            +["all"]


def get_orders():
    """Accepted values of order= in nonlinear_dc_conductivity"""
    return list(_orders)


def _symmetrize_fields(s):
    """Average of a (3,3,3,3) tensor over the permutations of its last three
    (field) indices"""
    perms = [(0,1,2,3),(0,1,3,2),(0,2,1,3),(0,2,3,1),(0,3,1,2),(0,3,2,1)]
    return sum(np.transpose(s,p) for p in perms)/6.


def get_qmd_conventions():
    """Accepted values of qmd= in nonlinear_dc_conductivity"""
    return list(_qmd)


def get_channel_names():
    """Accepted values of channel= in nonlinear_dc_conductivity"""
    return list(_channels)


def _multiplets(es,tol):
    """same[k,n,m]: True when bands n and m are in the same degenerate
    multiplet at k (consecutive energies closer than tol, chained, the
    grouping of topologytk/anomaloushall.py), the diagonal included"""
    gap = np.diff(es,axis=1)>=tol
    lab = np.concatenate([np.zeros((es.shape[0],1),dtype=np.int64),
                          np.cumsum(gap,axis=1)],axis=1)
    return lab[:,:,None]==lab[:,None,:]


def _band_blocks(es,h1,h2,tol):
    """The block matrices of the module docstring on a batch of k-points,
    from the output of shg._eigenbasis_derivatives. Returns a dict with the
    multiplet mask "same" (nk,n,n), the velocity "u" (nk,3,n,n), and
    "ubc", "G", "Gt" and "Omega", each (nk,3,3,n,n); all of them are kept
    only between two states of the same multiplet, and their diagonals are
    the band quantities u^n_a, u^n_bc, G^n_bc, Gt^n_bc and Omega^n_bc."""
    same = _multiplets(es,tol)
    de = es[:,:,None] - es[:,None,:] # eps_nm
    inv = np.where(same,0.,1./np.where(same,1.,de)) # 1/eps_nm, m not in P(n)
    def Y(p): # Y[k,b,c,n,m'] = sum_m h^b_nm h^c_mm'/eps_nm^p
        return (h1[:,:,None]*(inv**p)[:,None,None])@h1[:,None,:]
    def sym(y): return y + np.swapaxes(y,1,2) # (b,c) + (c,b)
    y1,y2,y3 = Y(1),Y(2),Y(3)
    mask = same[:,None,None]
    out = dict(same=same,u=np.where(same[:,None],h1,0.))
    out["ubc"] = np.where(mask,h2 + sym(y1),0.)
    out["G"] = np.where(mask,sym(y2)/2.,0.)
    out["Gt"] = np.where(mask,sym(y3),0.)
    out["Omega"] = np.where(mask,1j*(y2 - np.swapaxes(y2,1,2)),0.)
    return out


def _weights(es,T):
    """Fermi-surface weight w = -f' and the second derivative f''"""
    f = kubo._fermi(es,T)
    return f*(1.-f)/T,f*(1.-f)*(1.-2.*f)/T**2


def _batch_tensors(es,h1,h2,T,tol):
    """The k-sums (not normalized) of the block traces that the five terms
    are made of, on one batch of k-points, each a (3,3,3) array:
    uu[a,b,c] = sum w Tr[U_a u_bc], uGt[a,b,c] = sum w Tr[U_a Gt_bc],
    uG2[a,b,c] = sum f'' Tr[U_a G_bc] and uOmega[c,a,b] = sum w
    Tr[U_c Omega_ab]. The weight is common to a multiplet, so taking it
    on either index of the block is the same"""
    w,f2 = _weights(es,T)
    b = _band_blocks(es,h1,h2,tol)
    def tr(wt,X): # sum_k sum_{n,m in P} wt_n U_a[m,n] X_bc[n,m]
        return np.real(np.einsum("kn,kamn,kbcnm->abc",wt,b["u"],X,
                optimize=True))
    return dict(uu=tr(w,b["ubc"]),uGt=tr(w,b["Gt"]),uG2=tr(f2,b["G"]),
                uOmega=tr(w,b["Omega"]))


def _batch_tensors3(es,h1,h2,T,tol):
    """The k-sums (not normalized) of the block traces of the third-order
    BCP term on one batch of k-points, each a (3,3,3,3) array:
    M[x,y,z,s] = sum f'' Tr[(1/2)(U_x U_y + U_y U_x) Gt_zs] and
    N[x,y,z,s] = sum w Tr[u_xy Gt_zs]. The blocks live inside a multiplet,
    where the weights are common, so the matrix products stay inside it"""
    w,f2 = _weights(es,T)
    b = _band_blocks(es,h1,h2,tol)
    U = b["u"]
    UU = U[:,:,None]@U[:,None,:] # U_x U_y, (nk,3,3,n,n)
    UU = 0.5*(UU + np.swapaxes(UU,1,2))
    def tr(wt,X,Y): # sum_k sum_n wt_n (X_xy Y_zs)_nn
        return np.real(np.einsum("kn,kxynm,kzsmn->xyzs",wt,X,Y,
                optimize=True))
    return dict(M=tr(f2,UU,b["Gt"]),N=tr(w,b["ubc"],b["Gt"]))


def _batch_size(n):
    """k-points per batch: the vertices of shg._eigenbasis_derivatives and
    the blocks above hold about a hundred (n,n) complex matrices per
    k-point"""
    return int(max(64,min(8192,2e8/(2000.*n*n))))


def _mesh_tensors(h,nk,T,degeneracy_tol,kernel=_batch_tensors):
    """The normalized mesh sums of kernel over the uniform nk^dim mesh, for
    one Hamiltonian (one spin channel, or the charge channel)"""
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    ks = uniform_mesh(h.dimensionality,nk)
    n = tms.shape[1]
    nb = _batch_size(n)
    parallel.set_num_threads() # threads of the numba diagonalization
    acc = None
    for i0 in range(0,len(ks),nb): # loop over batches of k-points
        es,h1,h2,h3 = shg._eigenbasis_derivatives(tms,dirs,d,hkgen,
                ks[i0:i0+nb])
        t = kernel(es,h1,h2,T,degeneracy_tol*scale)
        if acc is None: acc = t
        else: acc = {key:acc[key]+t[key] for key in acc}
    return {key:v/(len(ks)*cellvol) for (key,v) in acc.items()}


def _check(h,T,channel,order=2):
    """The guards shared by the public routines"""
    if h.has_eh: raise NotImplementedError("the nonlinear dc conductivity "
            "is not implemented for Nambu (superconducting) Hamiltonians: "
            "the charge current operator there is not i[H,r]")
    if h.dimensionality not in (1,2,3): raise ValueError("the nonlinear "
            "dc conductivity needs a periodic Hamiltonian (dimensionality "
            "1, 2 or 3), got dimensionality "+str(h.dimensionality))
    if order not in _orders: raise NotImplementedError("the nonlinear dc "
            "conductivity in powers of tau is implemented at "
            +" and ".join("order="+str(o) for o in _orders)+", got order="
            +str(order))
    if not T>0.: raise ValueError("the nonlinear dc conductivity is a "
            "Fermi-surface sum, whose weight -df/dE is zero on a k-mesh at "
            "T=0: it needs T>0, got T="+str(T))
    if channel not in _channels: raise ValueError("unknown channel "
            +repr(channel)+" of the nonlinear dc conductivity, accepted "
            "values are "+", ".join(get_channel_names()))
    if order==3 and channel!="charge": raise NotImplementedError("the "
            "third-order dc conductivity is implemented in the charge "
            "channel only, got channel="+repr(channel))


def _channel_tensors(h,nk,T,channel,degeneracy_tol,kernel=_batch_tensors):
    """Mesh tensors in the requested channel. The charge channel works on
    the whole Hamiltonian, spin-orbit coupling included; the spin channels
    need spin to be conserved, and run on the two spin blocks of
    nonlineardrude._spin_channels, combined as nonlineardrude._combine"""
    if channel=="charge":
        return _mesh_tensors(h,nk,T,degeneracy_tol,kernel=kernel)
    out = {s:_mesh_tensors(hs,nk,T,degeneracy_tol,kernel=kernel)
           for (s,hs) in nonlineardrude._spin_channels(h)}
    return {key:nonlineardrude._combine({s:out[s][key] for s in out},
            channel) for key in out[1]}


def _default_labels(h):
    """Cartesian labels of the periodic directions"""
    return "xyz"[:h.dimensionality]


def _components(h,components,order=2):
    """(list of labels, list of index tuples, single) for components=,
    a string such as "xyy" ("xxyy" at third order), a list of them, or None
    for every component along the periodic directions"""
    single = isinstance(components,str)
    if components is None:
        lab = _default_labels(h)
        components = [""]
        for i in range(order+1):
            components = [c+a for c in components for a in lab]
    if single: components = [components]
    example = {2:"'xyy'",3:"'xxyy'"}[order]
    name = {2:"second",3:"third"}[order]
    trip = []
    for comp in components:
        if not isinstance(comp,str) or len(comp)!=order+1: raise ValueError(
            "a component of the "+name+"-order conductivity is "
            +str(order+1)+" Cartesian labels such as "+example+", got "
            +repr(comp))
        trip.append(tuple(kubo._index(s) for s in comp))
    return components,trip,single


def _pick(tensor,components,trip,single):
    """The requested components of a (3,)*(order+1) tensor"""
    if single: return tensor[trip[0]]
    return {c:tensor[t] for (c,t) in zip(components,trip)}


def _term_tensor(t,term,tau,qmd):
    """The (3,3,3) tensor of one term, with its power of tau"""
    if term=="qmd": return sum(_term_tensor(t,p,tau,qmd) for p in _qmd[qmd])
    fun,power = _terms[term]
    return fun(t)*tau**power


def _drude3_tensor(h,nk,T,degeneracy_tol):
    """tau^3-free third-order Drude tensor <f d_a d_b d_c d_d eps>, a
    (3,3,3,3) array, from the band derivatives of nonlineardrude._Channel on
    the whole Hamiltonian (charge channel). The fourth derivative is
    symmetric in its four indices, so each multiset of indices is computed
    once"""
    ks = uniform_mesh(h.dimensionality,nk)
    chan = nonlineardrude._Channel(h.get_multicell(),ks,
            degeneracy_tol=degeneracy_tol)
    f = kubo._fermi(chan.es,T)
    norm = 1./(len(ks)*kubo._cell_factors(h)[2])
    chan.prepare(4)
    out = np.zeros((3,3,3,3))
    done = dict()
    for idx in np.ndindex(3,3,3,3):
        key = tuple(sorted(idx))
        if key not in done:
            if any(i>=h.dimensionality for i in key): # not periodic
                done[key] = 0.
            else: done[key] = np.sum(f*chan.derivative(list(key)))*norm
        out[idx] = done[key]
    return out


def _order3(h,term,tau,nk,T,components,degeneracy_tol):
    """nonlinear_dc_conductivity at order=3"""
    if term not in get_term_names(3): raise ValueError("unknown term "
            +repr(term)+" of the third-order dc conductivity, accepted "
            "values are "+", ".join(get_term_names(3)))
    components,trip,single = _components(h,components,order=3)
    t = dict()
    if term in ("drude","all"):
        t["dddd"] = _drude3_tensor(h,nk,T,degeneracy_tol)
    if term in ("bcp","all"):
        t.update(_mesh_tensors(h,nk,T,degeneracy_tol,kernel=_batch_tensors3))
    def tensor(p):
        fun,power = _terms3[p]
        return fun(t)*tau**power
    if term!="all": return _pick(tensor(term),components,trip,single)
    out = {p:tensor(p) for p in _terms3}
    out["total"] = out["drude"] + out["bcp"]
    return {p:_pick(v,components,trip,single) for (p,v) in out.items()}


def nonlinear_dc_conductivity(h,order=2,term="all",tau=1.,qmd="shibata",
        channel="charge",nk=100,T=0.05,components=None,degeneracy_tol=1e-6):
    """Second-order dc conductivity sigma_abc, J_a = sigma_abc E_b E_c,
    decomposed in powers of the relaxation time tau, see the module
    docstring for the formulas. At order=3, the third-order
    sigma_abcd, J_a = sigma_abcd E_b E_c E_d, with term "drude" (tau^3),
    "bcp" (tau^1, the Berry connection polarizability term) or
    "all" (a dict with "drude", "bcp" and "total"), components of four
    labels such as "xxyy", the charge channel only, and qmd= unused.

    term is "drude" (tau^2), "bcd" (tau^1), "qmd" (tau^0, in the
    convention qmd=), one of the quantum-metric pieces "qmd_hall",
    "qmd_disp" and "qmd_intra", or "all", a dict with "drude", "bcd",
    "qmd" and their sum "total", each with its power of tau included.
    components is a label such as "xyy" (the value is returned), a list of
    them, or None for every component along the periodic directions (a
    dict keyed by the labels is returned). channel is "charge" for any
    Hamiltonian, or "spin", "up", "dn" for a spin-conserving one. T>0 and a
    mesh whose energy spacing is below T are needed; convergence is
    checked by doubling nk at fixed T."""
    _check(h,T,channel,order=order)
    if order==3: return _order3(h,term,tau,nk,T,components,degeneracy_tol)
    if term not in get_term_names(): raise ValueError("unknown term "
            +repr(term)+" of the nonlinear dc conductivity, accepted values "
            "are "+", ".join(get_term_names()))
    if qmd not in _qmd: raise ValueError("unknown quantum-metric "
            "convention qmd="+repr(qmd)+", accepted values are "
            +", ".join(get_qmd_conventions()))
    components,trip,single = _components(h,components)
    t = _channel_tensors(h,nk,T,channel,degeneracy_tol)
    if term!="all":
        return _pick(_term_tensor(t,term,tau,qmd),components,trip,single)
    out = {p:_term_tensor(t,p,tau,qmd) for p in ["drude","bcd","qmd"]}
    out["total"] = out["drude"] + out["bcd"] + out["qmd"]
    return {p:_pick(v,components,trip,single) for (p,v) in out.items()}


def berry_curvature_dipole(h,nk=100,T=0.05,channel="charge",
        degeneracy_tol=1e-6):
    """Berry curvature dipole D_a = <w u_a Omega_xy> = <f d_a Omega_xy>
    in 2D, an array [D_x,D_y], such that sigma^BCD_xyy = tau D_y and
    sigma^BCD_yxx = -tau D_x. In 3D the tensor D[a,b] = <w u_a Omega_b>,
    with the pseudovector Omega_x = Omega_yz, Omega_y = Omega_zx and
    Omega_z = Omega_xy."""
    _check(h,T,channel)
    if h.dimensionality==1: raise ValueError("the Berry curvature dipole "
            "needs a 2D or 3D periodic Hamiltonian, a 1D one has no Berry "
            "curvature")
    t = _channel_tensors(h,nk,T,channel,degeneracy_tol)["uOmega"]
    if h.dimensionality==2: return np.array([t[0,0,1],t[1,0,1]])
    planes = [(1,2),(2,0),(0,1)] # Omega_x, Omega_y, Omega_z
    return np.array([[t[a,p,q] for (p,q) in planes] for a in range(3)])


def _intra_covariant_batch(es,h1,h2,T,tol):
    """The intra term in its covariant-derivative form, -(1/2) sum_k
    sum_n w_n d_a G^n_bc, with

      d_a G^n_bc = Re sum_{m != n} [ r^b_nm;a r^c_mn + r^b_nm r^c_mn;a ]

    and the generalized derivative r^c_nm;b of shg._sigma2_length (from
    the identity [D^b,h^c] = h^bc of the tight-binding model). An
    independent code path for the f'' form of _batch_tensors, which it
    must equal up to the aliasing of the mesh. Every pair of bands has to
    be nondegenerate, since r and r;b divide by eps_nm."""
    nk,n = es.shape
    de = es[:,:,None] - es[:,None,:] # eps_nm
    off = ~np.eye(n,dtype=bool)
    if np.min(np.abs(de[:,off]))<tol: raise ValueError("the covariant "
            "form of the intra term needs a nondegenerate spectrum at every "
            "k-point")
    inv = np.where(off,1./np.where(off,de,1.),0.)
    r = h1*inv[:,None]/1j # r^a_nm, (nk,3,n,n)
    u = np.real(np.einsum("kann->kan",h1))
    Delta = u[:,:,:,None] - u[:,:,None,:] # u^b_n - u^b_m
    def rgd(c,b): # r^c_nm;b
        gd = h2[:,b,c] + 1j*(r[:,b]@h1[:,c] - h1[:,c]@r[:,b])
        return (gd - 1j*Delta[:,b]*r[:,c])*inv/1j
    w = _weights(es,T)[0]
    out = np.zeros((3,3,3))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                dG = np.real(np.einsum("knm,kmn->kn",rgd(b,a),r[:,c])
                            +np.einsum("knm,kmn->kn",r[:,b],rgd(c,a)))
                out[a,b,c] = -0.5*np.sum(w*dG)
    return dict(intra=out)


def _intra_covariant(h,nk=100,T=0.05,degeneracy_tol=1e-6):
    """sigma^intra_abc as a (3,3,3) array, from the covariant derivative of
    the metric (see _intra_covariant_batch), for the charge channel"""
    _check(h,T,"charge")
    return _mesh_tensors(h,nk,T,degeneracy_tol,
            kernel=_intra_covariant_batch)["intra"]
