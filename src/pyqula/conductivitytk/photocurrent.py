# Second-order photocurrents and frequency mixing of a periodic
# tight-binding Hamiltonian: the rectified current sigma_abc(0;w,-w) under
# light, split into the injection current (a rate) and the shift current,
# and the sum and difference frequency conductivities sigma_abc(w1,w2) at
# w_j = c_j*w. Every total comes from the general velocity-gauge recursion
# of nonlinearkernel.py; the injection rate has its own closed form. The
# derivation and the checks are recorded in future_development/
# nonlinear_response.md ("The two-frequency kernel: photocurrents and sum
# frequency").
#
# Conventions, as in shg.py and nonlinearkernel.py: hbar = 1, carriers of
# charge q = -1, energies in units of the hoppings, every field frequency
# carrying the broadening w -> w + i delta (adiabatic switching), the
# chemical potential at zero, and the conductivity per unit cell volume
# (area in 2D, length in 1D), in units of e^3 a^(3-d)/(hbar t). For the
# real field E(t) = Re E0 exp(-i w t) the rectified current is
#
#   J_a = (1/2) sigma_abc(w,-w) E0_b E0_c^*
#
# and the injection current grows as dJ_a/dt = eta_abc E0_b E0_c^*.
import itertools

import numpy as np

from .. import parallel
from . import kubo
from . import shg
from . import nonlinearkernel


_charge = shg._charge # charge q of the carriers, in units of e: electrons


def _all_components():
    """Every Cartesian component of a rank-3 tensor, the current first"""
    return ["".join(c) for c in itertools.product("xyz",repeat=3)]


def _check(h,delta,what):
    """Guards of the routines that do not go through the general kernel,
    with what the noun phrase that names the routine"""
    if h.dimensionality not in (1,2,3): raise NotImplementedError(
            what+" is implemented for dimensionality 1, 2 or 3, got "
            "dimensionality "+str(h.dimensionality))
    if h.has_eh: raise NotImplementedError(what+" is not implemented for "
            "Nambu (superconducting) Hamiltonians: the charge current "
            "operator there is not i[H,r]")
    if not delta>0.: raise ValueError("the broadening delta of "+what
            +" must be positive, the rectified current has a pole "
            "eta/delta, got delta="+str(delta))


def _triplets(components):
    """Cartesian index triplets (a,b,c) of component strings such as
    "yxx", the current first"""
    out = []
    for comp in components:
        if not isinstance(comp,str) or len(comp)!=3: raise ValueError(
            "a component of a second-order photocurrent is three Cartesian "
            "labels, the current first and then the two fields, such as "
            "'xyy', got "+repr(comp))
        out.append(tuple(kubo._index(s) for s in comp))
    return out


def _out(energies,s,components,single):
    """(energies, array) for a single component string, (energies, dict)
    otherwise, as shg_conductivity"""
    if single: return energies,s[:,0]
    return energies,{c:s[:,i] for (i,c) in enumerate(components)}


def injection_rate(es,h1,f,w,delta,triplets,tol):
    """Golden-rule injection rate eta_abc(w), summed (not normalized) over
    a batch of k-points, for every frequency of w and every Cartesian
    triplet (a,b,c), as a (len(w),len(triplets)) complex array.

    es[k,n] are the band energies, h1[k,a,n,m] the eigenbasis matrix
    elements of the velocity dH/dK_a (shg._eigenbasis_derivatives) and
    f[k,n] the occupations. With eps_mn = E_m - E_n, f_nm = f_n - f_m,
    Delta^a_mn = h^a_mm - h^a_nn and r^a_nm = h^a_nm/(i(E_n - E_m)), for
    nondegenerate bands

      eta_abc = (pi q^3/2) sum_k sum_{n,m} f_nm Delta^a_mn r^b_mn r^c_nm
                L_delta(w - eps_mn),   L_delta(x) = (delta/pi)/(x^2 + delta^2)

    where the sum leaves out every pair of bands closer than tol, the
    pairs inside one degenerate multiplet, which carry no interband
    transition. When bands are degenerate the diagonal h^a_mm depends on
    the basis that the diagonalization picked inside the multiplet, unless
    the velocity there is a multiple of the identity (Kramers pairs of a
    spin-degenerate model). That is not the case where bands meet on lines
    or points of the zone, as in every supercell, whose folded bands are
    degenerate along whole lines. So Delta^a_mn r^c_nm is replaced by its
    multiplet-block form [r^c, P h^a P]_nm, with P h^a P the velocity
    restricted to the pairs inside a multiplet (its diagonal included):

      sum_{m'} r^c_nm' h^a_m'm - sum_{n'} h^a_nn' r^c_n'm

    with m' in the multiplet of m and n' in the one of n. Summed over the
    two multiplets this is Tr[P_M h^a P_M r^b P_N r^c] - Tr[P_N h^a P_N
    r^c P_M r^b], invariant under a rotation inside each multiplet, and
    it reduces to Delta^a_mn r^c_nm for nondegenerate bands. Nothing here
    assumes an insulator: f can be any occupation"""
    w = np.atleast_1d(np.array(w,dtype=np.float64))
    nk,n = es.shape
    eps = es[:,None,:] - es[:,:,None] # eps[k,n,m] = E_m - E_n
    keep = np.abs(eps)>tol # pairs in different multiplets
    inv = np.where(keep,1./np.where(keep,eps,1.),0.) # 1/(E_m - E_n)
    fnm = (f[:,:,None] - f[:,None,:])*keep # f_n - f_m, zero inside a multiplet
    pairs = np.nonzero(fnm) # only pairs with an occupation difference enter
    e = eps[pairs] # (npair,)
    weights = np.zeros((len(e),len(triplets)),dtype=np.complex128)
    rbs = {} # r^b_mn at [k,n,m], built once per b
    def rb_mn(b):
        if b not in rbs: rbs[b] = np.swapaxes(h1[:,b],1,2)*inv/1j
        return rbs[b]
    Drcs = {} # [r^c, P h^a P], built once per (a,c)
    def Drc(a,c):
        if (a,c) not in Drcs:
            rc = -h1[:,c]*inv/1j # r^c_nm at [k,n,m], 1/(E_n - E_m) = -inv
            va = h1[:,a]*(~keep) # velocity inside the multiplets, P h^a P
            # [r^c, P h^a P]_nm, Delta^a_mn r^c_nm if nondegenerate
            Drcs[(a,c)] = rc@va - va@rc
        return Drcs[(a,c)]
    for (t,(a,b,c)) in enumerate(triplets):
        weights[:,t] = (fnm*rb_mn(b)*Drc(a,c))[pairs]
    L = (delta/np.pi)/((w[:,None] - e[None,:])**2 + delta**2) # (nw,npair)
    return np.pi*_charge**3/2.*(L@weights)


_batch_bytes = 2e8 # memory budget of one batch of k-points of injection_current


def _batch_size(n,nw,nt):
    """k-points per batch of injection_current for n bands, nw frequencies
    and nt triplets: per k-point the velocities, the eigenvectors and the
    pair arrays of injection_rate hold about (40 + nt) complex (n,n)
    matrices, and the Lorentzians nw real numbers per pair. A fixed batch
    of 64 k-points made a few-band 3D mesh a loop of many small batches"""
    per_k = (16.*(40+nt) + 24.*nw)*n*n # 24: L and its two temporaries
    return int(max(1,min(8192,_batch_bytes/per_k)))


def injection_current(h,energies=None,nk=20,T=None,delta=0.1,
        components=None,degeneracy_tol=1e-6):
    """Injection rate eta_abc(w) of the rectified current under light,
    dJ_a/dt = eta_abc E0_b E0_c^* for E(t) = Re E0 exp(-i w t), from the
    golden-rule closed form of injection_rate on the uniform k-mesh, with
    the transitions broadened into Lorentzians of width delta.

    eta is complex, with eta_acb = eta_abc^*. Its real part, symmetric in
    b and c, is the linear injection: the current rate under linearly
    polarized light, which vanishes with time reversal. Its imaginary
    part, antisymmetric in b and c, is the circular injection,
    eta^circ_abc = (eta_abc - eta_acb)/(2i) = Im eta_abc, the current rate
    under circularly polarized light, allowed with time reversal once
    inversion is broken. For E0 = (1, i, 0)/sqrt(2), say, the rate is
    dJ_a/dt = (Re eta_axx + Re eta_ayy)/2 + Im eta_axy.

    With adiabatic switching the rate integrates to J = eta |E0|^2/(2
    delta), so eta/delta is the pole of sigma_abc(w,-w) (exactly the
    diagonal part of the length gauge, shg.sigma2_length_diagonal, at the
    same delta, for a nondegenerate spectrum; degenerate bands, as in a
    supercell, are handled here by the multiplet blocks of
    injection_rate). T defaults to delta as in shg_conductivity and the
    occupations can be those of a metal. In units of e^3 a^(3-d)/hbar^2.
    Returns (energies, array) for a single component string and (energies,
    dict) otherwise"""
    if energies is None: energies = np.linspace(0.,4.,100)
    energies = np.array(energies,dtype=np.float64)
    if T is None: T = delta
    single = isinstance(components,str)
    if components is None: components = _all_components()
    if single: components = [components]
    _check(h,delta,"the injection current")
    triplets = _triplets(components)
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    ks = kubo._kmesh(h,nk)
    tol = degeneracy_tol*scale
    parallel.set_num_threads() # set the number of threads for numba
    acc = np.zeros((len(energies),len(triplets)),dtype=np.complex128)
    nb = _batch_size(tms.shape[1],len(energies),len(triplets))
    for i0 in range(0,len(ks),nb): # loop over batches of k-points
        es,h1,h2,h3 = shg._eigenbasis_derivatives(tms,dirs,d,hkgen,
                ks[i0:i0+nb],vertices=shg.vertex_keys(1)) # h1 only
        acc += injection_rate(es,h1,kubo._fermi(es,T),energies,delta,
                triplets,tol)
    acc = acc/(len(ks)*cellvol) # (1/V) sum_k, V = N_k V_cell
    return _out(energies,acc,components,single)


def sum_frequency_conductivity(h,multipliers=(1,1),energies=None,nk=20,
        T=None,delta=0.1,components=None,degeneracy_tol=1e-6,
        prec="double"):
    """Second-order conductivity sigma_abc(w1 + w2;w1,w2) at the field
    frequencies w_j = c_j*energies + i delta, with multipliers = (c1,c2):
    (1,1) is the second harmonic, (1,2) the sum frequency w + 2w, (2,-1)
    the difference frequency 2w - w and (1,-1) the rectified current. The
    field along b is at c1*w and the one along c at c2*w, and sigma is
    averaged over the two orderings of the fields, so that
    sigma_abc(c1,c2) = sigma_acb(c2,c1). For E(t) = Re[E1 exp(-i w1 t) +
    E2 exp(-i w2 t)] with w1 != w2 the current at w1 + w2 is
    Re[sigma_abc(w1,w2) E1_b E2_c exp(-i(w1 + w2)t)].

    This is nonlinearkernel.nonlinear_conductivity at order 2, the
    velocity-gauge recursion, with its keywords, units and backend (prec
    is per call, "single" needs the device); see there"""
    c = np.array(multipliers,dtype=np.float64)
    if c.ndim!=1 or len(c)!=2: raise ValueError("multipliers are the two "
            "factors (c1,c2) of the field frequencies w_j = c_j*energies of "
            "the second-order conductivity, such as (1,2), got "
            +repr(multipliers))
    if not delta>0.: raise ValueError("the broadening delta must be "
            "positive, the response has a 1/((w1 + i delta)(w2 + i delta)) "
            "prefactor, got delta="+str(delta))
    return nonlinearkernel.nonlinear_conductivity(h,order=2,
            energies=energies,frequencies=tuple(c),nk=nk,T=T,delta=delta,
            components=components,degeneracy_tol=degeneracy_tol,prec=prec)


def rectified_conductivity(h,energies=None,nk=20,T=None,delta=0.1,
        components=None,degeneracy_tol=1e-6,prec="double"):
    """Rectified (bulk photovoltaic) conductivity sigma_abc(0;w,-w), the
    dc current J_a = (1/2) sigma_abc(w,-w) E0_b E0_c^* under the field
    E(t) = Re E0 exp(-i w t), with its two parts:

      sigma_abc(w,-w) = eta_abc(w)/delta + sigma^shift_abc(w)

    The first is the injection current (injection_current), the pole of
    the rectified response: a rate eta that adiabatic switching turns into
    a current eta/(2 delta) per unit |E0|^2. The second, the shift
    current, is what remains, finite as delta -> 0 in an insulator; it
    still moves at order delta (on a Haldane-gapped honeycomb lattice at
    w = 2, near the absorption edge, by 15 % between delta = 0.1 and 0.05
    on a converged mesh, while eta/delta more than doubles). Both
    terms are taken at the same delta, nk and T. The total is the
    velocity-gauge recursion of nonlinearkernel.py; the split is not done
    inside it, since its n = m part is not the pole (the sum rules move
    weight between its terms, and only the total is gauge invariant).

    The total, and so the shift current, is Hermitian in b and c:
    sigma_acb(w,-w) = sigma_abc(w,-w)^*. When time reversal is broken the
    1/delta pole amplifies the aliasing of the mesh, so the total needs a
    larger nk than the second harmonic (on a Haldane-gapped honeycomb
    lattice with delta = 0.1 the two gauges differ by 6 % at nk=48 and 0.2 %
    at nk=96). In a metal the total also holds Fermi-surface terms (the
    Berry curvature dipole and Drude-like poles) that eta does not, and
    the remainder is then not only the shift current.

    Keywords and units as nonlinearkernel.nonlinear_conductivity, T
    defaulting to delta. Returns (energies, dict) with the keys "total",
    "shift" and "injection" (eta, complex), each an array for a single
    component string and a dict of arrays otherwise"""
    if energies is None: energies = np.linspace(0.,4.,100)
    energies = np.array(energies,dtype=np.float64)
    if T is None: T = delta
    _check(h,delta,"the rectified conductivity")
    single = isinstance(components,str)
    comps = _all_components() if components is None else components
    _triplets([comps] if single else comps) # the labels, checked here
    (w,tot) = nonlinearkernel.nonlinear_conductivity(h,order=2,
            energies=energies,frequencies=(1,-1),nk=nk,T=T,delta=delta,
            components=comps,degeneracy_tol=degeneracy_tol,prec=prec)
    (w,eta) = injection_current(h,energies=energies,nk=nk,T=T,delta=delta,
            components=comps,degeneracy_tol=degeneracy_tol)
    if single: shift = tot - eta/delta
    else: shift = {c:tot[c] - eta[c]/delta for c in comps}
    return energies,{"total":tot,"shift":shift,"injection":eta}


def shift_current(h,energies=None,nk=20,T=None,delta=0.1,components=None,
        degeneracy_tol=1e-6,prec="double"):
    """Shift-current conductivity sigma^shift_abc(w) = sigma_abc(0;w,-w)
    - eta_abc(w)/delta, the part of the rectified current under light that
    stays finite as delta -> 0, J_a = (1/2) sigma^shift_abc E0_b E0_c^*.
    It needs broken inversion and is allowed with time reversal. See
    rectified_conductivity for the conventions, the mesh and the caveat on
    metals. Returns (energies, array) for a single component string and
    (energies, dict) otherwise"""
    (w,out) = rectified_conductivity(h,energies=energies,nk=nk,T=T,
            delta=delta,components=components,
            degeneracy_tol=degeneracy_tol,prec=prec)
    return w,out["shift"]
