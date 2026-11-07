# General n-th order nonlinear optical conductivity of a periodic
# tight-binding Hamiltonian, by the velocity-gauge density-matrix
# recursion, for any order and any tuple of complex frequencies. The
# derivation, the conventions and the checks are recorded in
# future_development/nonlinear_response.md ("The broadened recursion to
# any order"); this module holds the machinery and the public builders.
#
# Conventions, as in shg.py and kubo.py: hbar = 1, carriers of charge
# q = -1, lengths in the units of the geometry and energies in those of the
# hoppings. The field convention is that of Passos et al., PRB 97, 235446
# (2018), arXiv:1712.04924, Eq. (defresponsefunction):
#
#   J_b(w) = Int prod_j dw_j/(2 pi) sigma_{b a1..an}(w1..wn)
#            E_a1(w1)...E_an(wn) 2 pi delta(w - sum_j w_j)
#
# with time dependence exp(-i w t) and sigma symmetric under the exchange
# of the pairs (a_j,w_j). For a real field E(t) = Re E0 exp(-i w t) the
# second harmonic is (1/2) Re[sigma(w,w) E0^2 exp(-2 i w t)], the
# rectified current (1/2) sigma(w,-w) |E0|^2, and a static field gives
# J = sigma(0,...,0) E^n.
import itertools
import math
import warnings
from collections import Counter

import numpy as np

from .. import parallel
from ..htk.eigenvectors import hk_matrix_batch, parallel_diagonalization
from . import kubo
from . import shg


_charge = shg._charge # charge q of the carriers, in units of e: electrons.
                      # The n-th order response goes as q^(n+1)

_batch_bytes = 1e9 # memory budget of the vertices and density matrices of
                   # one k-batch, see _batch_size


def _backend():
    """Array module of the recursion. numpy on the CPU; the recursion is
    written as functional array code (no slice assignment into its arrays)
    so that jax.numpy can be plugged in here for the device"""
    return np


def _labels(components,order):
    """Cartesian index tuples of component strings such as "xyy" (the
    current first, then the n fields)"""
    out = []
    for comp in components:
        if not isinstance(comp,str) or len(comp)!=order+1: raise ValueError(
            "a component of the order-"+str(order)+" conductivity is "
            +str(order+1)+" Cartesian labels, the current first and then "
            "the "+str(order)+" fields, such as "+repr("x"*(order+1))
            +", got "+repr(comp))
        out.append(tuple(kubo._index(s) for s in comp))
    return out


def _all_components(order):
    """Every Cartesian component of the order-n tensor"""
    return ["".join(c) for c in itertools.product("xyz",repeat=order+1)]


def _frequency_classes(omegas):
    """For every field j, the first field whose frequency column equals
    the one of j. Two fields of the same class are at the same frequency
    in every row, so exchanging them only exchanges Cartesian labels"""
    n = omegas.shape[1]
    cls = []
    for j in range(n):
        for i in range(j+1):
            if np.array_equal(omegas[:,i],omegas[:,j]):
                cls.append(i) ; break
    return cls


def _vertex(labels):
    """Key of the vertex h^{a1..am}, symmetric in its labels"""
    return tuple(sorted(labels))


def _plan(labels,cls):
    """What the recursion has to compute for the requested components.

    A field is the pair (Cartesian label, frequency class). The symmetrized
    conductivity is the average over the n! orderings of the fields of the
    unsymmetrized one, so terms[ic] lists, for component ic, every
    distinct ordering of its fields with its weight (multiplicity/n!).
    Orderings that only exchange two fields of the same class and the same
    label coincide and are computed once.

    The density matrix at level p depends on the last p fields of an
    ordering (a suffix) only, so suffixes is the set of every suffix of
    every ordering, sorted by length so that each one comes after the
    shorter ones it is built from, and vertices the set of the sorted
    label tuples of every vertex the recursion and the traces use"""
    n = len(cls)
    terms = []
    for lab in labels:
        a = lab[1:]
        count = Counter(tuple((a[P[i]],cls[P[i]]) for i in range(n))
                for P in itertools.permutations(range(n)))
        terms.append([(fields,c/math.factorial(n))
                for (fields,c) in count.items()])
    suffixes = set()
    vertices = set()
    for (lab,tt) in zip(labels,terms):
        for (fields,w) in tt:
            for p in range(n+1): # vertices of the trace
                vertices.add(_vertex((lab[0],)+tuple(x[0] for x in fields[:p])))
            for p in range(n): # suffixes and the vertices that build them
                s = fields[p:]
                suffixes.add(s)
                for m in range(1,len(s)+1):
                    vertices.add(_vertex(tuple(x[0] for x in s[:m])))
    suffixes = sorted(suffixes,key=lambda s: (len(s),s))
    vertices = sorted(vertices,key=lambda v: (len(v),v))
    return terms,suffixes,vertices


def _batch_size(n,nvertices,nsuffixes):
    """k-points per batch, so that the vertices, the density matrices of
    every suffix and a few work arrays of one batch fit in _batch_bytes"""
    per_k = (nvertices + nsuffixes + 6)*n*n*16.
    return max(1,int(_batch_bytes/per_k))


def _batch_vertices(tms,dirs,d,hkgen,kb,vertices,T):
    """Band energies, occupations and the eigenbasis matrix elements of
    every needed vertex h^{a1..am} = d^m H/dK_a1..dK_am on a batch of
    k-points, see shg._derivative_tensors"""
    hks = hk_matrix_batch(hkgen,kb) # H(k) for the batch, densified
    es,wb = parallel_diagonalization(hks) # batched eigh
    es,wb = np.asarray(es),np.asarray(wb) # numpy, whatever the backend
    wc = np.conjugate(wb).transpose(0,2,1) # the bras <n|
    ph = kubo._bloch_phases(dirs,kb)
    verts = {v:wc@shg._cartesian_derivative(tms,d,ph,list(v))@wb
            for v in vertices}
    return es,kubo._fermi(es,T),verts


def _batch_sum(xp,es,f,verts,omegas,labels,terms,suffixes,dtype):
    """Unnormalized k-sum over one batch of the trace part of the
    conductivity, sum_k sum_p (1/p!) Tr[h^{b a1..ap} rho^{a(p+1)..an}],
    averaged over the orderings of the fields, for every frequency row and
    every component, as a (nw,ncomp) array.

    With rho^(0) = diag(f) and dE_nm = E_n - E_m the recursion is
    Passos et al., Eq. (rho), with hbar = 1:

      rho^{a1..ap}_nm = sum_{m=1}^{p} (1/m!) [h^{a1..am},rho^{a(m+1)..ap}]_nm
                        /(w1 + ... + wp - dE_nm)

    and [h,rho^(0)]_nm = h_nm (f_m - f_n). The denominators never vanish
    for complex frequencies, so degenerate bands need no special care"""
    n = len(omegas[0])
    es = xp.asarray(es)
    f = xp.asarray(f,dtype=dtype)
    dE = (es[:,:,None] - es[:,None,:]).astype(dtype) # E_n - E_m
    df = f[:,None,:] - f[:,:,None] # f_m - f_n
    hv = {v:xp.asarray(m,dtype=dtype) for (v,m) in verts.items()}
    out = np.zeros((len(omegas),len(terms)),dtype=np.complex128)
    for iw in range(len(omegas)): # loop over frequency tuples
        w = omegas[iw]
        rho = dict()
        for s in suffixes: # level by level, shorter suffixes first
            num = 0.
            for m in range(1,len(s)+1):
                v = hv[_vertex(tuple(x[0] for x in s[:m]))]
                rest = s[m:]
                if len(rest)==0: c = v*df # [h,rho^(0)]
                else: c = v@rho[rest] - rho[rest]@v
                num = num + c/math.factorial(m)
            Om = sum(w[c] for (a,c) in s) # frequency sum of the suffix
            rho[s] = num/(dtype(Om) - dE)
        for (ic,tt) in enumerate(terms): # loop over components
            b = labels[ic][0]
            acc = 0j
            for (fields,weight) in tt: # loop over orderings of the fields
                for p in range(n+1):
                    A = hv[_vertex((b,)+tuple(x[0] for x in fields[:p]))]
                    rest = fields[p:]
                    if len(rest)==0: # Tr[A rho^(0)] = sum_n A_nn f_n
                        tr = xp.sum(xp.diagonal(A,axis1=1,axis2=2)*f)
                    else: # Tr[A rho] = sum_nm A_nm rho_mn
                        tr = xp.sum(A*xp.swapaxes(rho[rest],1,2))
                    acc += weight*complex(tr)/math.factorial(p)
            out[iw,ic] = acc
    return out


def nonlinear_response(h,order,omegas,components,nk=20,T=0.1,
        degeneracy_tol=1e-6,prec="double"):
    """Symmetrized n-th order conductivity sigma_{b a1..an}(w1..wn) at the
    complex frequencies omegas, an (nw,order) array whose row i holds the
    n frequencies of the fields, the broadening already included. Returns
    a (nw,len(components)) array.

    This is the internal entry point: it takes the frequencies as they are
    and never checks the sign of their imaginary part, since the Laurent
    extraction of the dc limit evaluates it on a circle of complex
    broadenings. The public builder is nonlinear_conductivity.

    The conductivity is Passos et al., Eq. (responsefunction),

      sigma_{b a1..an} = (i^n q^(n+1)/(w1...wn)) (1/(N_k V_cell)) sum_k
                         sum_{p=0}^{n} (1/p!) Tr[h^{b a1..ap} rho^{a(p+1)..an}]

    averaged over the n! orderings of the pairs (a_j,w_j), with the
    density matrices of _batch_sum. degeneracy_tol is accepted for the
    uniform signature of the nonlinear routines: the recursion never
    divides by a band energy difference, so it is not used"""
    from .. import gpu
    if not isinstance(order,(int,np.integer)) or order<1: raise ValueError(
        "the order of the nonlinear conductivity is a positive integer, "
        "got "+repr(order))
    omegas = np.array(omegas,dtype=np.complex128)
    if omegas.ndim!=2 or omegas.shape[1]!=order: raise ValueError(
        "the frequencies of the order-"+str(order)+" conductivity are an "
        "(nw,"+str(order)+") array, one column per field, got shape "
        +str(omegas.shape))
    if h.dimensionality not in (1,2,3): raise NotImplementedError(
        "the nonlinear conductivity is implemented for dimensionality 1, 2 "
        "or 3, got dimensionality "+str(h.dimensionality))
    if h.has_eh: raise NotImplementedError("the nonlinear conductivity is "
        "not implemented for Nambu (superconducting) Hamiltonians: the "
        "charge current operator there is not i[H,r]")
    prec = gpu.resolve_prec(prec,"prec","the numpy recursion of "
            "nonlinearkernel.py")
    if prec=="single" and np.any(np.abs(omegas.real)<np.abs(omegas.imag)):
        warnings.warn("prec='single' with a field frequency below its "
            "broadening: the 1/(w1...wn) prefactor cancels terms of order "
            "one there, and the float32 error grows as (t/|w|)^n")
    dtype = np.complex64 if prec=="single" else np.complex128
    labels = _labels(components,order)
    terms,suffixes,vertices = _plan(labels,_frequency_classes(omegas))
    tms,dirs,d,hkgen,cellvol,scale = shg._derivative_tensors(h)
    ks = kubo._kmesh(h,nk)
    nb = tms.shape[1]
    kbatch = _batch_size(nb,len(vertices),len(suffixes))
    xp = _backend()
    parallel.set_num_threads() # set the number of threads for numba
    acc = np.zeros((len(omegas),len(labels)),dtype=np.complex128)
    for i0 in range(0,len(ks),kbatch): # loop over batches of k-points
        es,f,verts = _batch_vertices(tms,dirs,d,hkgen,ks[i0:i0+kbatch],
                vertices,T)
        acc += _batch_sum(xp,es,f,verts,omegas,labels,terms,suffixes,dtype)
    pref = (1j**order)*_charge**(order+1)/np.prod(omegas,axis=1)
    return acc*pref[:,None]/(len(ks)*cellvol) # (1/V) sum_k, V = N_k V_cell


def _omegas(energies,frequencies,order,delta):
    """Complex frequencies w_j = c_j*energies + i delta of the public
    builders, an (nw,order) array"""
    if frequencies is None: frequencies = np.ones(order)
    c = np.array(frequencies,dtype=np.float64)
    if c.ndim!=1 or len(c)!=order: raise ValueError(
        "frequencies are the "+str(order)+" multipliers c_j of the fields "
        "of the order-"+str(order)+" conductivity, w_j = c_j*energies, "
        "such as "+repr(tuple([1]*order))+", got "+repr(frequencies))
    if not delta>0.: raise ValueError("the broadening delta must be "
        "positive, the response has a 1/((w1 + i delta)...(wn + i delta)) "
        "prefactor, got delta="+str(delta))
    return energies[:,None]*c[None,:] + 1j*delta


def nonlinear_conductivity(h,order=2,energies=None,frequencies=None,nk=20,
        T=None,delta=0.1,components=None,degeneracy_tol=1e-6,
        prec="double"):
    """n-th order conductivity sigma_{b a1..an}(w1..wn) with the field
    frequencies w_j = c_j*energies + i delta, every field carrying the
    same broadening (the adiabatic switching of Passos et al., not a
    relaxation time).

    frequencies are the multipliers c_j: None is all ones (harmonic
    generation), (1,-1) the rectified current, (1,2) a sum frequency,
    (1,1,0) the field-induced second harmonic and (1,-1,0) the jerk
    current. components are strings of order+1 Cartesian labels, the
    current first ("xyy" at second order, "xxyy" at third), all of them by
    default. Returns (energies, array) for a single component string and
    (energies, dict) otherwise, as shg_conductivity.

    energies=[0.] gives the dc value of the adiabatic-switching recursion
    at finite delta. It is not the tau-expanded dc conductivity and cannot
    be read at delta = 1/tau: at dc the n-field Drude term goes as
    1/(n! delta^n) where the relaxation time gives tau^n, a one-pole term
    as 1/delta, and the delta^0 coefficient matches none of the
    relaxation-time conventions (see future_development/
    nonlinear_response.md, "The dc limit and its Laurent structure").

    T defaults to delta, as in shg_conductivity; the chemical potential
    is zero (use h.shift_fermi). The conductivity is per unit cell volume
    (area in 2D, length in 1D), in units of e^(n+1) a^(n+1-d)/(hbar
    t^(n-1)). prec="single" needs the device (gpu.set_gpu(True)) and is
    meaningful only for |c_j*energies| above delta"""
    if energies is None: energies = np.linspace(0.,4.,100)
    energies = np.array(energies,dtype=np.float64)
    if T is None: T = delta
    omegas = _omegas(energies,frequencies,order,delta)
    single = isinstance(components,str)
    if components is None: components = _all_components(order)
    if single: components = [components]
    s = nonlinear_response(h,order,omegas,components,nk=nk,T=T,
            degeneracy_tol=degeneracy_tol,prec=prec)
    if single: return energies,s[:,0]
    return energies,{c:s[:,i] for (i,c) in enumerate(components)}


def nonlinear_susceptibility(h,order=2,length_unit=1.0,energy_unit=1.0,
        **kwargs):
    """n-th order susceptibility chi^(n) in SI units, from
    nonlinear_conductivity (same keywords), see to_susceptibility"""
    energies,sigma = nonlinear_conductivity(h,order=order,**kwargs)
    energies = np.array(energies,dtype=np.float64)
    omegas = _omegas(energies,kwargs.get("frequencies",None),order,
            kwargs.get("delta",0.1))
    return energies,to_susceptibility(h,omegas,sigma,length_unit,
            energy_unit)


def to_susceptibility(h,omegas,sigma,length_unit=1.0,energy_unit=1.0):
    """Convert an n-th order conductivity (an array, or a dict of arrays)
    at the complex frequencies omegas, an (nw,n) array, to the SI
    susceptibility.

    With J = dP/dt the polarization at the sum frequency Omega = w1 + ...
    + wn is P = J/(-i Omega), so chi^(n) = sigma^(n)/(-i Omega epsilon_0),
    the generalization of shg.to_susceptibility (n = 2). In the units of
    the conductivity (e = hbar = 1, energies in units of t and lengths in
    units of a) this is

      chi[SI] = sigma/(-i Omega) * e a^(n+1-d)/(epsilon_0 t^n)

    with a = length_unit in Angstrom and t = energy_unit in eV, returned in
    nm^(n+2-d)/V^(n-1) (nm^2/V for the sheet susceptibility of a 2D system
    at second order)"""
    omegas = np.array(omegas,dtype=np.complex128)
    n = omegas.shape[1]
    d = h.dimensionality
    a = length_unit*1e-10 # in m
    fac = shg._e_over_eps0*a**(n+1-d)/energy_unit**n/(1e-9)**(n+2-d)
    Om = np.sum(omegas,axis=1)
    def conv(s): return s/(-1j*Om)*fac
    if isinstance(sigma,dict): return {c:conv(s) for (c,s) in sigma.items()}
    return conv(sigma)
