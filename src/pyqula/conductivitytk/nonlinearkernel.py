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
import functools
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
    """Array module of the recursion, from the package-wide CPU/GPU switch
    (pyqula/gpu.py): numpy on the CPU and jax.numpy on the device. The
    recursion (_row) is written as functional array code, with no slice
    assignment into its arrays, so the same function runs on both"""
    from .. import gpu # sets jax to double precision before any jax array
    if gpu.get_gpu():
        import jax.numpy as jnp
        return jnp
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


def _program(labels,terms,suffixes,vertices):
    """The recursion and the traces of _plan as a fixed list of operations
    on integer indices into the lists of vertices and suffixes, a hashable
    tuple, so that the device kernel can be compiled once per plan.

    steps[i] builds rho of suffixes[i]: the frequency classes of its fields
    and, for every m, the index of the vertex h^{a1..am}, the index of the
    shorter suffix it acts on (-1 for rho^(0)) and m. traces lists the
    distinct pairs (vertex, suffix) of every Tr[h^{b a1..ap} rho^{...}],
    and comps[ic] the traces of component ic with their weights
    (ordering weight)/p!, summed over the orderings that share a trace"""
    n = len(labels[0]) - 1 # the order
    iv = {v:i for (i,v) in enumerate(vertices)}
    isuf = {s:i for (i,s) in enumerate(suffixes)}
    steps = []
    for s in suffixes:
        parts = tuple((iv[_vertex(tuple(x[0] for x in s[:m]))],
                isuf[s[m:]] if m<len(s) else -1,m)
                for m in range(1,len(s)+1))
        steps.append((tuple(c for (a,c) in s),parts))
    traces = dict()
    comps = []
    for (lab,tt) in zip(labels,terms):
        comp = dict()
        for (fields,weight) in tt:
            for p in range(n+1):
                key = (iv[_vertex((lab[0],)+tuple(x[0] for x in fields[:p]))],
                        isuf[fields[p:]] if p<n else -1)
                it = traces.setdefault(key,len(traces))
                comp[it] = comp.get(it,0.) + weight/math.factorial(p)
        comps.append(tuple(comp.items()))
    return tuple(steps),tuple(traces),tuple(comps)


def _row(xp,mm,dE,df,f,hv,w,program):
    """The trace part of the conductivity at one frequency row w (its n
    complex frequencies) for every component, an (ncomp,) array, summed
    over the k-points of a batch. hv[i] is vertex i on the batch, mm the
    matrix product (numpy's, or jax's at full float32 precision).

    With rho^(0) = diag(f) and dE_nm = E_n - E_m the recursion is
    Passos et al., Eq. (rho), with hbar = 1:

      rho^{a1..ap}_nm = sum_{m=1}^{p} (1/m!) [h^{a1..am},rho^{a(m+1)..ap}]_nm
                        /(w1 + ... + wp - dE_nm)

    and [h,rho^(0)]_nm = h_nm (f_m - f_n). The denominators never vanish
    for complex frequencies, so degenerate bands need no special care.
    Written without slice assignment, so that jax can trace it"""
    steps,traces,comps = program
    rho = []
    for (cls,parts) in steps: # level by level, shorter suffixes first
        num = 0.
        for (i,j,m) in parts:
            v = hv[i]
            if j<0: c = v*df # [h,rho^(0)]
            else: c = mm(v,rho[j]) - mm(rho[j],v)
            num = num + c/math.factorial(m)
        Om = sum(w[c] for c in cls) # frequency sum of the suffix
        rho.append(num/(Om - dE))
    tr = []
    for (i,j) in traces:
        A = hv[i]
        if j<0: # Tr[A rho^(0)] = sum_n A_nn f_n
            tr.append(xp.sum(xp.diagonal(A,axis1=1,axis2=2)*f))
        else: # Tr[A rho] = sum_nm A_nm rho_mn
            tr.append(xp.sum(A*xp.swapaxes(rho[j],1,2)))
    return xp.stack([sum(wt*tr[i] for (i,wt) in comp) for comp in comps])


def _batch_sum(xp,es,f,verts,omegas,labels,terms,suffixes,dtype):
    """Unnormalized k-sum over one batch of the trace part of the
    conductivity, sum_k sum_p (1/p!) Tr[h^{b a1..ap} rho^{a(p+1)..an}],
    averaged over the orderings of the fields, for every frequency row and
    every component, as a (nw,ncomp) array, one row at a time (see _row).
    This is the CPU path; the device one is _device_sum"""
    vertices = list(verts)
    program = _program(labels,terms,suffixes,vertices)
    es = xp.asarray(es)
    f = xp.asarray(f,dtype=dtype)
    dE = (es[:,:,None] - es[:,None,:]).astype(dtype) # E_n - E_m
    df = f[:,None,:] - f[:,:,None] # f_m - f_n
    hv = [xp.asarray(verts[v],dtype=dtype) for v in vertices]
    ws = np.array(omegas,dtype=dtype)
    out = np.zeros((len(omegas),len(terms)),dtype=np.complex128)
    for iw in range(len(omegas)): # loop over frequency tuples
        out[iw] = np.asarray(_row(xp,xp.matmul,dE,df,f,hv,ws[iw],program))
    return out


# The device path
#
# On the device the recursion of one frequency row is a fixed graph of
# batched (nk,n,n) matmuls once the plan is fixed, so it is compiled with
# the plan (_program) static and vmapped over a chunk of frequency rows:
# the device sees (nw,nk,n,n) arrays. The eigenpairs come from
# htk/eigenvectorsjax.peigh_gpu, the vertices are built and rotated on the
# device and stay there for the whole frequency sweep of their k-batch.
# The last k-batch and the last frequency chunk are padded to the shape of
# the others, so the kernel compiles once per call: a padded k-point is a
# copy of a real one with zero occupations, which makes every rho and every
# trace vanish there exactly, and a padded frequency row is dropped.
#
# Every contraction asks for Precision.HIGHEST: without it jax runs float32
# products on an Ampere or later card in TF32, with a ten-bit mantissa
# (see bsetk/chebyshevjax.py).

_device_bytes = 1e9 # device memory budget of one k-batch and frequency
                    # chunk, see _device_chunks, which counts the vertices
                    # and the work buffers of the sweep (measured 2.5 to 3.2
                    # times the density matrices); the card may be shared
                    # with other jax processes, see _device_budget


@functools.lru_cache(maxsize=None)
def _device_kernels():
    """The two compiled device functions, built once per process: the
    eigenbasis vertices of a k-batch, and the frequency sweep"""
    from .. import gpu # double precision before any jax array
    import jax
    import jax.numpy as jnp
    hi = jax.lax.Precision.HIGHEST
    def mm(a,b): return jnp.matmul(a,b,precision=hi)
    def vertices(ph,W,wb):
        """<n|d^m H/dK_a1..dK_am|m> for every vertex, (nvert,nk,n,n), from
        the Bloch phases ph (nk,nhop), the vertex hoppings W
        (nvert,nhop,n,n) and the eigenvectors wb (nk,n,n)"""
        hk = jnp.einsum("kh,vhnm->vknm",ph,W,precision=hi)
        wc = jnp.conj(jnp.swapaxes(wb,1,2)) # the bras <n|
        return mm(mm(wc[None],hk),wb[None])
    def sweep(hv,dE,df,f,ws,program):
        def row(w): return _row(jnp,mm,dE,df,f,hv,w,program)
        return jax.vmap(row)(ws)
    return jax.jit(vertices),jax.jit(sweep,static_argnums=(5,))


def _device_budget():
    """Bytes for one k-batch and frequency chunk on the device:
    _device_bytes, capped by half of the free memory of the card when jax
    reports it (None on some backends)"""
    import jax
    budget = _device_bytes
    st = jax.devices()[0].memory_stats()
    if st and "bytes_limit" in st:
        budget = min(budget,0.5*(st["bytes_limit"]-st["bytes_in_use"]))
    return max(budget,1e8)


def _device_chunks(nk,n,nvertices,nsuffixes,nw,itemsize):
    """k-points per batch and frequency rows per chunk of the device sweep,
    chosen so that the whole footprint fits in _device_budget(): the vertex
    kernel holds hk, its product with the bras and hv, three arrays of
    (nvertices,kbatch,n,n); the sweep keeps hv, dE and df resident and
    needs about 3.5 density matrices per suffix and frequency row (the
    measured 3.2 with a margin)"""
    budget = _device_budget()
    unit = n*n*itemsize # one (n,n) matrix
    kbatch = max(1,min(nk,int(budget/((3*nvertices+2)*unit)),
            int(budget/((nvertices+2+3.5*nsuffixes)*unit))))
    resident = (nvertices+2)*kbatch*unit
    wchunk = max(1,int(min(nw,(budget-resident)/(3.5*nsuffixes*kbatch*unit))))
    return kbatch,wchunk


def _pad(a,size,fill=None):
    """a padded along its first axis to size, with copies of its last
    entry (fill=None) or with fill"""
    if len(a)>=size: return a
    extra = np.repeat(a[-1:],size-len(a),axis=0)
    if fill is not None: extra = np.full_like(extra,fill)
    return np.concatenate([a,extra])


def _device_sum(tms,dirs,d,hkgen,ks,vertices,T,omegas,labels,terms,
        suffixes,prec):
    """The unnormalized k-sum of _batch_sum over the whole mesh ks, on the
    device, as an (nw,ncomp) complex128 array"""
    from ..htk import eigenvectorsjax
    import jax.numpy as jnp
    dtype = np.complex64 if prec=="single" else np.complex128
    nb = tms.shape[1]
    program = _program(labels,terms,suffixes,vertices)
    kbatch,wchunk = _device_chunks(len(ks),nb,len(vertices),len(suffixes),
            len(omegas),np.dtype(dtype).itemsize)
    fvertices,sweep = _device_kernels()
    W = np.array([tms*np.prod([1j*d[:,a] for a in v],axis=0) for v in
            vertices]) # hoppings of every vertex, (nvert,nhop,n,n)
    W = jnp.asarray(W,dtype=dtype)
    ws = np.array(omegas,dtype=dtype)
    acc = np.zeros((len(omegas),len(labels)),dtype=np.complex128)
    for i0 in range(0,len(ks),kbatch): # loop over batches of k-points
        kb = np.array(ks[i0:i0+kbatch])
        es,wb = eigenvectorsjax.peigh_gpu(hk_matrix_batch(hkgen,kb),
                prec=prec)
        f = _pad(kubo._fermi(es,T),kbatch,fill=0.) # empty padded k-points
        es,wb,kb = _pad(es,kbatch),_pad(wb,kbatch),_pad(kb,kbatch)
        ph = kubo._bloch_phases(dirs,kb)
        hv = fvertices(jnp.asarray(ph,dtype=dtype),W,
                jnp.asarray(wb,dtype=dtype))
        dE = jnp.asarray(es[:,:,None] - es[:,None,:],dtype=dtype)
        df = jnp.asarray(f[:,None,:] - f[:,:,None],dtype=dtype)
        fd = jnp.asarray(f,dtype=dtype)
        for j0 in range(0,len(ws),wchunk): # loop over chunks of frequencies
            wj = ws[j0:j0+wchunk]
            out = sweep(hv,dE,df,fd,jnp.asarray(_pad(wj,wchunk)),program)
            acc[j0:j0+len(wj)] += np.asarray(out,dtype=np.complex128)[:len(wj)]
    return acc


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
    density matrices of _row. degeneracy_tol is accepted for the
    uniform signature of the nonlinear routines: the recursion never
    divides by a band energy difference, so it is not used.

    Under pyqula.gpu.set_gpu(True) the recursion runs on the device
    (_device_sum), where prec="single" is also accepted; the CPU path is
    double precision only"""
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
    if xp is np: # the CPU path
        parallel.set_num_threads() # set the number of threads for numba
        acc = np.zeros((len(omegas),len(labels)),dtype=np.complex128)
        for i0 in range(0,len(ks),kbatch): # loop over batches of k-points
            es,f,verts = _batch_vertices(tms,dirs,d,hkgen,ks[i0:i0+kbatch],
                    vertices,T)
            acc += _batch_sum(xp,es,f,verts,omegas,labels,terms,suffixes,
                    dtype)
    else: acc = _device_sum(tms,dirs,d,hkgen,ks,vertices,T,omegas,labels,
            terms,suffixes,prec) # the device path
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
