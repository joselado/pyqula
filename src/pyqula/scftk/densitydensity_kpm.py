# KPM-based (Chebyshev, sparse) alternative to scftk/densitydensity.py
#
# This is a parallel implementation of the density-density mean-field SCF:
# same interaction dictionary "v" (U, V1, V2, V3, Vr), same mean-field
# update rule (get_mf/normal_term_ii/jj/ij/ji), same mixing/convergence
# logic -- reused directly from densitydensity.py -- but the density
# matrix itself is computed with kpmtk.densitymatrix_kpm.get_dm_kpm (which
# samples the same k-mesh as the exact-diagonalization path and gets each
# needed element via Chebyshev recursion on the small Bloch Hamiltonian
# H(k), instead of diagonalizing H(k)). Intended for large/sparse
# Hamiltonians where the diagonalization in the exact-diagonalization path
# becomes the bottleneck; only the "plain" (fixed-point mixing) solver is
# supported.
from .. import filewrite
import numpy as np
import os
from .. import filesystem as fs
import time
from copy import copy, deepcopy

from .. import inout
from .. import algebra
from ..kpmtk.densitymatrix_kpm import get_dm_kpm, DEFAULT_NK, DEFAULT_NPOL

mf_file = "MF.pkl"  # same filename densitydensity.py uses

# NOTE: densitydensity.py itself is imported by meanfield.py, which this
# module is in turn imported from (meanfield.py exposes Vinteraction_kpm/
# hubbard_kpm) -- so densitydensity.py, this module, and meanfield.py form
# an import cycle. A module-level import of densitydensity.py's helpers
# anywhere in this file -- even at the bottom -- is only safe for entry
# orders where densitydensity.py has already fully finished loading by the
# time that line runs; it breaks if densitydensity.py itself is the first
# of the three modules touched (its own bottom import of ..meanfield fires
# before its *own* SCF class/obj2geometryarray are defined, and unwinding
# back through meanfield.py to this file's top-level import would then see
# a still-partial densitydensity module). Importing these names inside
# each function body instead (executed only when the function is actually
# *called*, long after every module involved has finished its own initial
# load) sidesteps the cycle regardless of which module a caller happens to
# import first.


def generic_densitydensity_kpm(h0, mf=None, mix=0.1, v=None, nk=DEFAULT_NK,
        maxerror=1e-5, callback_mf=None, callback_dm=None, load_mf=True,
        compute_cross=True, compute_dd=True, verbose=1,
        compute_anomalous=True, compute_normal=True, maxite=1000,
        T=1e-7, callback_h=None,
        scale=None, npol=DEFAULT_NPOL, ne=None, cores=None, write=None,
        kpm_prec=None, fermi=None, **kwargs):
    """KPM analogue of scftk.densitydensity.generic_densitydensity.
    Only the "plain" mixing solver is implemented (the alternate
    root-finding solvers there are not KPM-specific and are not needed for
    this backend). write=False keeps the converged mean field out of MF.pkl
    in the working directory. kpm_prec is the precision of the Chebyshev
    recursion, None for single on the GPU and double on the CPU (see
    kpmtk.densitymatrix_kpm.resolve_kpm_prec). fermi is a
    kpmtk.densitymatrix_kpm.LaggedFermi that shifts every Hamiltonian by
    its Fermi level, after callback_h, and whose filling error enters the
    convergence check."""
    write = filewrite.resolve(write,True) # the call, else the global switch
    from .densitydensity import (get_mf, mix_mf, diff_mf, update_hamiltonian,
            hamiltonian2dict, set_hoppings, SCF, random_hermitian_guess,
            mf_matches_hamiltonian, reject_leftover_kwargs)
    reject_leftover_kwargs(kwargs) # the end of the KPM call chain
    from .mfconstrains import obj2mf
    # a sparse Hamiltonian goes through the sparse counterparts of the
    # density matrix and the mean field (scftk/sparsemeanfield.py), so that
    # no n x n matrix is ever built; a dense one through the dense ones
    sparse = h0.is_sparse
    if sparse:
        from . import sparsemeanfield
        from scipy.sparse import csr_matrix
        from ..kpmtk.densitymatrix_kpm import get_dm_kpm_sparse
        v = {d: csr_matrix(m) for (d, m) in v.items()}
    h1 = h0.copy()
    h1.nk = nk
    if mf is None:
        try:
            if load_mf:
                mf = inout.load(mf_file)
                if not mf_matches_hamiltonian(h0,mf):
                    raise ValueError("cached MF.pkl shape does not match this Hamiltonian")
            else: raise
        except:
            if sparse: mf = sparsemeanfield.random_guess(v,h1)
            else: mf = random_hermitian_guess(v,h1.intra.shape)
    elif type(mf) == str:
        from ..meanfield import guess
        mf = guess(h0, mode=mf)
    else: pass
    mf = obj2mf(mf)
    fs.rmfile("STOP")
    hop0 = hamiltonian2dict(h1)
    if sparse: # the density-matrix entries the mean field reads, once
        norb = h1.intra.shape[0]
        needed = sparsemeanfield.needed_entries(v,norb,has_eh=h0.has_eh)
        def get_dm(h, trace=False):
            out = get_dm_kpm_sparse(h, needed, nk=nk, scale=scale, npol=npol,
                    ne=ne, cores=cores, T=T, kpm_prec=kpm_prec, trace=trace)
            dm = out[0] if trace else out
            for d in list(v)+[tuple(-x for x in d) for d in v]: # every one
                if d not in dm: dm[d] = csr_matrix((norb,norb),dtype=np.complex128)
            return out
        def mf_from_dm(dm):
            return sparsemeanfield.get_mf(v, dm, compute_cross=compute_cross,
                    compute_dd=compute_dd, has_eh=h0.has_eh,
                    compute_anomalous=compute_anomalous,
                    compute_normal=compute_normal)
    else:
        def get_dm(h, trace=False):
            return get_dm_kpm(h, v, nk=nk, scale=scale, npol=npol, ne=ne,
                    cores=cores, T=T, kpm_prec=kpm_prec, trace=trace)
        def mf_from_dm(dm):
            return get_mf(v, dm, compute_cross=compute_cross,
                    compute_dd=compute_dd, has_eh=h0.has_eh,
                    compute_anomalous=compute_anomalous,
                    compute_normal=compute_normal)
    def f(mf, h=h1):
        # Shallow copies, not deepcopy -- see the same two changes in
        # scftk/densitydensity.py's generic_densitydensity for the full
        # reasoning: `mf` is never mutated here, and set_hoppings rebinds
        # h.intra/h.hopping to fresh matrices on the next line, so nothing
        # deep-copied would have survived. Each iteration still gets a
        # distinct Hamiltonian object.
        mf0 = dict(mf) if isinstance(mf,dict) else mf
        h = copy(h1)
        h.data = dict(h1.data)
        hop = update_hamiltonian(hop0, mf)
        set_hoppings(h, hop)
        if callback_h is not None: h = callback_h(h)
        if fermi is not None:
            h = fermi.shift(h)
            if verbose>1: print("Fermi energy",h.fermi)
        t0 = time.perf_counter()
        if fermi is not None and fermi.wants_trace(h):
            dm, trace = get_dm(h, trace=True)
            fermi.update(h, trace)
        else:
            dm = get_dm(h)
            if fermi is not None: fermi.update(h, None)
        if callback_dm is not None: dm = callback_dm(dm)
        t1 = time.perf_counter()
        mf = mf_from_dm(dm)
        if callback_mf is not None: mf = callback_mf(mf)
        t2 = time.perf_counter()
        if verbose>1:
            print("Time in KPM density matrix = ",t1-t0)
            print("Time in the normal term = ",t2-t1)
        scf = SCF()
        scf.hamiltonian = h
        scf.hamiltonian.V = v
        scf.hamiltonian0 = h0
        scf.mf = mf
        if os.path.exists("STOP"): scf.mf = mf0
        scf.dm = dm
        scf.v = v
        scf.tol = maxerror
        return scf
    ite = 0
    while True:
        scf = f(mf)
        mfnew = scf.mf
        diff = diff_mf(mfnew, mf)
        # the iterates are at the requested filling only once the Fermi
        # level stops lagging behind the mean field, see LaggedFermi
        if fermi is not None: diff = max(diff, fermi.error)
        mf = mix_mf(mfnew, mf, mix=mix)
        if callback_mf is not None: mf = callback_mf(mf)
        if verbose>0: print("ERROR in the KPM SCF cycle",ite,diff)
        if diff<maxerror:
            scf = f(mfnew)
            scf.converged = True
            if write: inout.save(scf.mf, mf_file) # MF.pkl, unless write=False
            return scf
        if maxite is not None and ite>=maxite:
            scf.converged = False
            print("No convergence has been reached in",maxite,"iterations, stopping")
            return scf
        ite += 1


def densitydensity_kpm(h, filling=0.5, mu=None, verbose=0, nk=DEFAULT_NK,
        scale=None, npol=DEFAULT_NPOL, ne=None, cores=None, kpm_prec=None,
        **kwargs):
    """KPM analogue of scftk.densitydensity.densitydensity"""
    from ..checkclass import is_iterable
    if is_iterable(filling): # see VJinteraction's docstring
        raise NotImplementedError("A per-site (array) filling is only "
                "supported by VJinteraction (h.get_mean_field_hamiltonian "
                "with integration=\"ed\") for a spinful Hamiltonian; "
                "the KPM density-density engine (Vinteraction_kpm) "
                "takes a single scalar filling, got %r" % (filling,))
    from .densitydensity import electron_dimension, require_hermitian
    require_hermitian(h,"the KPM mean field (Vinteraction_kpm)")
    from ..kpmtk.densitymatrix_kpm import LaggedFermi
    h = h.get_multicell()
    if not h.is_sparse: h = h.get_dense() # a sparse one stays sparse
    if mu is None:
        # the KPM Fermi level, from the trace of the previous iteration's
        # recursion (LaggedFermi), so this never diagonalizes anything,
        # at T, not T=0: the density matrix this Fermi level feeds is built
        # with Fermi-Dirac occupations, so locating it with a step count
        # makes the converged electron count drift away from `filling` as
        # T grows
        fermi = LaggedFermi(filling, nk=nk, scale=scale, npol=npol, ne=ne,
                cores=cores, T=kwargs.get("T",1e-7), kpm_prec=kpm_prec)
        callback_h = None
    else:
        fermi = None
        def callback_h(h):
            h.shift_fermi(-mu)
            return h
    scf = generic_densitydensity_kpm(h, callback_h=callback_h, fermi=fermi,
            verbose=verbose, nk=nk, scale=scale, npol=npol, ne=ne,
            cores=cores, kpm_prec=kpm_prec, **kwargs)
    h = scf.hamiltonian
    if h.is_sparse: # Tr(H rho) from KPM, rather than diagonalizing
        from ..kpmtk.densitymatrix_kpm import get_band_energy_kpm
        from . import sparsemeanfield as dc # its double countings, below
        # at T=0, as the sum of the occupied levels the dense engine takes
        etot = get_band_energy_kpm(h, nk=h.nk, scale=scale, npol=npol,
                ne=ne, cores=cores, T=0., kpm_prec=kpm_prec)
    else:
        from . import densitydensity as dc
        etot = h.get_total_energy(nk=h.nk)
    # electron_dimension, not h.intra.shape[0] -- see the identical
    # comment in densitydensity.densitydensity
    if mu is None: etot += h.fermi*electron_dimension(h)*filling
    # get_dc_energy assumes dm's shape matches v's, which is never
    # Nambu-doubled even when h (hence scf.dm) is BdG -- see the identical
    # fix/comment in densitydensity.densitydensity for why the electron
    # sector must be extracted first for a BdG h.
    dm_dc = scf.dm
    if h.has_eh:
        from .. import superconductivity
        dm_dc = {key: superconductivity.get_eh_sector(m,i=0,j=0)
                for (key,m) in scf.dm.items()}
    etot += dc.get_dc_energy(scf.v, dm_dc)
    if h.has_eh and kwargs.get("compute_anomalous",True):
        # the pairing part, see the identical step in
        # densitydensity.densitydensity
        mf = dc.get_mf(scf.v, scf.dm, has_eh=True)
        if h.is_sparse: etot += dc.get_dc_energy_anomalous(mf, scf.dm)
        else:
            from .superscf import get_dc_energy_anomalous
            etot += get_dc_energy_anomalous(mf, scf.dm)
    etot = etot.real
    scf.total_energy = etot
    if verbose>1:
        print("##################")
        print("Total energy (KPM)",etot)
        print("##################")
    return scf


def hubbard_kpm(h, U=1.0, constrains=[], **kwargs):
    """KPM analogue of scftk.densitydensity.hubbard"""
    from .densitydensity import obj2geometryarray, reject_spinless_U
    h = h.copy()
    h.turn_multicell()
    U = obj2geometryarray(U, h.geometry)
    reject_spinless_U(h, U) # the same refusal as Vinteraction_kpm
    n = len(h.geometry.r)
    i = np.arange(n)
    U = np.asarray(U, dtype=np.complex128)
    if h.has_spin: rows, cols, dim = 2*i, 2*i+1, 2*n # U on (2i,2i+1)
    else: rows, cols, dim = i, i, n
    if h.is_sparse:
        from scipy.sparse import csr_matrix
        zero = csr_matrix((U, (rows, cols)), shape=(dim,dim))
    else:
        zero = np.zeros((dim,dim),dtype=np.complex128)
        zero[rows,cols] = U
    v = dict()
    v[(0,0,0)] = zero
    callback_mf = _constrains_callback(h, constrains)
    if h.has_spin:
        return densitydensity_kpm(h, v=v, callback_mf=callback_mf, **kwargs)
    else:
        return densitydensity_kpm(h, v=v, compute_cross=False,
                callback_mf=callback_mf, **kwargs)


def Vinteraction_kpm(h, V1=0.0, V2=0.0, V3=0.0, U=0.0, constrains=[],
        Vr=None, rcut=None, **kwargs):
    """KPM analogue of scftk.densitydensity.Vinteraction: mean
    field with density-density interactions (U onsite, V1/V2/V3 first/
    second/third neighbor), computed via sparse KPM instead of exact
    diagonalization -- see kpmtk.densitymatrix_kpm.get_dm_kpm.

    Performance: the density matrix is one block Chebyshev recursion over
    every orbital and k-point, with numba on the CPU
    (kpmtk/pairmomentsnumba.py) and with jax on the GPU when
    pyqula.gpu.set_gpu(True) (kpmtk/pairmomentsjax.py), in single
    precision there unless kpm_prec says otherwise, and the Fermi level of
    each iteration comes from the trace of the previous one
    (kpmtk.densitymatrix_kpm.LaggedFermi), so an iteration is one
    recursion. An iteration on honeycomb islands at npol=200 takes 1.1 s
    on six desktop cores and 0.37 s on a consumer RTX A2000 at 1728
    orbitals, where exact diagonalization takes 4.1 s, and 5.7 s and
    1.8 s for a 3456-orbital Nambu island, which still searches the Fermi
    level of its electron-only Hamiltonian every iteration, against 36 s;
    with the jax engine on both backends and a Fermi search of its own
    these were 6.2 s and 20 s on the CPU and 0.8 s and 1.9 s on the card
    (future_development/sparse_kpm_mean_field.md). For a small cell on a
    k-mesh exact diagonalization stays the faster choice.

    A sparse Hamiltonian (built with is_sparse=True) goes through the
    sparse engine of scftk/sparsemeanfield.py, which holds the interaction,
    the density matrix and the mean field as sparse matrices with only the
    entries the interaction couples, so that the memory is linear in the
    number of orbitals, above working buffers that are fixed up to 65,536
    orbitals and take 2 kB per orbital beyond (kpmtk/pairmomentsnumba.py);
    it gives the dense engine's mean field to roundoff. The time per
    iteration still grows as the square of the number of orbitals. Its
    total energy is Tr(H rho) from the KPM density matrix rather than the
    sum of the diagonalized occupied levels, which it reaches as npol
    grows, and Vr on a finite system needs an explicit rcut
    (future_development/sparse_kpm_mean_field.md)."""
    from .densitydensity import (obj2geometryarray, reject_legacy_kwargs,
            reject_spinless_U)
    kwargs = reject_legacy_kwargs(kwargs) # the same refusals as Vinteraction
    h = h.get_multicell()
    if h.is_sparse: # the interaction as sparse matrices, from a KD-tree
        from .sparsemeanfield import interaction
        U = obj2geometryarray(U, h.geometry)
        reject_spinless_U(h, U)
        v = interaction(h, V1=V1, V2=V2, V3=V3, U=U, Vr=Vr, rcut=rcut)
        return densitydensity_kpm(h, v=v,
                callback_mf=_constrains_callback(h, constrains), **kwargs)
    h = h.get_dense()
    nd = h.geometry.neighbor_distances()
    from .. import specialhopping
    mgenerator = specialhopping.distance_hopping_matrix([V1/2.,V2/2.,V3/2.],nd[0:3])
    hv = h.geometry.get_hamiltonian(has_spin=False,is_multicell=True,
            mgenerator=mgenerator)
    v = hv.get_hopping_dict()
    if Vr is not None: # every pair within rcut, halved as in Vinteraction
        from .densitydensity import add_pair_interaction
        add_pair_interaction(v,h.geometry,Vr,rcut=rcut)
    U = obj2geometryarray(U, h.geometry)
    reject_spinless_U(h, U)
    if h.has_spin:
        for d in v:
            m = v[d] ; n = m.shape[0]
            m1 = np.zeros((2*n,2*n),dtype=np.complex128)
            for i in range(n):
              for j in range(n):
                  m1[2*i,2*j] = m[i,j]
                  m1[2*i+1,2*j] = m[i,j]
                  m1[2*i,2*j+1] = m[i,j]
                  m1[2*i+1,2*j+1] = m[i,j]
            v[d] = m1
        n = len(h.geometry.r)
        for i in range(n):
            v[(0,0,0)][2*i,2*i+1] += U[i]/2.
            v[(0,0,0)][2*i+1,2*i] += U[i]/2.
    return densitydensity_kpm(h, v=v,
            callback_mf=_constrains_callback(h, constrains), **kwargs)


def _constrains_callback(h, constrains):
    """The callback that enforces the constrains on the mean field, or None"""
    if not constrains: return None
    from . import mfconstrains
    def callback_mf(mf):
        return mfconstrains.enforce_constrains(mf, h, constrains)
    return callback_mf
