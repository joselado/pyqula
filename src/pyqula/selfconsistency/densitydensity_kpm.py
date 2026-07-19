# KPM-based (Chebyshev, sparse) alternative to selfconsistency/densitydensity.py
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
import numpy as np
import os
import time
from copy import deepcopy

from .. import inout
from ..multihopping import MultiHopping
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
        compute_anomalous=True, compute_normal=True, maxite=None,
        T=1e-7, callback_h=None,
        scale=None, npol=DEFAULT_NPOL, ne=None, cores=None, **kwargs):
    """KPM analogue of selfconsistency.densitydensity.generic_densitydensity.
    Only the "plain" mixing solver is implemented (the alternate
    root-finding solvers there are not KPM-specific and are not needed for
    this backend)."""
    from .densitydensity import (get_mf, mix_mf, diff_mf, update_hamiltonian,
            hamiltonian2dict, set_hoppings, SCF)
    from .mfconstrains import obj2mf
    h1 = h0.copy()
    h1.nk = nk
    if mf is None:
        try:
            if load_mf:
                mf = inout.load(mf_file)
                MultiHopping(h0.get_dict()) + MultiHopping(mf)
            else: raise
        except:
            mf = dict()
            for d in v: mf[d] = np.exp(1j*np.random.random(h1.intra.shape))
            mf[(0,0,0)] = mf[(0,0,0)] + mf[(0,0,0)].T.conjugate()
    elif type(mf) == str:
        from ..meanfield import guess
        mf = guess(h0, mode=mf)
    else: pass
    mf = obj2mf(mf)
    os.system("rm -f STOP")
    hop0 = hamiltonian2dict(h1)
    def f(mf, h=h1):
        mf0 = deepcopy(mf)
        h = h1.copy()
        hop = update_hamiltonian(hop0, mf)
        set_hoppings(h, hop)
        if callback_h is not None: h = callback_h(h)
        t0 = time.perf_counter()
        dm = get_dm_kpm(h, v, nk=nk, scale=scale, npol=npol, ne=ne,
                cores=cores, T=T)
        if callback_dm is not None: dm = callback_dm(dm)
        t1 = time.perf_counter()
        mf = get_mf(v, dm, compute_cross=compute_cross,
                compute_dd=compute_dd, has_eh=h0.has_eh,
                compute_anomalous=compute_anomalous,
                compute_normal=compute_normal)
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
        mf = mix_mf(mfnew, mf, mix=mix)
        if callback_mf is not None: mf = callback_mf(mf)
        if verbose>0: print("ERROR in the KPM SCF cycle",ite,diff)
        if diff<maxerror:
            scf = f(mfnew)
            scf.converged = True
            inout.save(scf.mf, mf_file)
            return scf
        if maxite is not None and ite>=maxite:
            scf.converged = False
            print("No convergence has been reached in",maxite,"iterations, stopping")
            return scf
        ite += 1


def densitydensity_kpm(h, filling=0.5, mu=None, verbose=0, nk=DEFAULT_NK,
        scale=None, npol=DEFAULT_NPOL, ne=None, cores=None, **kwargs):
    """KPM analogue of selfconsistency.densitydensity.densitydensity"""
    from .densitydensity import get_dc_energy
    from ..kpmtk.densitymatrix_kpm import get_fermi4filling_kpm
    if h.has_eh:
        if not h.has_spin: return NotImplemented  # only for spinful, as in ED
    h = h.get_multicell()
    h = h.get_dense()
    def callback_h(h):
        if mu is None:
            # KPM Fermi-energy search (get_fermi4filling_kpm) instead of
            # h.get_fermi4filling, so this never diagonalizes anything --
            # see that function's docstring
            fermi = get_fermi4filling_kpm(h, filling, nk=nk, scale=scale,
                    npol=npol, ne=ne, cores=cores)
            if verbose>1: print("Fermi energy",fermi)
            h.fermi = fermi
            h.shift_fermi(-fermi)
        else: h.shift_fermi(-mu)
        return h
    scf = generic_densitydensity_kpm(h, callback_h=callback_h,
            verbose=verbose, nk=nk, scale=scale, npol=npol, ne=ne,
            cores=cores, **kwargs)
    h = scf.hamiltonian
    etot = h.get_total_energy(nk=h.nk)
    if mu is None: etot += h.fermi*h.intra.shape[0]*filling
    etot += get_dc_energy(scf.v, scf.dm)
    etot = etot.real
    scf.total_energy = etot
    if verbose>1:
        print("##################")
        print("Total energy (KPM)",etot)
        print("##################")
    return scf


def hubbard_kpm(h, U=1.0, constrains=[], **kwargs):
    """KPM analogue of selfconsistency.densitydensity.hubbard"""
    from .densitydensity import obj2geometryarray
    h = h.copy()
    h.turn_multicell()
    U = obj2geometryarray(U, h.geometry)
    n = len(h.geometry.r)
    if h.has_spin:
        zero = np.zeros((2*n,2*n),dtype=np.complex128)
        for i in range(n): zero[2*i,2*i+1] = U[i]
    else:
        zero = np.zeros((n,n),dtype=np.complex128)
        for i in range(n): zero[i,i] = U[i]
    v = dict()
    v[(0,0,0)] = zero
    callback_mf = None
    if constrains:
        from . import mfconstrains
        def callback_mf(mf):
            return mfconstrains.enforce_constrains(mf, h, constrains)
    if h.has_spin:
        return densitydensity_kpm(h, v=v, callback_mf=callback_mf, **kwargs)
    else:
        return densitydensity_kpm(h, v=v, compute_cross=False,
                callback_mf=callback_mf, **kwargs)


def Vinteraction_kpm(h, V1=0.0, V2=0.0, V3=0.0, U=0.0, constrains=[],
        Vr=None, **kwargs):
    """KPM analogue of selfconsistency.densitydensity.Vinteraction: mean
    field with density-density interactions (U onsite, V1/V2/V3 first/
    second/third neighbor), computed via sparse KPM instead of exact
    diagonalization -- see kpmtk.densitymatrix_kpm.get_dm_kpm."""
    from .densitydensity import obj2geometryarray
    h = h.get_multicell()
    h = h.get_dense()
    nd = h.geometry.neighbor_distances()
    from .. import specialhopping
    mgenerator = specialhopping.distance_hopping_matrix([V1/2.,V2/2.,V3/2.],nd[0:3])
    hv = h.geometry.get_hamiltonian(has_spin=False,is_multicell=True,
            mgenerator=mgenerator)
    if Vr is not None:
        hv1 = h.geometry.get_hamiltonian(has_spin=False,is_multicell=True,
                tij=Vr)
        hv = hv + hv1
    v = hv.get_hopping_dict()
    U = obj2geometryarray(U, h.geometry)
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
    callback_mf = None
    if constrains:
        from . import mfconstrains
        def callback_mf(mf):
            return mfconstrains.enforce_constrains(mf, h, constrains)
    return densitydensity_kpm(h, v=v, callback_mf=callback_mf, **kwargs)
