import numpy as np
from .. import klist
from .. import algebra
from ..ldostk.ldoswaves import ldos_waves_jit
from ..htk.matrixcomponent import full2profile


def _eigenstates_for_energy_window(hk,num_waves,energies,delta,
        margin=5.0,gap_tol=1e-6,growth=1.5,max_tries=10):
    """Diagonalize hk for (about) num_waves eigenstates nearest the
    requested energies, growing the count until two conditions both
    hold, retrying with more states whenever either fails:

    1. The returned eigenvalue range actually reaches past every
       requested energy by margin*delta on each side. Without this, a
       fixed num_waves picked around the mean of `energies` could miss
       states close to whichever requested energy sits farthest from
       that mean, silently dropping their (still non-negligible,
       Lorentzian-tailed) contribution to the LDOS at that energy --
       especially with num_waves chosen small for a widely-spaced
       multi-energy sweep.
    2. The cutoff does not fall in the middle of a degenerate manifold.
       This matters because the LDOS built from a partial cut of a
       degenerate eigenspace is *not* basis-independent -- only summing
       |psi|^2 over the *complete* manifold is -- so an arbitrary
       partial cut makes the result depend on ARPACK's Krylov starting
       vector, an artifact rather than physics. Symmetric lattices
       (e.g. honeycomb) routinely have large exact degeneracies at
       high-symmetry k-points (e.g. a 6x6 honeycomb supercell has a
       15-fold degenerate level at its own Gamma point), so this is not
       a corner case in practice.

    If growth still hasn't satisfied both conditions after max_tries
    (e.g. a near-flat band with degeneracy spanning a large fraction of
    the spectrum), the best-effort result is returned with a warning --
    full diagonalization is the only way to resolve that case exactly."""
    n = hk.shape[0]
    e0 = (np.min(energies)+np.max(energies))/2.
    target_lo = np.min(energies)-margin*delta
    target_hi = np.max(energies)+margin*delta
    k = min(num_waves,n-1)
    eig = eigvec = None
    for _ in range(max_tries):
        kk = min(k+1,n) # one extra state, to check the boundary
        eig,eigvec = algebra.smalleig(hk,numw=kk,e0=e0,evecs=True)
        order = np.argsort(np.abs(eig-e0))
        eig,eigvec = eig[order],eigvec[order]
        if kk>=n: return eig,eigvec # full spectrum: nothing left to miss
        window_ok = np.min(eig[:k])<=target_lo and np.max(eig[:k])>=target_hi
        boundary_ok = abs(eig[k]-eig[k-1])>gap_tol
        if window_ok and boundary_ok: return eig[:k],eigvec[:k]
        k = min(int(np.ceil(k*growth))+1,n-1) # grow and retry
    print("Warning: could not both cover the requested energy window and",
            "clear a degenerate boundary after",max_tries,"tries at",
            "num_waves =",k,"-- LDOS may be inaccurate or slightly",
            "starting-vector-dependent; increase num_waves or use full",
            "diagonalization for this energy/system")
    return eig[:k],eigvec[:k]


def real_space_ldos(h,energies,delta=0.05,num_waves=20,nk=2,margin=5.0,
        gap_tol=1e-6,**kwargs):
    """Compute the real-space LDOS map of h at each energy in energies,
    using ARPACK partial diagonalization (only num_waves-ish eigenstates
    covering the requested energies, grown as needed -- see
    _eigenstates_for_energy_window) instead of the full spectrum, which
    is what makes this tractable for the large supercells a real-space
    impurity calculation needs. h is expected to already be sparse (see
    qpitk.impurity.build_impurity_hamiltonian).

    All requested energies are evaluated from the same diagonalization
    per k-point (Lorentzian-reweighting one eigenbasis for every
    energy), mirroring the existing ldos.ldosmap's per-k design, rather
    than re-diagonalizing once per energy.

    Returns (r,ldos_r): r is the (nsites,3) array of real-space
    positions, ldos_r is (len(energies),nsites)."""
    hkgen = h.get_hk_gen()
    ks = klist.kmesh(h.dimensionality,nk=nk)
    energies = np.array(energies,dtype=np.float64)
    n = h.intra.shape[0]
    ds = [] # one (nenergies,nsites_full) array per k
    for k in ks:
        hk = hkgen(k)
        eig,eigvec = _eigenstates_for_energy_window(hk,num_waves,energies,delta,
                margin=margin,gap_tol=gap_tol,**kwargs)
        weights = np.ones(len(eig))
        v2s = (np.conjugate(eigvec)*eigvec).real
        d0 = np.zeros((len(energies),n))
        ds.append(ldos_waves_jit(energies,eigvec,eig,weights,v2s,d0,delta))
    d = np.mean(ds,axis=0) # average over k, shape (nenergies,nsites_full)
    d = np.array([full2profile(h,di) for di in d]) # resum spin/e-h
    return h.geometry.r,d
