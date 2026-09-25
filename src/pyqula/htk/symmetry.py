import numpy as np
from scipy.sparse import diags
from ..superconductivity import time_reversal


def has_time_reversal_symmetry(h,tol=1e-7):
    """Check whether a Hamiltonian is time-reversal symmetric

    The time-reversal operator is complex conjugation K for a spinless
    Hamiltonian and i sigma_y K for a spinful one. A Nambu Hamiltonian
    uses the basis (c_up, c_dn, c_dn^dag, -c_up^dag), in which the hole
    spinor transforms like the electron one, so there it is
    tau_0 i sigma_y K, which is time_reversal applied to the whole matrix.

    A superconductor counts as time-reversal symmetric when this holds up
    to a global phase of the pairing, since Delta = e^{i phi} Delta_0 with
    a symmetric Delta_0 is the same state in another gauge. That phase is
    removed first, see _remove_pairing_phase."""
    if h.has_eh:
        return _tr_residual(_remove_pairing_phase(h),time_reversal)<tol
    if h.has_spin: return _tr_residual(h,time_reversal)<tol
    return _tr_residual(h,np.conjugate)<tol


def _tr_residual(h,f):
    """Squared norm of H - T H T^-1, with T H T^-1 given by f acting on
    every hopping matrix"""
    h1 = h.copy()
    h1.modify_hamiltonian_matrices(f) # apply time reversal
    dd = h.get_multihopping() - h1.get_multihopping() # difference
    return dd.dot(dd).real # scalar product


def _remove_pairing_phase(h):
    """Return a copy of a Nambu Hamiltonian with the global phase of its
    pairing gauged away.

    If Delta = e^{i phi} Delta_0 with Delta_0 time-reversal symmetric,
    then T Delta T^-1 = e^{-2i phi} Delta, so the projection of the
    time-reversed electron-hole block onto the original one gives
    z = e^{-2i phi}. The gauge transformation
    U = diag(w on electrons, w^* on holes) with w^4 = z multiplies the
    electron-hole block by w^2 = e^{-i phi}, leaving Delta_0 (either root
    does, up to a sign). A pairing that is not symmetric in any gauge gives
    a z that does not make T Delta T^-1 = z Delta hold, and the check on the
    returned copy then fails, as it should."""
    n = h.intra.shape[0] # dimension of the Nambu matrices
    ise = np.array([i%4<2 for i in range(n)]) # electron components
    pe = diags(ise.astype(float)) # projector on the electrons
    ph = diags((~ise).astype(float)) # projector on the holes
    ha = h.copy()
    ha.modify_hamiltonian_matrices(lambda m: pe@m@ph) # pairing only
    hta = h.copy()
    hta.modify_hamiltonian_matrices(lambda m: pe@time_reversal(m)@ph)
    a = ha.get_multihopping() # electron-hole block
    ta = hta.get_multihopping() # its time-reversed partner
    a2 = a.dot(a).real
    if a2<1e-14: return h # no pairing, nothing to gauge away
    z = a.dot(ta)/a2 # e^{-2i phi} for a pairing with a global phase
    if np.abs(z)<1e-7: return h # not symmetric in any gauge
    w = np.sqrt(np.sqrt(z/np.abs(z))) # w^2 = e^{-i phi}
    u = diags(np.where(ise,w,np.conjugate(w))) # gauge transformation
    ud = diags(np.where(ise,np.conjugate(w),w)) # its inverse
    h1 = h.copy()
    h1.modify_hamiltonian_matrices(lambda m: u@m@ud)
    return h1

