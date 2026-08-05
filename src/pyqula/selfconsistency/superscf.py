import numpy as np
from numba import jit



def get_mf_anomalous(v,dm):
    """Compute the anomalous part of the mean-field,
    input in the anomalous density matrix in Nambu basis"""
    zero = dm[(0,0,0)]*0. # zero
    mf = dict()
    for d in v: mf[d] = zero.copy()  # initialize
    for d in v: # loop over directions
        d2 = tuple(-np.array(d))
        m = anomalous_term_ij(2*v[d],dm[d2]) # get matrix
        mf[d] = mf[d] + m # add normal term
    return mf



def anomalous_term_ij(v,dm):
    """Return the anomalous term of the mean-field,
    assuming a nmabu basis"""
    # we will assume that v contains up,down in alternating order
    n = dm.shape[0] # number of spinless sites
    out = np.zeros((n,n),dtype=np.complex128)
    return anomalous_term_ij_jit(v,dm,out)



# this anomalous term enforces even superconductivity

@jit(nopython=True,cache=True)
def anomalous_term_ij_jit(v,dm,out):
    ns = len(dm)//2 # number of spinless sites
    for i in range(ns): # loop over spinless sites
        for j in range(ns): # loop over spinless sites
          out[2*i,2*j] = v[2*i,2*j+1]*dm[2*j,2*i]  # down,up
          out[2*i,2*j+1] = v[2*i,2*j]*dm[2*j+1,2*i]  # up,up
          out[2*i+1,2*j+1] = v[2*i+1,2*j]*dm[2*j+1,2*i+1]  # up,down
          out[2*i+1,2*j] = v[2*i+1,2*j+1]*dm[2*j,2*i+1]  # down,down
    return out



def enforce_eh_symmetry_anomalous(d01):
    """Enforce electron-hole symmetry in the two sectors"""
    d01 = enforce_eh_symmetry_anomalous_sector(d01)
    d10 = enforce_eh_from_sector(d01)
    return d01,d10



def enforce_eh_from_sector(d):
    """Given one sector of the Hamiltonian, return the other one"""
    out = dict() # dictionary
    for key in d:
        m = d[key] # one key
        o01 = m*0.0
        key2 = tuple(-np.array(key)) # the opposite
        out[key2] = enforce_eh_from_sector_jit(m,o01)
    return out # return


@jit(nopython=True,cache=True)
def enforce_eh_from_sector_jit(d,o):
    """Given the ee sector, return the hh sector"""
    return np.conjugate(d.T) # hermitian conjugate


def enforce_eh_symmetry_anomalous_sector(d01):
    """Enforce electron-hole symmetry in one of the sectors"""
    out01 = dict()
    for key in d01: # loop over keys
        key2 = tuple(-np.array(key)) # minus one
        o01 = d01[key]*0.0
        o01 = enforce_eh_symmetry_anomalous_jit(d01[key],d01[key2],o01)
        out01[key] = o01
    return out01



@jit(nopython=True,cache=True)
def enforce_eh_symmetry_anomalous_jit(d01,d10,o01):
    """Enforce electron-hole symmetry"""
    ns = len(d01)//2 # number of spinless sites
    for i in range(ns): # loop
        for j in range(ns): # loop
            # enforce the up|up sector
            o01[2*i,2*j+1] = d01[2*i,2*j+1] - d10[2*j,2*i+1]
            # enforce the down|down sector1
            o01[2*i+1,2*j] = d01[2*i+1,2*j] - d10[2*j+1,2*i]
            # enforce the up|down sector1 (beware of the minus sign)
            o01[2*i,2*j] = d01[2*i,2*j] + d10[2*j+1,2*i+1]
            o01[2*i+1,2*j+1] = d01[2*i+1,2*j+1] + d10[2*j,2*i]
    return o01/2.



def get_mf_bdg(v,dm,compute_anomalous=True,compute_normal=True,**kwargs):
    """Get the full BdG (Nambu) mean-field matrix.

    v: interaction matrix dict, spin-orbital sized (NOT Nambu-doubled) --
    the same convention densitydensity.get_mf_normal takes.
    dm: Nambu-doubled density matrix dict, as returned by diagonalizing a
    BdG Hamiltonian (h.has_eh=True).

    Same v/dm-in, mf-out contract as get_mf_normal(v,dm) -- this is the
    has_eh=True counterpart called by densitydensity.get_mf -- but
    internally it has to combine two genuinely different Wick-contraction
    topologies of the same density-density interaction: the normal
    (Hartree+Fock) decoupling of the electron sector <c^+c> (get_mf_normal,
    reused as-is) and the anomalous (pairing) decoupling of the <cc> sector
    (get_mf_anomalous, this module). These are not the same formula run on
    different data -- compare anomalous_term_ij_jit's index gymnastics
    against normal_term_ij_jit's -- so neither can be dropped in favor of
    the other; this function's job is only to package the two into one
    self-contained BdG decoupling step."""
    from .. import superconductivity
    from ..multihopping import MultiHopping
    from .densitydensity import get_mf_normal
    dme = dict() # electron sector of the density matrix, one per direction
    dma10 = dict() # anomalous (electron-hole) sector
    for key in dm:
        m = dm[key]
        dme[key] = superconductivity.get_eh_sector(m,i=0,j=0)
        dma10[key] = superconductivity.get_eh_sector(m,i=0,j=1)
    mfe = get_mf_normal(v,dme,**kwargs) # electron part of the mean field
    mfa01 = get_mf_anomalous(v,dma10) # anomalous part
    mfa01,mfa10 = enforce_eh_symmetry_anomalous(mfa01)
    mf = dict()
    for key in v:
        if not compute_normal: mfe[key] = mfe[key]*0.0
        if compute_anomalous:
            mf[key] = superconductivity.build_nambu_matrix(mfe[key],
                    c12=mfa10[key],c21=mfa01[key])
        else:
            mf[key] = superconductivity.build_nambu_matrix(mfe[key])
    if not MultiHopping(mf).is_hermitian(): # sanity check on the result
        raise ValueError("Non-Hermitian BdG mean field:\n%s" %
                np.round(mf[(0,0,0)],2))
    return mf # return mean field matrix

