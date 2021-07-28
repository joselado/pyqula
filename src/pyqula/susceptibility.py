import numpy as np
from . import parallel
from . import algebra
from . import magnetism

def dominant_correlation(h0,filling=0.5,dm=1e-1,
        write=False,**kwargs):
    """Compute the dominant magnetic correlator"""
    h = h0.copy() # copy hamiltonian
    h.turn_dense()
    h.set_filling(filling) # set the desired filling
    if not h.has_spin: raise # only for spinful
    n = len(h.geometry.r) # number of sites
    def getrow(ii): # compute a row of the susceptibility matrix
        hi = h.copy() # copy Hamiltonian
        ms = [[0.,0.,0.] for j in range(n)]
        ms[ii] = [0.,0.,dm] # perturbation
        magnetism.add_magnetism(hi,ms) # add local exchange fields
        hi.set_filling(filling,**kwargs) # set the desired filling
        print("Done correlator site",ii)
        out = hi.compute_vev("sz",delta=dm/2.,**kwargs) # compute magnetization
        return -np.array(out)/dm # return output
    out = parallel.pcall(getrow,range(n)) # perform all the computations
    from . import rkky
#    out = rkky.rkky0d(h)
    out = np.array(out) # as array
#    print("Next")
    print(np.max(np.abs(out-out.T)),np.mean(np.abs(out)))
    out = (out + out.T)/2.0 # make Hermitian
    (es,vs) = algebra.eigh(out) # diagonalize
    print(es)
    chi = vs.T[vs.shape[0]-1] # biggest eigenvector
    if write: h.geometry.write_profile(chi,name="CHI_PROFILE.OUT")
    return chi








#def magnetic_chi_ij_perturation_theory(h0,filling=0.5,dm=1e-1,write=False)
