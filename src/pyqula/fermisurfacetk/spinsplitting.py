import numpy as np
from .. import algebra

def average_spin_splitting(h,nk=20):
    """Compute the average spin splitting in the BZ"""
    # this assumes that spin up and down are good quantum numbers
    if not h.has_spin: raise
    hup = h.copy() ; hup.remove_spin(channel="up") 
    hdn = h.copy() ; hdn.remove_spin(channel="dn") 
    hkup = hup.get_hk_gen() # get generator
    hkdn = hdn.get_hk_gen() # get generator
    from ..klist import kmesh
    ks = kmesh(h.geometry.dimensionality,nk=nk)
    def am(k):
        eup = algebra.eigvalsh(hkup(k)) # eigenvalues for up
        edn = algebra.eigvalsh(hkdn(k)) # eigenvalues for dn
        de = (eup-edn)**2 # square difference
        ea = (eup + edn)/2. # average
        return np.sum(np.sqrt(de)) # square root
    out = np.mean([am(k) for k in ks]) # average altermagnetism
    return out # return result





def spin_splitting_density(h,nk=20,energies=None,delta=1e-2):
    """Compute the average spin splitting in the BZ"""
    # this assumes that spin up and down are good quantum numbers
    if energies is None: energies = np.linspace(-3.0,3.0,400)
    if not h.has_spin: raise
    hup = h.copy() ; hup.remove_spin(channel="up")
    hdn = h.copy() ; hdn.remove_spin(channel="dn")
    hkup = hup.get_hk_gen() # get generator
    hkdn = hdn.get_hk_gen() # get generator
    from ..klist import kmesh
    ks = kmesh(h.geometry.dimensionality,nk=nk)
    from ..dos import calculate_dos
    def am(k):
        eup = algebra.eigvalsh(hkup(k)) # eigenvalues for up
        edn = algebra.eigvalsh(hkdn(k)) # eigenvalues for dn
        de = (eup-edn)**2 # square difference
        ea = (eup + edn)/2. # average
        return calculate_dos(ea,energies,delta,w=de)
#        return np.sum(np.sqrt(de)) # square root
    out = np.mean([am(k) for k in ks],axis=0) # average altermagnetism
    return energies,out # return result




