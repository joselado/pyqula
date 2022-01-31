import numpy as np
from .. import algebra
from numba import jit

def rkky(h,delta=None,R=np.array([0.,0.,0.]),nk=100):
    """Compute the RKKY interaction for the ground state
    - R location of the unit cell
    - delta analytic continuation
    """
    # R should be smaller than nk
    if delta is None: delta = 1./nk
    R = np.array(R)
    R2 = np.sqrt(R.dot(R)) # square root
    if R2>nk/5: 
        print("Warnign in RKKY, kmesh is too small, redefining")
        nk = int(R2)
    es,ws,ks = h.get_eigenvectors(kpoints=True,nk=nk) # get eigenvectors
    phis = np.array([h.geometry.bloch_phase(R,k) for k in ks]) # phase
    fs = (np.sign(es)+1.)/2. # occupation
    fs = (np.tanh(es/delta) + 1.0)/2. # smearing
    nw = len(es)
    print("Doing")
    return rkky_loop(es,phis,fs,delta)/(nw**2) # return RKKY interaction



@jit(nopython=True)
def rkky_loop(es,phis,fs,delta):
    """Summation for the RKKY interaction"""
    n = len(es) # number of states
    out = 0.0j # output
    delta2 = delta**2
    for i in range(n): # loop
        ei,phii,fi = es[i],phis[i],fs[i]
        for j in range(n): # loop
            ej,phij,fj = es[j],phis[j],fs[j]
            phi = phii/phij # relative phase
            out += phi*1./(ei-ej+1j*delta2)*(fi-fj)
    return out.real


def rkky_pm(h,nk=20,delta=1e-1,**kwargs):
    """Compute the RKKY using a poor man convolution of the k-DOS"""
    raise
    if h.dimensionality!=2: raise
    qs0 = h.geometry.get_kmesh(nk=nk)
    qs = np.array([fR(q) for q in qs0]) # convert
    from ..fermisurface import fermi_surface_generator
    energies = [0.0] # above and below Fermi
    es,ks,ds = fermi_surface_generator(h,reciprocal=False,info=info,
            energies=energies,delta=delta,
            nk=nk,**kwargs)
    # we now have the energies, k-points and DOS, lets do a convolution
    d = poor_man_qpi_single_energy(ks,ds[:,i],qs) # parallel function
    out = parallel.pcall(fp,range(len(es))) # compute in parallel



