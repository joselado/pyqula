import numpy as np
from .. import algebra
from numba import jit

def rkky(h,R=[0,0,0],ii=0,jj=0,**kwargs):
    get = rkky_generator(h,**kwargs) # get generator
    return get(R,ii,jj) # evaluate once


def rkky_generator(h,delta=None,nk=100):
    """Compute the RKKY interaction for the ground state
    - R location of the unit cell
    - delta analytic continuation
    """
    # R should be smaller than nk
    h = h.copy()
    h.remove_spin() # remove spin degree of freedom
    es,ws,ks = h.get_eigenvectors(kpoints=True,nk=nk) # get eigenvectors
    if delta is None: delta = 1./nk
    def get(R,ii,jj):
        d1s = np.array([w[ii] for w in ws]) # densities
        d2s = np.array([w[jj] for w in ws]) # densities
        R = np.array(R) # convert to array
        phis = np.array([h.geometry.bloch_phase(R,k) for k in ks]) # phase
        fs = (np.tanh(es/delta) + 1.0)/2. # smearing
        nw = len(es)/len(h.geometry.r)
        return rkky_loop(es,phis,fs,d1s,d2s,delta)/(nw**2) # return RKKY int
    return get



@jit(nopython=True)
def rkky_loop(es,phis,fs,d1s,d2s,delta):
    """Summation for the RKKY interaction"""
    n = len(es) # number of states
    out = 0.0j # output
    delta2 = delta**2
    for i in range(n): # loop
        ei,phii,fi = es[i],phis[i],fs[i]
        for j in range(n): # loop
            ej,phij,fj = es[j],phis[j],fs[j]
            phi = phii/phij # relative phase
            dd = d1s[i]*np.conjugate(d1s[j])
            dd *= np.conjugate(d2s[i])*d2s[j]
            out += phi*1./(ei-ej+1j*delta2)*(fi-fj)*dd
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



