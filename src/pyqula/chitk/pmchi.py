import numpy as np
from numba import jit
from scipy.signal import hilbert

# compute a response function using ultrafast routines, by
# ignoring matrix elements

def pmchi(h,**kwargs):
    return pmimchi(h,**kwargs)
    return omega,1j*np.conjugate(hilbert(out))

def pmimchi(h,energies=np.linspace(-3,3,300),delta=1e-2,**kwargs):
    """Compute the chi charge-charge response function
    by doing a selfconvolution of the density of states"""
    (es,dos) = h.get_dos(delta=delta/2.,energies=energies,
            **kwargs) # compute energies and DOS
    omega,out = chi_from_dos_jit(es,dos,delta=delta,omega=energies)
    return omega,out




@jit(nopython=True)
def chi_from_dos_jit(es,dos,T=1e-9,delta=1e-3,omega=None):
    """Compute the response function"""
    ne = len(es) # initial mesh of energies
    out  = omega*0.0j # initialize
    for ii in range(ne): # loop over energies
        ei = es[ii] # this energy
        if ei<0.0: oi = 1.0 # first occupation
        else: oi = 0.0 # occupation
        for jj in range(ne): # second loop over energies
            ej = es[jj] # energy
            if ej<0.0: oj = 1.0 # second occupation
            else: oj = 0.0
            fac = oj-oi # occupation factor
            fac = fac*dos[ii]*dos[jj] # scale by the DOS
            out = out - fac*(1./(ei-ej - omega + 1j*delta))
    return omega,out/ne # return






