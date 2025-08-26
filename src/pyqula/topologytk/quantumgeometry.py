from scipy.signal import hilbert
import numpy as np

def get_QG_dE(h,k=[0.,0.,0.],delta=1e-1,**kwargs):
    """Compute the dqg/dE (k,E), the derivative of the 
    quantum geometry with respect to the energy
    by using a Hilbert transform"""
    es = np.linspace(-6,6,int(50/delta)) # energies
    from .green import dOmega_dE_generator
    bef = dOmega_dE_generator(h,delta=delta,**kwargs) # generator of the Berry curvature
    bes = np.array([bef(k=k,e=e) for e in es]) # Berry curvature at the different energies
    yz = hilbert(bes) # Hilbert transform
    qge = np.imag(yz) # pick the imaginary component
    return es,qge,bes # return energy, dQG/dE and dOmega/dE


def get_QG(h,delta=1e-1,**kwargs):
    """Compute the quantum geometry (and Berry curvature) by summing all the
    accupied states"""
    es,qge,bes = get_QG_dE(h,delta=delta,**kwargs) # return dQG/dE and dOmega/dE
    dE = es[1]-es[0] # dE
    weight1 = (1.-np.tanh(es/delta))/2. # weight function (soft step)
    weight2 = delta/(es**2 + delta**2)*1./np.pi # weight function (soft delta)
    qgew = qge*weight2 # at fermi with a smearing
    besw = bes*weight1 # below fermi with a smearing
    return np.trapz(qgew,dx=dE), np.trapz(besw,dx=dE) # return the integral


def get_QG_kpath(h,kpath=None,nk=100,**kwargs):
    from ..import klist
    kpath = klist.get_kpath(h.geometry,kpath=kpath,nk=nk) # take default kpath
    out = np.array([get_QG(h,k=k,**kwargs) for k in kpath]) # compute all
    qg = out[:,0] # quantum geometry
    be = out[:,1] # Berry curvature
    inds = np.array(range(len(qg))) # index counter
    return inds,qg,be
    



