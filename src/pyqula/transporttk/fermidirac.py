import numpy as np

def fermidirac(es,temp=0.):
    """Return a Fermi Dirac distribution at a certain temperature"""
    if temp==0.: return (1.0 - np.sign(es))/2. # zero temperature
    else: return 1./(1. + np.exp(es/temp))


def dFDdT(es,temp=0.):
    """Return a Fermi Dirac distribution at a certain temperature"""
    if temp==0.:
        raise ValueError("the temperature derivative of the Fermi-Dirac "
                "distribution diverges at zero temperature, so temp must be "
                "finite")
    else: return 1./temp*np.exp(es/temp)/(1. + np.exp(es/temp))**2
