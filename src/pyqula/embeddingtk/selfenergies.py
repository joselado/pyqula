import numpy as np


# routines to create selfenergies for different types of leads

class Selfenergy():
    """Self-energy object"""
    def __init__(self,m=None): # pass
        self.m = m
    def __call__(self,**kwargs):
        return self.m



def metal(H,potential):
    """Return the selfenergy of a metallic lead"""
    # create a selfenergy in a single edge (the left one) to kill magnetism
    hs = H.copy()*0.
    hs.add_onsite(lambda r: np.abs(potential(r))) # absolute value
    selfe = -1j*hs.intra # selfenergy on edge, beware of minus sign
    return selfe


def semimetal(*args,zero=0.):
    selfe = metal(*args) # selfenergy of a metal
    def f(energy=0.,**kwargs):
        return selfe*np.abs(energy-zero) # selfenergy of a semimetal
    return f



def get_selfenergy_from_potential(*args,mode="metal",**kwargs):
    if mode=="metal": return metal(*args,**kwargs)
    elif mode=="semimetal": return semimetal(*args,**kwargs)
    else: raise


