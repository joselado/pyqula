import numpy as np


def radial_vector_decay(**kwargs):
    from .profiles import radial_decay
    f = radial_decay(**kwargs) # function for the radial decay
    def fs(r,dr):
        r0 = np.sqrt(r.dot(r))
        dr0 = np.sqrt(dr.dot(dr))
        if r0<1e-4 or dr0<1e-4: return 1.0
        ur0,udr0 = r/r0,dr/dr0
        udr = np.abs(ur0.dot(udr0))
        return 1.0*(1.-udr) + f(r)*udr # return
    return fs
